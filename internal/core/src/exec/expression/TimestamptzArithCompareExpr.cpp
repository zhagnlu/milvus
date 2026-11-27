#include "TimestamptzArithCompareExpr.h"
#include "exec/expression/TimestampArithmetic.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <memory>
#include <variant>

#include "bitset/bitset.h"
#include "common/EasyAssert.h"
#include "common/Types.h"
#include "common/Vector.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/Expr.h"
#include "expr/ITypeExpr.h"
#include "pb/plan.pb.h"
#include "storage/PrefetchThreadPool.h"

namespace milvus {
namespace exec {

namespace {

using scalar::ApplyTimestampInterval;

bool
CompareTimestamp(int64_t lhs, int64_t rhs, proto::plan::OpType compare_op) {
    switch (compare_op) {
        case proto::plan::OpType::Equal:
            return lhs == rhs;
        case proto::plan::OpType::NotEqual:
            return lhs != rhs;
        case proto::plan::OpType::GreaterThan:
            return lhs > rhs;
        case proto::plan::OpType::GreaterEqual:
            return lhs >= rhs;
        case proto::plan::OpType::LessThan:
            return lhs < rhs;
        case proto::plan::OpType::LessEqual:
            return lhs <= rhs;
        default:
            ThrowInfo(UnexpectedError,
                      "unsupported compare op for "
                      "timestamptz_arith_compare_expr: {}",
                      compare_op);
    }
}

bool
EvaluateTimestamp(int64_t current_ts_us,
                  proto::plan::ArithOpType arith_op,
                  const proto::plan::Interval& interval,
                  proto::plan::OpType compare_op,
                  int64_t compare_us) {
    const auto final_us =
        ApplyTimestampInterval(current_ts_us, arith_op, interval);
    return CompareTimestamp(final_us, compare_us, compare_op);
}

}  // namespace

std::string
PhyTimestamptzArithCompareExpr::ToString() const {
    return expr_->ToString();
}

void
PhyTimestamptzArithCompareExpr::DetermineExecPath() {
    // Prefer field data when it is present. TIMESTAMPTZ calendar arithmetic
    // cannot be transformed into a scalar-index range query, and keeping the
    // raw-data path also preserves its batch cursor when another raw predicate
    // is evaluated alongside it.
    if (num_data_chunk_ > 0 || active_count_ == 0) {
        exec_path_ = ExprExecPath::RawData;
        return;
    }

    auto req = MakeIndexRequirement(RequiredReader::ValueLookup);
    req.value_type = DataType::TIMESTAMPTZ;
    if (SelectAndCreateIndexReader(req)) {
        return;
    }

    ThrowInfo(UnexpectedError,
              "TIMESTAMPTZ arithmetic comparison requires field data or a "
              "scalar index with raw data for field {}",
              field_id_.get());
}

void
PhyTimestamptzArithCompareExpr::Eval(EvalCtx& context, VectorPtr& result) {
    WaitPrefetch();
    auto input = context.get_offset_input();
    SetHasOffsetInput((input != nullptr));
    result = ExecCompareVisitorImpl<int64_t>(input);
}

template <typename T>
VectorPtr
PhyTimestamptzArithCompareExpr::ExecCompareVisitorImpl(OffsetVector* input) {
    if (exec_path_ == ExprExecPath::ScalarIndex) {
        return ExecCompareVisitorImplForIndex<T>(input);
    }
    return ExecCompareVisitorImplForAll<T>(input);
}

template <typename T>
VectorPtr
PhyTimestamptzArithCompareExpr::ExecCompareVisitorImplForIndex(
    OffsetVector* input) {
    if (!arg_inited_) {
        interval_ = expr_->interval_;
        compare_value_.SetValue<T>(expr_->compare_value_);
        arg_inited_ = true;
    }

    const auto arith_op = expr_->arith_op_;
    const auto compare_op = expr_->compare_op_;
    const auto compare_value = compare_value_.GetValue<T>();
    const auto interval = interval_;
    const int64_t real_batch_size =
        input != nullptr ? input->size() : GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }
    auto exec_index =
        [ arith_op,
          compare_op ]<FilterType filter_type = FilterType::sequential>(
            const index::IScalarValueReader<T>* index_ptr,
            int64_t size,
            T compare_value,
            const proto::plan::Interval& interval,
            const int32_t* offsets = nullptr) {
        TargetBitmap result(size, false);
        for (int64_t i = 0; i < size; ++i) {
            size_t offset = i;
            if constexpr (filter_type == FilterType::random) {
                offset = offsets == nullptr ? i : offsets[i];
            }
            auto raw = index_ptr->Lookup(offset);
            if (raw.has_value()) {
                result[i] = EvaluateTimestamp(
                    raw.value(), arith_op, interval, compare_op, compare_value);
            }
        }
        return result;
    };

    VectorPtr result;
    if (input != nullptr) {
        result = ProcessValueIndexByOffsets<T>(
            exec_index, input, compare_value, interval);
    } else {
        result = ProcessValueIndex<T>(exec_index, compare_value, interval);
    }
    AssertInfo(result != nullptr && result->size() == real_batch_size,
               "internal error: expr processed rows {} not equal "
               "expect batch size {}",
               result == nullptr ? 0 : result->size(),
               real_batch_size);
    return result;
}

template <typename T>
VectorPtr
PhyTimestamptzArithCompareExpr::ExecCompareVisitorImplForAll(
    OffsetVector* input) {
    if (!arg_inited_) {
        interval_ = expr_->interval_;
        compare_value_.SetValue<T>(expr_->compare_value_);
        arg_inited_ = true;
    }

    auto arith_op = expr_->arith_op_;
    auto compare_op = expr_->compare_op_;
    auto compare_value = this->compare_value_.GetValue<T>();
    auto interval = interval_;

    auto real_batch_size =
        has_offset_input_ ? input->size() : GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto res_vec =
        std::make_shared<ColumnVector>(TargetBitmap(real_batch_size, false),
                                       TargetBitmap(real_batch_size, true));

    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
    TargetBitmapView valid_res(res_vec->GetValidRawData(), real_batch_size);
    auto exec_sub_batch =
        [ arith_op,
          compare_op ]<FilterType filter_type = FilterType::sequential>(
            const T* data,
            ValidityView valid_data,
            const int32_t* offsets,
            const int size,
            TargetBitmapView res,
            TargetBitmapView valid_res,
            T compare_value,
            proto::plan::Interval interval) {
        if (data == nullptr) {
            return;
        }
        const int64_t compare_us = compare_value;
        for (int i = 0; i < size; ++i) {
            int32_t offset = i;
            if constexpr (filter_type == FilterType::random) {
                offset = offsets == nullptr ? i : offsets[i];
            }
            if (valid_data && !valid_data[offset]) {
                // NULL never matches, under either polarity (three-valued
                // logic); do not evaluate the storage placeholder value.
                res[i] = valid_res[i] = false;
                continue;
            }
            res[i] = EvaluateTimestamp(
                data[offset], arith_op, interval, compare_op, compare_us);
        }
    };
    int64_t processed_size;
    if (has_offset_input_) {
        processed_size = ProcessDataByOffsets<T>(exec_sub_batch,
                                                 std::nullptr_t{},
                                                 input,
                                                 res,
                                                 valid_res,
                                                 compare_value,
                                                 interval);
    } else {
        processed_size = ProcessDataChunks<T>(exec_sub_batch,
                                              std::nullptr_t{},
                                              res,
                                              valid_res,
                                              compare_value,
                                              interval);
    }
    AssertInfo(processed_size == real_batch_size,
               "internal error: expr processed rows {} not equal "
               "expect batch size {}",
               processed_size,
               real_batch_size);

    return res_vec;
}

}  // namespace exec
}  // namespace milvus
