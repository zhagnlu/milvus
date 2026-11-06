// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Build with run_delete_batch_benchmark.py, which extracts the current production
// reorder and galloping kernels into delete_kernels.h. Measures preparation and
// lookup ONLY; the callback hashes matches instead of inserting into a skiplist.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <variant>
#include <vector>

#include "common/GallopingSearch.h"

using namespace milvus;
using PkType = std::variant<std::monostate, int64_t, std::string>;
using Timestamp = uint64_t;
struct SegOffset {
    int64_t n;
    explicit SegOffset(int64_t value) : n(value) {
    }
};

#include "delete_kernels.h"

namespace {
using Clock = std::chrono::steady_clock;

std::vector<std::string>
Split(const std::string& input) {
    std::stringstream stream(input);
    std::vector<std::string> result;
    for (std::string value; std::getline(stream, value, ',');) {
        result.push_back(value);
    }
    return result;
}

std::string
StringKey(uint64_t value) {
    std::string key(16, '0');
    constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < 16; ++i) {
        key[15 - i] = hex[value & 15];
        value >>= 4;
    }
    return key;
}

template <bool Strings>
struct Column {
    size_t size;
    std::vector<int64_t> ints;
    // Match StringChunk's contiguous payload plus uint32 offsets; no vector of
    // individually allocated std::strings in the indexed column.
    std::vector<char> chars;
    std::vector<uint32_t> offsets;
    std::vector<Timestamp> timestamps;

    explicit Column(size_t rows) : size(rows), timestamps(rows) {
        if constexpr (Strings) {
            chars.resize(rows * 16);
            offsets.resize(rows + 1);
            for (size_t i = 0; i < rows; ++i) {
                const auto key = StringKey(i * 2);
                std::copy(key.begin(), key.end(), chars.begin() + i * 16);
                offsets[i] = i * 16;
            }
            offsets[rows] = rows * 16;
        } else {
            ints.resize(rows);
            for (size_t i = 0; i < rows; ++i) {
                ints[i] = i * 2;
            }
        }
        for (size_t i = 0; i < rows; ++i) {
            timestamps[i] = i % 7;
        }
    }

    auto
    value(size_t i) const {
        if constexpr (Strings) {
            return std::string_view(chars.data() + offsets[i],
                                    offsets[i + 1] - offsets[i]);
        } else {
            return ints[i];
        }
    }

    auto
    key(const PkType& pk) const {
        if constexpr (Strings) {
            return std::string_view(std::get<std::string>(pk));
        } else {
            return std::get<int64_t>(pk);
        }
    }

    PkType
    pk(int64_t k) const {
        if constexpr (Strings) {
            return StringKey(k);
        } else {
            return k;
        }
    }

    // Same old lookup algorithms: std::lower_bound for int64 and
    // StringChunk::binary_search_string's first-equal search for VARCHAR.
    size_t
    binary(size_t base, size_t end, const PkType& pk) const {
        const auto target = key(pk);
        if constexpr (!Strings) {
            return std::lower_bound(
                       ints.begin() + base, ints.begin() + end, target) -
                   ints.begin();
        } else {
            int64_t left = base, right = static_cast<int64_t>(end) - 1;
            size_t result = end;
            while (left <= right) {
                const int64_t mid = left + (right - left) / 2;
                const auto cmp = value(mid).compare(target);
                if (cmp == 0) {
                    result = mid;
                    right = mid - 1;
                } else if (cmp < 0) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return result;
        }
    }
};

struct Batch {
    std::vector<PkType> pks;
    std::vector<Timestamp> timestamps;
    uint64_t expected_hash = 0;
    size_t expected_count = 0;
};

uint64_t
Hash(uint64_t row, Timestamp ts) {
    return (row + 1) * 0x9e3779b97f4a7c15ULL + ts * 0xbf58476d1ce4e5b9ULL;
}

template <bool Strings>
Batch
MakeBatch(const Column<Strings>& column,
          size_t count,
          const std::string& shape,
          bool same_ts,
          uint64_t seed) {
    std::mt19937_64 rng(seed);
    Batch batch;
    std::vector<int64_t> keys(count);
    if (shape == "uniform_unique" && count > column.size) {
        throw std::runtime_error("uniform_unique batch exceeds segment rows");
    }
    std::unordered_set<size_t> sampled_rows;
    if (shape == "uniform_unique") {
        sampled_rows.reserve(count);
    }
    const size_t start =
        rng() % (column.size - std::min(count, column.size) + 1);
    std::vector<size_t> repeated(
        shape == "repeated_half" ? std::max(size_t{1}, count / 2) : size_t{16});
    for (auto& row : repeated) {
        row = rng() % column.size;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t row;
        if (shape == "uniform_unique") {
            // Floyd sampling: uniform subset without replacement, O(count).
            const size_t upper = column.size - count + i;
            row = rng() % (upper + 1);
            if (!sampled_rows.insert(row).second) {
                row = upper;
                sampled_rows.insert(row);
            }
        } else if (shape == "dense" || shape == "ordered") {
            row = (start + i) % column.size;
        } else if (shape == "repeated" || shape == "repeated_half") {
            row = repeated[i % repeated.size()];
        } else {
            row = rng() % column.size;
        }
        keys[i] = row * 2 + (shape == "miss" ? 1 : 0);
    }
    if (shape == "ordered") {
        std::sort(keys.begin(), keys.end());
    } else {
        std::shuffle(keys.begin(), keys.end(), rng);
    }
    for (const auto k : keys) {
        const Timestamp ts = same_ts ? 100 : 1 + rng() % 100;
        batch.pks.push_back(column.pk(k));
        batch.timestamps.push_back(ts);
        // Independent O(M) oracle using the generated column's arithmetic keys.
        if (k % 2 == 0 && column.timestamps[k / 2] < ts) {
            batch.expected_hash += Hash(k / 2, ts);
            ++batch.expected_count;
        }
    }
    return batch;
}

struct Timing {
    double prepare = 0;
    double lookup = 0;
    double
    total() const {
        return prepare + lookup;
    }
};

template <bool Strings>
Timing
Run(const Column<Strings>& column,
    const Batch& batch,
    const std::vector<size_t>& chunk_offsets,
    bool load,
    bool sorted) {
    // Parsing IDs into PkType is common to both paths. Model it with a copy
    // OUTSIDE the measured region; timestamps are copied only as needed below.
    auto pks = batch.pks;
    const auto begin = Clock::now();
    std::vector<Timestamp> timestamps;
    const Timestamp* ts = batch.timestamps.data();
    if (!load || sorted) {
        timestamps = batch.timestamps;
        ts = timestamps.data();
        ReorderDeleteBatch(pks, timestamps, [&](size_t a, size_t b) {
            if (sorted) {
                if (load) {
                    return std::tie(pks[a], timestamps[a], a) <
                           std::tie(pks[b], timestamps[b], b);
                }
                return std::tie(pks[a], timestamps[a]) <
                       std::tie(pks[b], timestamps[b]);
            }
            return std::tie(timestamps[a], pks[a]) <
                   std::tie(timestamps[b], pks[b]);
        });
    } else {
        // Replay retains master's earliest-delete ordering in the fallback.
        ts = SortByTimestamp(pks, ts, timestamps);
    }
    const auto prepared = Clock::now();
    uint64_t hash = 0;
    size_t hits = 0;
    const std::function<Timestamp(size_t)> get_timestamp = [&](size_t i) {
        return ts[i];
    };
    const auto read_ts = [&](int64_t row) { return column.timestamps[row]; };
    const std::function<void(SegOffset, Timestamp)> callback =
        [&](SegOffset offset, Timestamp timestamp) {
            hash += Hash(offset.n, timestamp);
            ++hits;
        };
    if (sorted) {
        if (!std::is_sorted(pks.begin(), pks.end())) {
            throw std::runtime_error("reorder did not sort PKs");
        }
        SortedLookup(
            pks,
            [&](auto search_chunk) {
                for (size_t c = 0; c + 1 < chunk_offsets.size(); ++c) {
                    const size_t base = chunk_offsets[c];
                    const size_t end = chunk_offsets[c + 1];
                    search_chunk(
                        end - base,
                        base,
                        [&](size_t i) { return column.value(base + i); },
                        [&](size_t i) { return column.key(pks[i]); });
                }
            },
            get_timestamp,
            read_ts,
            callback);
    } else {
        for (size_t c = 0; c + 1 < chunk_offsets.size(); ++c) {
            const size_t base = chunk_offsets[c];
            const size_t end = chunk_offsets[c + 1];
            for (size_t j = 0; j < pks.size(); ++j) {
                const auto target = column.key(pks[j]);
                const auto timestamp = get_timestamp(j);
                auto offset = column.binary(base, end, pks[j]);
                for (; offset < end && column.value(offset) == target;
                     ++offset) {
                    if (read_ts(offset) < timestamp) {
                        callback(SegOffset(offset), timestamp);
                    }
                }
            }
        }
    }
    const auto finished = Clock::now();
    if (hash != batch.expected_hash || hits != batch.expected_count) {
        throw std::runtime_error("lookup differs from independent oracle");
    }
    return {
        std::chrono::duration<double, std::micro>(prepared - begin).count(),
        std::chrono::duration<double, std::micro>(finished - prepared).count()};
}

double
Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <bool Strings>
void
Sweep(const Column<Strings>& column,
      const std::vector<std::string>& sizes,
      const std::vector<std::string>& chunk_counts,
      const std::vector<std::string>& shapes,
      const std::vector<std::string>& modes,
      size_t rounds,
      uint64_t seed,
      const std::vector<std::string>& chunk_rows) {
    for (const auto& chunk_count : chunk_counts) {
        const auto chunks = std::stoull(chunk_count);
        std::vector<size_t> chunk_offsets{0};
        if (chunk_rows.empty()) {
            for (size_t c = 0; c < chunks; ++c) {
                chunk_offsets.push_back((c + 1) * column.size / chunks);
            }
        } else {
            for (const auto& rows : chunk_rows) {
                chunk_offsets.push_back(chunk_offsets.back() +
                                        std::stoull(rows));
            }
            if (chunk_offsets.size() != chunks + 1 ||
                chunk_offsets.back() != column.size) {
                throw std::runtime_error("chunk rows do not match column");
            }
        }
        for (const auto& shape : shapes) {
            for (const auto& mode : modes) {
                for (const auto& size : sizes) {
                    const size_t n = std::stoull(size);
                    const size_t per_round = std::max(
                        size_t{1}, std::min(size_t{8}, size_t{4096} / n));
                    std::vector<Batch> batches;
                    for (size_t i = 0; i < per_round * rounds; ++i) {
                        batches.push_back(MakeBatch(column,
                                                    n,
                                                    shape,
                                                    mode == "stream_same",
                                                    seed + i * 997 + n));
                    }
                    // Symmetric untimed warmup. Timed rounds rotate independent
                    // batches and alternate path order, avoiding one hot input.
                    Run(column,
                        batches[0],
                        chunk_offsets,
                        mode == "load",
                        false);
                    Run(column,
                        batches[0],
                        chunk_offsets,
                        mode == "load",
                        true);
                    std::vector<double> old_total, new_total, old_prepare,
                        new_prepare, old_lookup, new_lookup, ratios;
                    for (size_t round = 0; round < rounds; ++round) {
                        Timing old{}, now{};
                        for (size_t i = 0; i < per_round; ++i) {
                            const auto& batch = batches[round * per_round + i];
                            for (size_t path = 0; path < 2; ++path) {
                                const bool sorted = (path + round + i) % 2;
                                const auto t = Run(column,
                                                   batch,
                                                   chunk_offsets,
                                                   mode == "load",
                                                   sorted);
                                auto& sum = sorted ? now : old;
                                sum.prepare += t.prepare / per_round;
                                sum.lookup += t.lookup / per_round;
                            }
                        }
                        old_total.push_back(old.total());
                        new_total.push_back(now.total());
                        old_prepare.push_back(old.prepare);
                        new_prepare.push_back(now.prepare);
                        old_lookup.push_back(old.lookup);
                        new_lookup.push_back(now.lookup);
                        ratios.push_back(old.total() / now.total());
                    }
                    std::printf(
                        "%s,%zu,%zu,%s,%s,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%."
                        "3f,%.4f,%.4f,%.4f\n",
                        Strings ? "varchar16" : "int64",
                        column.size,
                        static_cast<size_t>(chunks),
                        shape.c_str(),
                        mode.c_str(),
                        n,
                        rounds * per_round,
                        Median(old_prepare),
                        Median(new_prepare),
                        Median(old_lookup),
                        Median(new_lookup),
                        Median(old_total),
                        Median(new_total),
                        Median(ratios),
                        *std::min_element(ratios.begin(), ratios.end()),
                        *std::max_element(ratios.begin(), ratios.end()));
                    std::fflush(stdout);
                }
            }
        }
    }
}
}  // namespace

int
main(int argc, char** argv) {
    if (argc != 9 && argc != 10) {
        std::cerr << "Use run_delete_batch_benchmark.py\n";
        return 1;
    }
    std::puts(
        "type,rows,chunks,shape,mode,batch,samples,old_prepare_us,new_prepare_"
        "us,old_lookup_us,new_lookup_us,old_total_us,new_total_us,speedup,"
        "round_min_speedup,round_max_speedup");
    const auto chunk_rows =
        argc == 10 ? Split(argv[9]) : std::vector<std::string>{};
    for (const auto& type : Split(argv[4])) {
        for (const auto& rows : Split(argv[1])) {
            std::cerr << "Measuring " << type << " rows=" << rows << '\n';
            if (type == "int64") {
                Column<false> column(std::stoull(rows));
                Sweep(column,
                      Split(argv[2]),
                      Split(argv[3]),
                      Split(argv[5]),
                      Split(argv[6]),
                      std::stoull(argv[7]),
                      std::stoull(argv[8]),
                      chunk_rows);
            } else {
                Column<true> column(std::stoull(rows));
                Sweep(column,
                      Split(argv[2]),
                      Split(argv[3]),
                      Split(argv[5]),
                      Split(argv[6]),
                      std::stoull(argv[7]),
                      std::stoull(argv[8]),
                      chunk_rows);
            }
        }
    }
}
