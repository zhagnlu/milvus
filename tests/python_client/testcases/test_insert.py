from ssl import ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY
import threading

import numpy as np
import pandas as pd
import random
import pytest
from pymilvus import Index, DataType
from pymilvus.exceptions import MilvusException

from base.client_base import TestcaseBase
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks

prefix = "insert"
exp_name = "name"
exp_schema = "schema"
exp_num = "num_entities"
exp_primary = "primary"
default_schema = cf.gen_default_collection_schema()
default_binary_schema = cf.gen_default_binary_collection_schema()
default_index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
default_binary_index_params = {"index_type": "BIN_IVF_FLAT", "metric_type": "JACCARD", "params": {"nlist": 64}}
default_search_exp = "int64 >= 0"


class TestInsertParams(TestcaseBase):
    """ Test case of Insert interface """

    @pytest.fixture(scope="function", params=ct.get_invalid_strs)
    def get_non_data_type(self, request):
        if isinstance(request.param, list) or request.param is None:
            pytest.skip("list and None type is valid data type")
        yield request.param

    @pytest.fixture(scope="module", params=ct.get_invalid_strs)
    def get_invalid_field_name(self, request):
        if isinstance(request.param, (list, dict)):
            pytest.skip()
        yield request.param

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_dataframe_data(self):
        """
        target: test insert DataFrame data
        method: 1.create collection
                2.insert dataframe data
        expected: assert num entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = cf.gen_default_dataframe_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=df)
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_list_data(self):
        """
        target: test insert list-like data
        method: 1.create 2.insert list data
        expected: assert num entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        data = cf.gen_default_list_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=data)
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == data[0]
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_non_data_type(self, get_non_data_type):
        """
        target: test insert with non-dataframe, non-list data
        method: insert with data (non-dataframe and non-list type)
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        error = {ct.err_code: 0, ct.err_msg: "Data type is not support"}
        collection_w.insert(data=get_non_data_type, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("data", [[], pd.DataFrame()])
    def test_insert_empty_data(self, data):
        """
        target: test insert empty data
        method: insert empty
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        error = {ct.err_code: 0, ct.err_msg: "The data fields number is not match with schema"}
        collection_w.insert(data=data, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_dataframe_only_columns(self):
        """
        target: test insert with dataframe just columns
        method: dataframe just have columns
        expected: num entities is zero
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        columns = [ct.default_int64_field_name, ct.default_float_vec_field_name]
        df = pd.DataFrame(columns=columns)
        error = {ct.err_code: 0, ct.err_msg: "Cannot infer schema from empty dataframe"}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_empty_field_name_dataframe(self):
        """
        target: test insert empty field name df
        method: dataframe with empty column
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = cf.gen_default_dataframe_data(10)
        df.rename(columns={ct.default_int64_field_name: ' '}, inplace=True)
        error = {ct.err_code: 0, ct.err_msg: "The types of schema and data do not match"}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_invalid_field_name_dataframe(self, get_invalid_field_name):
        """
        target: test insert with invalid dataframe data
        method: insert with invalid field name dataframe
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = cf.gen_default_dataframe_data(10)
        df.rename(columns={ct.default_int64_field_name: get_invalid_field_name}, inplace=True)
        error = {ct.err_code: 0, ct.err_msg: "The types of schema and data do not match"}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    def test_insert_dataframe_index(self):
        """
        target: test insert dataframe with index
        method: insert dataframe with index
        expected: todo
        """
        pass

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_none(self):
        """
        target: test insert None
        method: data is None
        expected: return successfully with zero results
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        mutation_res, _ = collection_w.insert(data=None)
        assert mutation_res.insert_count == 0
        assert len(mutation_res.primary_keys) == 0
        assert collection_w.is_empty
        assert collection_w.num_entities == 0

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_numpy_data(self):
        """
        target: test insert numpy.ndarray data
        method: 1.create by schema 2.insert data
        expected: assert num_entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        data = cf.gen_numpy_data(nb=10)
        collection_w.insert(data=data)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_binary_dataframe(self):
        """
        target: test insert binary dataframe
        method: 1. create by schema 2. insert dataframe
        expected: assert num_entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name, schema=default_binary_schema)
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=df)
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_binary_data(self):
        """
        target: test insert list-like binary data
        method: 1. create by schema 2. insert data
        expected: assert num_entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name, schema=default_binary_schema)
        data, _ = cf.gen_default_binary_list_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=data)
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == data[0]
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_single(self):
        """
        target: test insert single
        method: insert one entity
        expected: verify num
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        data = cf.gen_default_list_data(nb=1)
        mutation_res, _ = collection_w.insert(data=data)
        assert mutation_res.insert_count == 1
        assert mutation_res.primary_keys == data[0]
        assert collection_w.num_entities == 1

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_dim_not_match(self):
        """
        target: test insert with not match dim
        method: insert data dim not equal to schema dim
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        dim = 129
        df = cf.gen_default_dataframe_data(ct.default_nb, dim=dim)
        error = {ct.err_code: 1,
                 ct.err_msg: f'Collection field dim is {ct.default_dim}, but entities field dim is {dim}'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_binary_dim_not_match(self):
        """
        target: test insert binary with dim not match
        method: insert binary data dim not equal to schema
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name, schema=default_binary_schema)
        dim = 120
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb, dim=dim)
        error = {ct.err_code: 1,
                 ct.err_msg: f'Collection field dim is {ct.default_dim}, but entities field dim is {dim}'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_field_name_not_match(self):
        """
        target: test insert field name not match
        method: data field name not match schema
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = cf.gen_default_dataframe_data(10)
        df.rename(columns={ct.default_float_field_name: "int"}, inplace=True)
        error = {ct.err_code: 0, ct.err_msg: 'The types of schema and data do not match'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_field_value_not_match(self):
        """
        target: test insert data value not match
        method: insert data value type not match schema
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 10
        df = cf.gen_default_dataframe_data(nb)
        new_float_value = pd.Series(data=[float(i) for i in range(nb)], dtype="float64")
        df.iloc[:, 1] = new_float_value
        error = {ct.err_code: 0, ct.err_msg: 'The types of schema and data do not match'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_value_less(self):
        """
        target: test insert value less than other
        method: int field value less than vec-field value
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 10
        int_values = [i for i in range(nb - 1)]
        float_values = [np.float32(i) for i in range(nb)]
        float_vec_values = cf.gen_vectors(nb, ct.default_dim)
        data = [int_values, float_values, float_vec_values]
        error = {ct.err_code: 0, ct.err_msg: 'Arrays must all be same length.'}
        collection_w.insert(data=data, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_vector_value_less(self):
        """
        target: test insert vector value less than other
        method: vec field value less than int field
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 10
        int_values = [i for i in range(nb)]
        float_values = [np.float32(i) for i in range(nb)]
        float_vec_values = cf.gen_vectors(nb - 1, ct.default_dim)
        data = [int_values, float_values, float_vec_values]
        error = {ct.err_code: 0, ct.err_msg: 'Arrays must all be same length.'}
        collection_w.insert(data=data, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_fields_more(self):
        """
        target: test insert with fields more
        method: field more than schema fields
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = cf.gen_default_dataframe_data(ct.default_nb)
        new_values = [i for i in range(ct.default_nb)]
        df.insert(3, 'new', new_values)
        error = {ct.err_code: 0, ct.err_msg: 'The data fields number is not match with schema.'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_fields_less(self):
        """
        target: test insert with fields less
        method: fields less than schema fields
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = cf.gen_default_dataframe_data(ct.default_nb)
        df.drop(ct.default_float_vec_field_name, axis=1, inplace=True)
        error = {ct.err_code: 0, ct.err_msg: 'The data fields number is not match with schema.'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_list_order_inconsistent_schema(self):
        """
        target: test insert data fields order inconsistent with schema
        method: insert list data, data fields order inconsistent with schema
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 10
        int_values = [i for i in range(nb)]
        float_values = [np.float32(i) for i in range(nb)]
        float_vec_values = cf.gen_vectors(nb, ct.default_dim)
        data = [float_values, int_values, float_vec_values]
        error = {ct.err_code: 0, ct.err_msg: 'The types of schema and data do not match'}
        collection_w.insert(data=data, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_dataframe_order_inconsistent_schema(self):
        """
        target: test insert with dataframe fields inconsistent with schema
        method: insert dataframe, and fields order inconsistent with schema
        expected: assert num entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 10
        int_values = pd.Series(data=[i for i in range(nb)])
        float_values = pd.Series(data=[float(i) for i in range(nb)], dtype="float32")
        float_vec_values = cf.gen_vectors(nb, ct.default_dim)
        df = pd.DataFrame({
            ct.default_float_field_name: float_values,
            ct.default_float_vec_field_name: float_vec_values,
            ct.default_int64_field_name: int_values
        })
        error = {ct.err_code: 0, ct.err_msg: 'The types of schema and data do not match'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_inconsistent_data(self):
        """
        target: test insert with inconsistent data
        method: insert with data that same field has different type data
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        data = cf.gen_default_list_data(nb=100)
        data[0][1] = 1.0
        error = {ct.err_code: 0, ct.err_msg: "The data in the same column must be of the same type"}
        collection_w.insert(data, check_task=CheckTasks.err_res, check_items=error)


class TestInsertOperation(TestcaseBase):
    """
    ******************************************************************
      The following cases are used to test insert interface operations
    ******************************************************************
    """

    @pytest.fixture(scope="function", params=[8, 4096])
    def dim(self, request):
        yield request.param

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_without_connection(self):
        """
        target: test insert without connection
        method: insert after remove connection
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        self.connection_wrap.remove_connection(ct.default_alias)
        res_list, _ = self.connection_wrap.list_connections()
        assert ct.default_alias not in res_list
        data = cf.gen_default_list_data(10)
        error = {ct.err_code: 0, ct.err_msg: 'should create connect first'}
        collection_w.insert(data=data, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_default_partition(self):
        """
        target: test insert entities into default partition
        method: create partition and insert info collection
        expected: the collection insert count equals to nb
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        partition_w1 = self.init_partition_wrap(collection_w)
        data = cf.gen_default_list_data(nb=ct.default_nb)
        mutation_res, _ = collection_w.insert(data=data, partition_name=partition_w1.name)
        assert mutation_res.insert_count == ct.default_nb

    def test_insert_partition_not_existed(self):
        """
        target: test insert entities in collection created before
        method: create collection and insert entities in it, with the not existed partition_name param
        expected: error raised
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(nb=ct.default_nb)
        error = {ct.err_code: 1, ct.err_msg: "partitionID of partitionName:p can not be existed"}
        mutation_res, _ = collection_w.insert(data=df, partition_name="p", check_task=CheckTasks.err_res,
                                              check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_partition_repeatedly(self):
        """
        target: test insert entities in collection created before
        method: create collection and insert entities in it repeatedly, with the partition_name param
        expected: the collection row count equals to nq
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        partition_w1 = self.init_partition_wrap(collection_w)
        partition_w2 = self.init_partition_wrap(collection_w)
        df = cf.gen_default_dataframe_data(nb=ct.default_nb)
        mutation_res, _ = collection_w.insert(data=df, partition_name=partition_w1.name)
        new_res, _ = collection_w.insert(data=df, partition_name=partition_w2.name)
        assert mutation_res.insert_count == ct.default_nb
        assert new_res.insert_count == ct.default_nb

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_partition_with_ids(self):
        """
        target: test insert entities in collection created before, insert with ids
        method: create collection and insert entities in it, with the partition_name param
        expected: the collection insert count equals to nq
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        partition_name = cf.gen_unique_str(prefix)
        partition_w1 = self.init_partition_wrap(collection_w, partition_name=partition_name)
        df = cf.gen_default_dataframe_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=df, partition_name=partition_w1.name)
        assert mutation_res.insert_count == ct.default_nb

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_field_type_not_match(self):
        """
        target: test insert entities, with the entity field type updated
        method: update entity field type
        expected: error raised
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_collection_schema_all_datatype
        error = {ct.err_code: 0, ct.err_msg: "Data type is not support"}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_lack_vector_field(self):
        """
        target: test insert entities, with no vector field
        method: remove entity values of vector field
        expected: error raised
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_collection_schema([cf.gen_int64_field(is_primary=True)])
        error = {ct.err_code: 0, ct.err_msg: "Primary key field can only be one"}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_no_vector_field_dtype(self):
        """
        target: test insert entities, with vector field type is error
        method: vector field dtype is not existed
        expected: error raised
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        vec_field, _ = self.field_schema_wrap.init_field_schema(name=ct.default_int64_field_name, dtype=DataType.NONE)
        field_one = cf.gen_int64_field(is_primary=True)
        field_two = cf.gen_int64_field()
        df = [field_one, field_two, vec_field]
        error = {ct.err_code: 0, ct.err_msg: "Field dtype must be of DataType."}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_no_vector_field_name(self):
        """
        target: test insert entities, with no vector field name
        method: vector field name is error
        expected: error raised
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        vec_field = cf.gen_float_vec_field(name=ct.get_invalid_strs)
        field_one = cf.gen_int64_field(is_primary=True)
        field_two = cf.gen_int64_field()
        df = [field_one, field_two, vec_field]
        error = {ct.err_code: 0, ct.err_msg: "Data type is not support."}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_drop_collection(self):
        """
        target: test insert and drop
        method: insert data and drop collection
        expected: verify collection if exist
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        collection_list, _ = self.utility_wrap.list_collections()
        assert collection_w.name in collection_list
        df = cf.gen_default_dataframe_data(ct.default_nb)
        collection_w.insert(data=df)
        collection_w.drop()
        collection_list, _ = self.utility_wrap.list_collections()
        assert collection_w.name not in collection_list

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_create_index(self):
        """
        target: test insert and create index
        method: 1. insert 2. create index
        expected: verify num entities and index
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(ct.default_nb)
        collection_w.insert(data=df)
        assert collection_w.num_entities == ct.default_nb
        collection_w.create_index(ct.default_float_vec_field_name, default_index_params)
        assert collection_w.has_index()[0]
        index, _ = collection_w.index()
        assert index == Index(collection_w.collection, ct.default_float_vec_field_name, default_index_params)
        assert collection_w.indexes[0] == index

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_after_create_index(self):
        """
        target: test insert after create index
        method: 1. create index 2. insert data
        expected: verify index and num entities
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        collection_w.create_index(ct.default_float_vec_field_name, default_index_params)
        assert collection_w.has_index()[0]
        index, _ = collection_w.index()
        assert index == Index(collection_w.collection, ct.default_float_vec_field_name, default_index_params)
        assert collection_w.indexes[0] == index
        df = cf.gen_default_dataframe_data(ct.default_nb)
        collection_w.insert(data=df)
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_binary_after_index(self):
        """
        target: test insert binary after index
        method: 1.create index 2.insert binary data
        expected: 1.index ok 2.num entities correct
        """
        schema = cf.gen_default_binary_collection_schema()
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix), schema=schema)
        collection_w.create_index(ct.default_binary_vec_field_name, default_binary_index_params)
        assert collection_w.has_index()[0]
        index, _ = collection_w.index()
        assert index == Index(collection_w.collection, ct.default_binary_vec_field_name, default_binary_index_params)
        assert collection_w.indexes[0] == index
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb)
        collection_w.insert(data=df)
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_auto_id_create_index(self):
        """
        target: test create index in auto_id=True collection
        method: 1.create auto_id=True collection and insert
                2.create index
        expected: index correct
        """
        schema = cf.gen_default_collection_schema(auto_id=True)
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix), schema=schema)
        df = cf.gen_default_dataframe_data()
        df.drop(ct.default_int64_field_name, axis=1, inplace=True)
        mutation_res, _ = collection_w.insert(data=df)
        assert cf._check_primary_keys(mutation_res.primary_keys, ct.default_nb)
        assert collection_w.num_entities == ct.default_nb
        # create index
        collection_w.create_index(ct.default_float_vec_field_name, default_index_params)
        assert collection_w.has_index()[0]
        index, _ = collection_w.index()
        assert index == Index(collection_w.collection, ct.default_float_vec_field_name, default_index_params)
        assert collection_w.indexes[0] == index

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_auto_id_true(self):
        """
        target: test insert ids fields values when auto_id=True
        method: 1.create collection with auto_id=True 2.insert without ids
        expected: verify primary_keys and num_entities
        """
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_collection_schema(auto_id=True)
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)
        df = cf.gen_default_dataframe_data()
        df.drop(ct.default_int64_field_name, axis=1, inplace=True)
        mutation_res, _ = collection_w.insert(data=df)
        assert cf._check_primary_keys(mutation_res.primary_keys, ct.default_nb)
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_twice_auto_id_true(self):
        """
        target: test insert ids fields twice when auto_id=True
        method: 1.create collection with auto_id=True 2.insert twice
        expected: verify primary_keys unique
        """
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_collection_schema(auto_id=True)
        nb = 10
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)
        df = cf.gen_default_dataframe_data(nb)
        df.drop(ct.default_int64_field_name, axis=1, inplace=True)
        mutation_res, _ = collection_w.insert(data=df)
        primary_keys = mutation_res.primary_keys
        assert cf._check_primary_keys(primary_keys, nb)
        mutation_res_1, _ = collection_w.insert(data=df)
        primary_keys.extend(mutation_res_1.primary_keys)
        assert cf._check_primary_keys(primary_keys, nb * 2)
        assert collection_w.num_entities == nb * 2

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_auto_id_true_list_data(self):
        """
        target: test insert ids fields values when auto_id=True
        method: 1.create collection with auto_id=True 2.insert list data with ids field values
        expected: assert num entities
        """
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_collection_schema(auto_id=True)
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)
        data = cf.gen_default_list_data()
        mutation_res, _ = collection_w.insert(data=data[1:])
        assert mutation_res.insert_count == ct.default_nb
        assert cf._check_primary_keys(mutation_res.primary_keys, ct.default_nb)
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_auto_id_true_with_dataframe_values(self):
        """
        target: test insert with auto_id=True
        method: create collection with auto_id=True
        expected: 1.verify num entities 2.verify ids
        """
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_collection_schema(auto_id=True)
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)
        df = cf.gen_default_dataframe_data(nb=100)
        error = {ct.err_code: 0, ct.err_msg: 'Auto_id is True, primary field should not have data'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)
        assert collection_w.is_empty

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_auto_id_true_with_list_values(self):
        """
        target: test insert with auto_id=True
        method: create collection with auto_id=True
        expected: 1.verify num entities 2.verify ids
        """
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_default_collection_schema(auto_id=True)
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)
        data = cf.gen_default_list_data(nb=100)
        error = {ct.err_code: 0, ct.err_msg: 'The data fields number is not match with schema'}
        collection_w.insert(data=data, check_task=CheckTasks.err_res, check_items=error)
        assert collection_w.is_empty

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_auto_id_false_same_values(self):
        """
        target: test insert same ids with auto_id false
        method: 1.create collection with auto_id=False 2.insert same int64 field values
        expected: raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 100
        data = cf.gen_default_list_data(nb=nb)
        data[0] = [1 for i in range(nb)]
        mutation_res, _ = collection_w.insert(data)
        assert mutation_res.insert_count == nb
        assert mutation_res.primary_keys == data[0]

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_auto_id_false_negative_values(self):
        """
        target: test insert negative ids with auto_id false
        method: auto_id=False, primary field values is negative
        expected: verify num entities
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 100
        data = cf.gen_default_list_data(nb)
        data[0] = [i for i in range(0, -nb, -1)]
        mutation_res, _ = collection_w.insert(data)
        assert mutation_res.primary_keys == data[0]
        assert collection_w.num_entities == nb

    @pytest.mark.tags(CaseLabel.L1)
    # @pytest.mark.xfail(reason="issue 15416")
    def test_insert_multi_threading(self):
        """
        target: test concurrent insert
        method: multi threads insert
        expected: verify num entities
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(ct.default_nb)
        thread_num = 4
        threads = []
        primary_keys = df[ct.default_int64_field_name].values.tolist()

        def insert(thread_i):
            log.debug(f'In thread-{thread_i}')
            mutation_res, _ = collection_w.insert(df)
            assert mutation_res.insert_count == ct.default_nb
            assert mutation_res.primary_keys == primary_keys

        for i in range(thread_num):
            x = threading.Thread(target=insert, args=(i,))
            threads.append(x)
            x.start()
        for t in threads:
            t.join()
        assert collection_w.num_entities == ct.default_nb * thread_num

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="Currently primary keys are not unique")
    def test_insert_multi_threading_auto_id(self):
        """
        target: test concurrent insert auto_id=True collection
        method: 1.create auto_id=True collection 2.concurrent insert
        expected: verify primary keys unique
        """
        pass

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_multi_times(self, dim):
        """
        target: test insert multi times
        method: insert data multi times
        expected: verify num entities
        """
        step = 120
        nb = 12000
        collection_w = self.init_collection_general(prefix, dim=dim)[0]
        for _ in range(nb // step):
            df = cf.gen_default_dataframe_data(step, dim)
            mutation_res, _ = collection_w.insert(data=df)
            assert mutation_res.insert_count == step
            assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()

        assert collection_w.num_entities == nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_all_datatype_collection(self):
        """
        target: test insert into collection that contains all datatype fields
        method: 1.create all datatype collection 2.insert data
        expected: verify num entities
        """
        self._connect()
        nb = 100
        df = cf.gen_dataframe_all_data_type(nb=nb)
        self.collection_wrap.construct_from_dataframe(cf.gen_unique_str(prefix), df,
                                                      primary_field=ct.default_int64_field_name)
        assert self.collection_wrap.num_entities == nb


class TestInsertAsync(TestcaseBase):
    """
    ******************************************************************
      The following cases are used to test insert async
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_sync(self):
        """
        target: test async insert
        method: insert with async=True
        expected: verify num entities
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data()
        future, _ = collection_w.insert(data=df, _async=True)
        future.done()
        mutation_res = future.result()
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_async_false(self):
        """
        target: test insert with false async
        method: async = false
        expected: verify num entities
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data()
        mutation_res, _ = collection_w.insert(data=df, _async=False)
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_async_callback(self):
        """
        target: test insert with callback func
        method: insert with callback func
        expected: verify num entities
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data()
        future, _ = collection_w.insert(data=df, _async=True, _callback=assert_mutation_result)
        future.done()
        mutation_res = future.result()
        assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_long(self):
        """
        target: test insert with async
        method: insert 5w entities with callback func
        expected: verify num entities
        """
        nb = 50000
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(nb)
        future, _ = collection_w.insert(data=df, _async=True)
        future.done()
        mutation_res = future.result()
        assert mutation_res.insert_count == nb
        assert mutation_res.primary_keys == df[ct.default_int64_field_name].values.tolist()
        assert collection_w.num_entities == nb

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_callback_timeout(self):
        """
        target: test insert async with callback
        method: insert 10w entities with timeout=1
        expected: raise exception
        """
        nb = 100000
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data(nb)
        future, _ = collection_w.insert(data=df, _async=True, _callback=None, timeout=0.2)
        with pytest.raises(MilvusException):
            future.result()

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_invalid_data(self):
        """
        target: test insert async with invalid data
        method: insert async with invalid data
        expected: raise exception
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        columns = [ct.default_int64_field_name, ct.default_float_vec_field_name]
        df = pd.DataFrame(columns=columns)
        error = {ct.err_code: 0, ct.err_msg: "Cannot infer schema from empty dataframe"}
        collection_w.insert(data=df, _async=True, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_invalid_partition(self):
        """
        target: test insert async with invalid partition
        method: insert async with invalid partition
        expected: raise exception
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_default_dataframe_data()
        err_msg = "partitionID of partitionName:p can not be find"
        future, _ = collection_w.insert(data=df, partition_name="p", _async=True)
        future.done()
        with pytest.raises(MilvusException, match=err_msg):
            future.result()

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_async_no_vectors_raise_exception(self):
        """
        target: test insert vectors with no vectors
        method: set only vector field and insert into collection
        expected: raise exception
        """
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix))
        df = cf.gen_collection_schema([cf.gen_int64_field(is_primary=True)])
        error = {ct.err_code: 0, ct.err_msg: "fleldSchema lack of vector field."}
        future, _ = collection_w.insert(data=df, _async=True, check_task=CheckTasks.err_res, check_items=error)


def assert_mutation_result(mutation_res):
    assert mutation_res.insert_count == ct.default_nb


class TestInsertBinary(TestcaseBase):

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_binary_partition(self):
        """
        target: test insert entities and create partition 
        method: create collection and insert binary entities in it, with the partition_name param
        expected: the collection row count equals to nb
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name, schema=default_binary_schema)
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb)
        partition_name = cf.gen_unique_str(prefix)
        partition_w1 = self.init_partition_wrap(collection_w, partition_name=partition_name)
        mutation_res, _ = collection_w.insert(data=df, partition_name=partition_w1.name)
        assert mutation_res.insert_count == ct.default_nb

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_binary_multi_times(self):
        """
        target: test insert entities multi times and final flush
        method: create collection and insert binary entity multi 
        expected: the collection row count equals to nb
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name, schema=default_binary_schema)
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb)
        nums = 2
        for i in range(nums):
            mutation_res, _ = collection_w.insert(data=df)
        assert collection_w.num_entities == ct.default_nb * nums

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_binary_create_index(self):
        """
        target: test build index insert after vector
        method: insert vector and build index
        expected: no error raised
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name, schema=default_binary_schema)
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=df)
        assert mutation_res.insert_count == ct.default_nb
        default_index = {"index_type": "BIN_IVF_FLAT", "params": {"nlist": 128}, "metric_type": "JACCARD"}
        collection_w.create_index("binary_vector", default_index)


class TestInsertInvalid(TestcaseBase):
    """
      ******************************************************************
      The following cases are used to test insert invalid params
      ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_ids_invalid(self):
        """
        target: test insert, with using auto id is invaild, which are not int64
        method: create collection and insert entities in it
        expected: raise exception
        """
        collection_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=collection_name)
        int_field = cf.gen_float_field(is_primary=True)
        vec_field = cf.gen_float_vec_field(name='vec')
        df = [int_field, vec_field]
        error = {ct.err_code: 0, ct.err_msg: "Primary key type must be DataType.INT64."}
        mutation_res, _ = collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_partition_name(self):
        """
        target: test insert with invalid scenario
        method: insert with invalid partition name
        expected: raise exception
        """
        collection_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=collection_name)
        df = cf.gen_default_list_data(ct.default_nb)
        error = {ct.err_code: 1, 'err_msg': "partition name is illegal"}
        mutation_res, _ = collection_w.insert(data=df, partition_name="p", check_task=CheckTasks.err_res,
                                              check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_with_invalid_field_value(self):
        """
        target: test insert with invalid field
        method: insert with invalid field value
        expected: raise exception
        """
        collection_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=collection_name)
        field_one = cf.gen_int64_field(is_primary=True)
        field_two = cf.gen_int64_field()
        vec_field = ct.get_invalid_vectors
        df = [field_one, field_two, vec_field]
        error = {ct.err_code: 0, ct.err_msg: "The field of schema type must be FieldSchema."}
        mutation_res, _ = collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)


class TestInsertInvalidBinary(TestcaseBase):
    """
      ******************************************************************
      The following cases are used to test insert invalid params of binary
      ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_ids_binary_invalid(self):
        """
        target: test insert, with using customize ids, which are not int64
        method: create collection and insert entities in it
        expected: raise exception
        """
        collection_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=collection_name)
        field_one = cf.gen_float_field(is_primary=True)
        field_two = cf.gen_float_field()
        vec_field, _ = self.field_schema_wrap.init_field_schema(name=ct.default_binary_vec_field_name,
                                                                dtype=DataType.BINARY_VECTOR)
        df = [field_one, field_two, vec_field]
        error = {ct.err_code: 0, ct.err_msg: "Data type is not support."}
        mutation_res, _ = collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_insert_with_invalid_binary_partition_name(self):
        """
        target: test insert with invalid scenario
        method: insert with invalid partition name
        expected: raise exception
        """
        collection_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=collection_name)
        partition_name = ct.get_invalid_strs
        df, _ = cf.gen_default_binary_dataframe_data(ct.default_nb)
        error = {ct.err_code: 1, 'err_msg': "The types of schema and data do not match."}
        mutation_res, _ = collection_w.insert(data=df, partition_name=partition_name, check_task=CheckTasks.err_res,
                                              check_items=error)


class TestInsertString(TestcaseBase):
    """
      ******************************************************************
      The following cases are used to test insert string
      ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_string_field_is_primary(self):
        """
        target: test insert string is primary
        method: 1.create a collection and string field is primary
                2.insert string field data
        expected: Insert Successfully
        """
        c_name = cf.gen_unique_str(prefix)
        schema = cf.gen_string_pk_default_collection_schema()
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)
        data = cf.gen_default_list_data(ct.default_nb)
        mutation_res, _ = collection_w.insert(data=data)
        assert mutation_res.insert_count == ct.default_nb
        assert mutation_res.primary_keys == data[2]

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize("string_fields", [[cf.gen_string_field(name="string_field1")],
                                               [cf.gen_string_field(name="string_field2")],
                                               [cf.gen_string_field(name="string_field3")]])
    def test_insert_multi_string_fields(self, string_fields):
        """
        target: test insert multi string fields
        method: 1.create a collection
                2.Insert multi string fields
        expected: Insert Successfully
        """

        schema = cf.gen_schema_multi_string_fields(string_fields)
        collection_w = self.init_collection_wrap(name=cf.gen_unique_str(prefix), schema=schema)
        df = cf.gen_dataframe_multi_string_fields(string_fields=string_fields)
        collection_w.insert(df)
        assert collection_w.num_entities == ct.default_nb

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_string_field_invalid_data(self):
        """
        target: test insert string field data is not match
        method: 1.create a collection
                2.Insert string field data is not match
        expected: Raise exceptions
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nb = 10
        df = cf.gen_default_dataframe_data(nb)
        new_float_value = pd.Series(data=[float(i) for i in range(nb)], dtype="float64")
        df.iloc[:, 2] = new_float_value
        error = {ct.err_code: 0, ct.err_msg: 'The types of schema and data do not match'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_string_field_name_invalid(self):
        """
        target: test insert string field name is invaild
        method: 1.create a collection  
                2.Insert string field name is invalid
        expected: Raise exceptions
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        df = [cf.gen_int64_field(), cf.gen_string_field(name=ct.get_invalid_strs), cf.gen_float_vec_field()]
        error = {ct.err_code: 0, ct.err_msg: 'Data type is not support.'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L0)
    def test_insert_string_field_length_exceed(self):
        """
        target: test insert string field exceed the maximum length
        method: 1.create a collection  
                2.Insert string field length is exceeded maximum value of 65535
        expected: Raise exceptions
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        nums = 70000
        field_one = cf.gen_int64_field()
        field_two = cf.gen_float_field()
        field_three = cf.gen_string_field(max_length=nums)
        vec_field = cf.gen_float_vec_field()
        df = [field_one, field_two, field_three, vec_field]
        error = {ct.err_code: 0, ct.err_msg: 'Data type is not support.'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_string_field_dtype_invalid(self):
        """
        target: test insert string field with invaild dtype
        method: 1.create a collection  
                2.Insert string field dtype is invalid
        expected: Raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        string_field = self.field_schema_wrap.init_field_schema(name="string", dtype=DataType.STRING)[0]
        int_field = cf.gen_int64_field(is_primary=True)
        vec_field = cf.gen_float_vec_field()
        df = [string_field, int_field, vec_field]
        error = {ct.err_code: 0, ct.err_msg: 'Data type is not support.'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_insert_string_field_auto_id_is_true(self):
        """
        target: test create collection with string field 
        method: 1.create a collection  
                2.Insert string field with auto id is true
        expected: Raise exception
        """
        c_name = cf.gen_unique_str(prefix)
        collection_w = self.init_collection_wrap(name=c_name)
        int_field = cf.gen_int64_field()
        vec_field = cf.gen_float_vec_field()
        string_field = cf.gen_string_field(is_primary=True, auto_id=True)
        df = [int_field, string_field, vec_field]
        error = {ct.err_code: 0, ct.err_msg: 'Data type is not support.'}
        collection_w.insert(data=df, check_task=CheckTasks.err_res, check_items=error)
