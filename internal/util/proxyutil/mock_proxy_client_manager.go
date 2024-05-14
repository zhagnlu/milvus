// Code generated by mockery v2.32.4. DO NOT EDIT.

package proxyutil

import (
	context "context"

	milvuspb "github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	mock "github.com/stretchr/testify/mock"

	proxypb "github.com/milvus-io/milvus/internal/proto/proxypb"

	sessionutil "github.com/milvus-io/milvus/internal/util/sessionutil"

	types "github.com/milvus-io/milvus/internal/types"

	typeutil "github.com/milvus-io/milvus/pkg/util/typeutil"
)

// MockProxyClientManager is an autogenerated mock type for the ProxyClientManagerInterface type
type MockProxyClientManager struct {
	mock.Mock
}

type MockProxyClientManager_Expecter struct {
	mock *mock.Mock
}

func (_m *MockProxyClientManager) EXPECT() *MockProxyClientManager_Expecter {
	return &MockProxyClientManager_Expecter{mock: &_m.Mock}
}

// AddProxyClient provides a mock function with given fields: session
func (_m *MockProxyClientManager) AddProxyClient(session *sessionutil.Session) {
	_m.Called(session)
}

// MockProxyClientManager_AddProxyClient_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AddProxyClient'
type MockProxyClientManager_AddProxyClient_Call struct {
	*mock.Call
}

// AddProxyClient is a helper method to define mock.On call
//   - session *sessionutil.Session
func (_e *MockProxyClientManager_Expecter) AddProxyClient(session interface{}) *MockProxyClientManager_AddProxyClient_Call {
	return &MockProxyClientManager_AddProxyClient_Call{Call: _e.mock.On("AddProxyClient", session)}
}

func (_c *MockProxyClientManager_AddProxyClient_Call) Run(run func(session *sessionutil.Session)) *MockProxyClientManager_AddProxyClient_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*sessionutil.Session))
	})
	return _c
}

func (_c *MockProxyClientManager_AddProxyClient_Call) Return() *MockProxyClientManager_AddProxyClient_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockProxyClientManager_AddProxyClient_Call) RunAndReturn(run func(*sessionutil.Session)) *MockProxyClientManager_AddProxyClient_Call {
	_c.Call.Return(run)
	return _c
}

// AddProxyClients provides a mock function with given fields: session
func (_m *MockProxyClientManager) AddProxyClients(session []*sessionutil.Session) {
	_m.Called(session)
}

// MockProxyClientManager_AddProxyClients_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AddProxyClients'
type MockProxyClientManager_AddProxyClients_Call struct {
	*mock.Call
}

// AddProxyClients is a helper method to define mock.On call
//   - session []*sessionutil.Session
func (_e *MockProxyClientManager_Expecter) AddProxyClients(session interface{}) *MockProxyClientManager_AddProxyClients_Call {
	return &MockProxyClientManager_AddProxyClients_Call{Call: _e.mock.On("AddProxyClients", session)}
}

func (_c *MockProxyClientManager_AddProxyClients_Call) Run(run func(session []*sessionutil.Session)) *MockProxyClientManager_AddProxyClients_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].([]*sessionutil.Session))
	})
	return _c
}

func (_c *MockProxyClientManager_AddProxyClients_Call) Return() *MockProxyClientManager_AddProxyClients_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockProxyClientManager_AddProxyClients_Call) RunAndReturn(run func([]*sessionutil.Session)) *MockProxyClientManager_AddProxyClients_Call {
	_c.Call.Return(run)
	return _c
}

// DelProxyClient provides a mock function with given fields: s
func (_m *MockProxyClientManager) DelProxyClient(s *sessionutil.Session) {
	_m.Called(s)
}

// MockProxyClientManager_DelProxyClient_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'DelProxyClient'
type MockProxyClientManager_DelProxyClient_Call struct {
	*mock.Call
}

// DelProxyClient is a helper method to define mock.On call
//   - s *sessionutil.Session
func (_e *MockProxyClientManager_Expecter) DelProxyClient(s interface{}) *MockProxyClientManager_DelProxyClient_Call {
	return &MockProxyClientManager_DelProxyClient_Call{Call: _e.mock.On("DelProxyClient", s)}
}

func (_c *MockProxyClientManager_DelProxyClient_Call) Run(run func(s *sessionutil.Session)) *MockProxyClientManager_DelProxyClient_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(*sessionutil.Session))
	})
	return _c
}

func (_c *MockProxyClientManager_DelProxyClient_Call) Return() *MockProxyClientManager_DelProxyClient_Call {
	_c.Call.Return()
	return _c
}

func (_c *MockProxyClientManager_DelProxyClient_Call) RunAndReturn(run func(*sessionutil.Session)) *MockProxyClientManager_DelProxyClient_Call {
	_c.Call.Return(run)
	return _c
}

// GetComponentStates provides a mock function with given fields: ctx
func (_m *MockProxyClientManager) GetComponentStates(ctx context.Context) (map[int64]*milvuspb.ComponentStates, error) {
	ret := _m.Called(ctx)

	var r0 map[int64]*milvuspb.ComponentStates
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context) (map[int64]*milvuspb.ComponentStates, error)); ok {
		return rf(ctx)
	}
	if rf, ok := ret.Get(0).(func(context.Context) map[int64]*milvuspb.ComponentStates); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(map[int64]*milvuspb.ComponentStates)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockProxyClientManager_GetComponentStates_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetComponentStates'
type MockProxyClientManager_GetComponentStates_Call struct {
	*mock.Call
}

// GetComponentStates is a helper method to define mock.On call
//   - ctx context.Context
func (_e *MockProxyClientManager_Expecter) GetComponentStates(ctx interface{}) *MockProxyClientManager_GetComponentStates_Call {
	return &MockProxyClientManager_GetComponentStates_Call{Call: _e.mock.On("GetComponentStates", ctx)}
}

func (_c *MockProxyClientManager_GetComponentStates_Call) Run(run func(ctx context.Context)) *MockProxyClientManager_GetComponentStates_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockProxyClientManager_GetComponentStates_Call) Return(_a0 map[int64]*milvuspb.ComponentStates, _a1 error) *MockProxyClientManager_GetComponentStates_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockProxyClientManager_GetComponentStates_Call) RunAndReturn(run func(context.Context) (map[int64]*milvuspb.ComponentStates, error)) *MockProxyClientManager_GetComponentStates_Call {
	_c.Call.Return(run)
	return _c
}

// GetProxyClients provides a mock function with given fields:
func (_m *MockProxyClientManager) GetProxyClients() *typeutil.ConcurrentMap[int64, types.ProxyClient] {
	ret := _m.Called()

	var r0 *typeutil.ConcurrentMap[int64, types.ProxyClient]
	if rf, ok := ret.Get(0).(func() *typeutil.ConcurrentMap[int64, types.ProxyClient]); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*typeutil.ConcurrentMap[int64, types.ProxyClient])
		}
	}

	return r0
}

// MockProxyClientManager_GetProxyClients_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetProxyClients'
type MockProxyClientManager_GetProxyClients_Call struct {
	*mock.Call
}

// GetProxyClients is a helper method to define mock.On call
func (_e *MockProxyClientManager_Expecter) GetProxyClients() *MockProxyClientManager_GetProxyClients_Call {
	return &MockProxyClientManager_GetProxyClients_Call{Call: _e.mock.On("GetProxyClients")}
}

func (_c *MockProxyClientManager_GetProxyClients_Call) Run(run func()) *MockProxyClientManager_GetProxyClients_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockProxyClientManager_GetProxyClients_Call) Return(_a0 *typeutil.ConcurrentMap[int64, types.ProxyClient]) *MockProxyClientManager_GetProxyClients_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_GetProxyClients_Call) RunAndReturn(run func() *typeutil.ConcurrentMap[int64, types.ProxyClient]) *MockProxyClientManager_GetProxyClients_Call {
	_c.Call.Return(run)
	return _c
}

// GetProxyCount provides a mock function with given fields:
func (_m *MockProxyClientManager) GetProxyCount() int {
	ret := _m.Called()

	var r0 int
	if rf, ok := ret.Get(0).(func() int); ok {
		r0 = rf()
	} else {
		r0 = ret.Get(0).(int)
	}

	return r0
}

// MockProxyClientManager_GetProxyCount_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetProxyCount'
type MockProxyClientManager_GetProxyCount_Call struct {
	*mock.Call
}

// GetProxyCount is a helper method to define mock.On call
func (_e *MockProxyClientManager_Expecter) GetProxyCount() *MockProxyClientManager_GetProxyCount_Call {
	return &MockProxyClientManager_GetProxyCount_Call{Call: _e.mock.On("GetProxyCount")}
}

func (_c *MockProxyClientManager_GetProxyCount_Call) Run(run func()) *MockProxyClientManager_GetProxyCount_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockProxyClientManager_GetProxyCount_Call) Return(_a0 int) *MockProxyClientManager_GetProxyCount_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_GetProxyCount_Call) RunAndReturn(run func() int) *MockProxyClientManager_GetProxyCount_Call {
	_c.Call.Return(run)
	return _c
}

// GetProxyMetrics provides a mock function with given fields: ctx
func (_m *MockProxyClientManager) GetProxyMetrics(ctx context.Context) ([]*milvuspb.GetMetricsResponse, error) {
	ret := _m.Called(ctx)

	var r0 []*milvuspb.GetMetricsResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context) ([]*milvuspb.GetMetricsResponse, error)); ok {
		return rf(ctx)
	}
	if rf, ok := ret.Get(0).(func(context.Context) []*milvuspb.GetMetricsResponse); ok {
		r0 = rf(ctx)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).([]*milvuspb.GetMetricsResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context) error); ok {
		r1 = rf(ctx)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// MockProxyClientManager_GetProxyMetrics_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'GetProxyMetrics'
type MockProxyClientManager_GetProxyMetrics_Call struct {
	*mock.Call
}

// GetProxyMetrics is a helper method to define mock.On call
//   - ctx context.Context
func (_e *MockProxyClientManager_Expecter) GetProxyMetrics(ctx interface{}) *MockProxyClientManager_GetProxyMetrics_Call {
	return &MockProxyClientManager_GetProxyMetrics_Call{Call: _e.mock.On("GetProxyMetrics", ctx)}
}

func (_c *MockProxyClientManager_GetProxyMetrics_Call) Run(run func(ctx context.Context)) *MockProxyClientManager_GetProxyMetrics_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context))
	})
	return _c
}

func (_c *MockProxyClientManager_GetProxyMetrics_Call) Return(_a0 []*milvuspb.GetMetricsResponse, _a1 error) *MockProxyClientManager_GetProxyMetrics_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *MockProxyClientManager_GetProxyMetrics_Call) RunAndReturn(run func(context.Context) ([]*milvuspb.GetMetricsResponse, error)) *MockProxyClientManager_GetProxyMetrics_Call {
	_c.Call.Return(run)
	return _c
}

// InvalidateCollectionMetaCache provides a mock function with given fields: ctx, request, opts
func (_m *MockProxyClientManager) InvalidateCollectionMetaCache(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest, opts ...ExpireCacheOpt) error {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, request)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *proxypb.InvalidateCollMetaCacheRequest, ...ExpireCacheOpt) error); ok {
		r0 = rf(ctx, request, opts...)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockProxyClientManager_InvalidateCollectionMetaCache_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'InvalidateCollectionMetaCache'
type MockProxyClientManager_InvalidateCollectionMetaCache_Call struct {
	*mock.Call
}

// InvalidateCollectionMetaCache is a helper method to define mock.On call
//   - ctx context.Context
//   - request *proxypb.InvalidateCollMetaCacheRequest
//   - opts ...ExpireCacheOpt
func (_e *MockProxyClientManager_Expecter) InvalidateCollectionMetaCache(ctx interface{}, request interface{}, opts ...interface{}) *MockProxyClientManager_InvalidateCollectionMetaCache_Call {
	return &MockProxyClientManager_InvalidateCollectionMetaCache_Call{Call: _e.mock.On("InvalidateCollectionMetaCache",
		append([]interface{}{ctx, request}, opts...)...)}
}

func (_c *MockProxyClientManager_InvalidateCollectionMetaCache_Call) Run(run func(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest, opts ...ExpireCacheOpt)) *MockProxyClientManager_InvalidateCollectionMetaCache_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]ExpireCacheOpt, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(ExpireCacheOpt)
			}
		}
		run(args[0].(context.Context), args[1].(*proxypb.InvalidateCollMetaCacheRequest), variadicArgs...)
	})
	return _c
}

func (_c *MockProxyClientManager_InvalidateCollectionMetaCache_Call) Return(_a0 error) *MockProxyClientManager_InvalidateCollectionMetaCache_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_InvalidateCollectionMetaCache_Call) RunAndReturn(run func(context.Context, *proxypb.InvalidateCollMetaCacheRequest, ...ExpireCacheOpt) error) *MockProxyClientManager_InvalidateCollectionMetaCache_Call {
	_c.Call.Return(run)
	return _c
}

// InvalidateCredentialCache provides a mock function with given fields: ctx, request
func (_m *MockProxyClientManager) InvalidateCredentialCache(ctx context.Context, request *proxypb.InvalidateCredCacheRequest) error {
	ret := _m.Called(ctx, request)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *proxypb.InvalidateCredCacheRequest) error); ok {
		r0 = rf(ctx, request)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockProxyClientManager_InvalidateCredentialCache_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'InvalidateCredentialCache'
type MockProxyClientManager_InvalidateCredentialCache_Call struct {
	*mock.Call
}

// InvalidateCredentialCache is a helper method to define mock.On call
//   - ctx context.Context
//   - request *proxypb.InvalidateCredCacheRequest
func (_e *MockProxyClientManager_Expecter) InvalidateCredentialCache(ctx interface{}, request interface{}) *MockProxyClientManager_InvalidateCredentialCache_Call {
	return &MockProxyClientManager_InvalidateCredentialCache_Call{Call: _e.mock.On("InvalidateCredentialCache", ctx, request)}
}

func (_c *MockProxyClientManager_InvalidateCredentialCache_Call) Run(run func(ctx context.Context, request *proxypb.InvalidateCredCacheRequest)) *MockProxyClientManager_InvalidateCredentialCache_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*proxypb.InvalidateCredCacheRequest))
	})
	return _c
}

func (_c *MockProxyClientManager_InvalidateCredentialCache_Call) Return(_a0 error) *MockProxyClientManager_InvalidateCredentialCache_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_InvalidateCredentialCache_Call) RunAndReturn(run func(context.Context, *proxypb.InvalidateCredCacheRequest) error) *MockProxyClientManager_InvalidateCredentialCache_Call {
	_c.Call.Return(run)
	return _c
}

// InvalidateShardLeaderCache provides a mock function with given fields: ctx, request
func (_m *MockProxyClientManager) InvalidateShardLeaderCache(ctx context.Context, request *proxypb.InvalidateShardLeaderCacheRequest) error {
	ret := _m.Called(ctx, request)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *proxypb.InvalidateShardLeaderCacheRequest) error); ok {
		r0 = rf(ctx, request)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockProxyClientManager_InvalidateShardLeaderCache_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'InvalidateShardLeaderCache'
type MockProxyClientManager_InvalidateShardLeaderCache_Call struct {
	*mock.Call
}

// InvalidateShardLeaderCache is a helper method to define mock.On call
//   - ctx context.Context
//   - request *proxypb.InvalidateShardLeaderCacheRequest
func (_e *MockProxyClientManager_Expecter) InvalidateShardLeaderCache(ctx interface{}, request interface{}) *MockProxyClientManager_InvalidateShardLeaderCache_Call {
	return &MockProxyClientManager_InvalidateShardLeaderCache_Call{Call: _e.mock.On("InvalidateShardLeaderCache", ctx, request)}
}

func (_c *MockProxyClientManager_InvalidateShardLeaderCache_Call) Run(run func(ctx context.Context, request *proxypb.InvalidateShardLeaderCacheRequest)) *MockProxyClientManager_InvalidateShardLeaderCache_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*proxypb.InvalidateShardLeaderCacheRequest))
	})
	return _c
}

func (_c *MockProxyClientManager_InvalidateShardLeaderCache_Call) Return(_a0 error) *MockProxyClientManager_InvalidateShardLeaderCache_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_InvalidateShardLeaderCache_Call) RunAndReturn(run func(context.Context, *proxypb.InvalidateShardLeaderCacheRequest) error) *MockProxyClientManager_InvalidateShardLeaderCache_Call {
	_c.Call.Return(run)
	return _c
}

// RefreshPolicyInfoCache provides a mock function with given fields: ctx, req
func (_m *MockProxyClientManager) RefreshPolicyInfoCache(ctx context.Context, req *proxypb.RefreshPolicyInfoCacheRequest) error {
	ret := _m.Called(ctx, req)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *proxypb.RefreshPolicyInfoCacheRequest) error); ok {
		r0 = rf(ctx, req)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockProxyClientManager_RefreshPolicyInfoCache_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'RefreshPolicyInfoCache'
type MockProxyClientManager_RefreshPolicyInfoCache_Call struct {
	*mock.Call
}

// RefreshPolicyInfoCache is a helper method to define mock.On call
//   - ctx context.Context
//   - req *proxypb.RefreshPolicyInfoCacheRequest
func (_e *MockProxyClientManager_Expecter) RefreshPolicyInfoCache(ctx interface{}, req interface{}) *MockProxyClientManager_RefreshPolicyInfoCache_Call {
	return &MockProxyClientManager_RefreshPolicyInfoCache_Call{Call: _e.mock.On("RefreshPolicyInfoCache", ctx, req)}
}

func (_c *MockProxyClientManager_RefreshPolicyInfoCache_Call) Run(run func(ctx context.Context, req *proxypb.RefreshPolicyInfoCacheRequest)) *MockProxyClientManager_RefreshPolicyInfoCache_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*proxypb.RefreshPolicyInfoCacheRequest))
	})
	return _c
}

func (_c *MockProxyClientManager_RefreshPolicyInfoCache_Call) Return(_a0 error) *MockProxyClientManager_RefreshPolicyInfoCache_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_RefreshPolicyInfoCache_Call) RunAndReturn(run func(context.Context, *proxypb.RefreshPolicyInfoCacheRequest) error) *MockProxyClientManager_RefreshPolicyInfoCache_Call {
	_c.Call.Return(run)
	return _c
}

// SetRates provides a mock function with given fields: ctx, request
func (_m *MockProxyClientManager) SetRates(ctx context.Context, request *proxypb.SetRatesRequest) error {
	ret := _m.Called(ctx, request)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *proxypb.SetRatesRequest) error); ok {
		r0 = rf(ctx, request)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockProxyClientManager_SetRates_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'SetRates'
type MockProxyClientManager_SetRates_Call struct {
	*mock.Call
}

// SetRates is a helper method to define mock.On call
//   - ctx context.Context
//   - request *proxypb.SetRatesRequest
func (_e *MockProxyClientManager_Expecter) SetRates(ctx interface{}, request interface{}) *MockProxyClientManager_SetRates_Call {
	return &MockProxyClientManager_SetRates_Call{Call: _e.mock.On("SetRates", ctx, request)}
}

func (_c *MockProxyClientManager_SetRates_Call) Run(run func(ctx context.Context, request *proxypb.SetRatesRequest)) *MockProxyClientManager_SetRates_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*proxypb.SetRatesRequest))
	})
	return _c
}

func (_c *MockProxyClientManager_SetRates_Call) Return(_a0 error) *MockProxyClientManager_SetRates_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_SetRates_Call) RunAndReturn(run func(context.Context, *proxypb.SetRatesRequest) error) *MockProxyClientManager_SetRates_Call {
	_c.Call.Return(run)
	return _c
}

// UpdateCredentialCache provides a mock function with given fields: ctx, request
func (_m *MockProxyClientManager) UpdateCredentialCache(ctx context.Context, request *proxypb.UpdateCredCacheRequest) error {
	ret := _m.Called(ctx, request)

	var r0 error
	if rf, ok := ret.Get(0).(func(context.Context, *proxypb.UpdateCredCacheRequest) error); ok {
		r0 = rf(ctx, request)
	} else {
		r0 = ret.Error(0)
	}

	return r0
}

// MockProxyClientManager_UpdateCredentialCache_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'UpdateCredentialCache'
type MockProxyClientManager_UpdateCredentialCache_Call struct {
	*mock.Call
}

// UpdateCredentialCache is a helper method to define mock.On call
//   - ctx context.Context
//   - request *proxypb.UpdateCredCacheRequest
func (_e *MockProxyClientManager_Expecter) UpdateCredentialCache(ctx interface{}, request interface{}) *MockProxyClientManager_UpdateCredentialCache_Call {
	return &MockProxyClientManager_UpdateCredentialCache_Call{Call: _e.mock.On("UpdateCredentialCache", ctx, request)}
}

func (_c *MockProxyClientManager_UpdateCredentialCache_Call) Run(run func(ctx context.Context, request *proxypb.UpdateCredCacheRequest)) *MockProxyClientManager_UpdateCredentialCache_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(context.Context), args[1].(*proxypb.UpdateCredCacheRequest))
	})
	return _c
}

func (_c *MockProxyClientManager_UpdateCredentialCache_Call) Return(_a0 error) *MockProxyClientManager_UpdateCredentialCache_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockProxyClientManager_UpdateCredentialCache_Call) RunAndReturn(run func(context.Context, *proxypb.UpdateCredCacheRequest) error) *MockProxyClientManager_UpdateCredentialCache_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockProxyClientManager creates a new instance of MockProxyClientManager. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockProxyClientManager(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockProxyClientManager {
	mock := &MockProxyClientManager{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}