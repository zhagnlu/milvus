package kvfactory

import (
	"context"
	"testing"

	"github.com/bytedance/mockey"
	"github.com/stretchr/testify/require"
	clientv3 "go.etcd.io/etcd/client/v3"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

func TestCloseEtcdClient(t *testing.T) {
	for _, state := range []string{"uninitialized", "embedded", "already closed", "remote", "remote already closed"} {
		t.Run(state, func(t *testing.T) {
			previous := clientCreator
			clientCreator = &etcdClientCreator{}
			t.Cleanup(func() { clientCreator = previous })

			if state != "uninitialized" {
				var client *clientv3.Client
				if state == "remote" || state == "remote already closed" {
					// A network client has a real gRPC connection even without a
					// reachable server. Closing it must still succeed.
					var err error
					client, err = clientv3.New(clientv3.Config{Endpoints: []string{"http://127.0.0.1:1"}})
					require.NoError(t, err)
					require.NotNil(t, client.ActiveConnection())
				} else {
					// v3client.New uses NewCtxClient for embedded etcd. Its Close
					// returns context.Canceled because it has no gRPC connection.
					client = clientv3.NewCtxClient(context.Background())
				}
				t.Cleanup(func() { client.Close() })
				clientCreator.client = client
				rootpath := "test/meta"
				clientCreator.rootpath = &rootpath
				if state == "already closed" {
					require.ErrorIs(t, client.Close(), context.Canceled)
				}
				if state == "remote already closed" {
					require.NoError(t, client.Close())
				}
			}

			client := clientCreator.client
			require.NotPanics(t, CloseEtcdClient)
			if client != nil {
				require.ErrorIs(t, client.Ctx().Err(), context.Canceled)
			}
			require.Nil(t, clientCreator.client)
			require.Nil(t, clientCreator.rootpath)
			require.NotPanics(t, CloseEtcdClient)
		})
	}
}

func TestCloseEtcdClientWrappedCancellation(t *testing.T) {
	previous := clientCreator
	client := clientv3.NewCtxClient(context.Background())
	clientCreator = &etcdClientCreator{client: client}
	t.Cleanup(func() {
		clientCreator = previous
		client.Close()
	})
	patch := mockey.Mock((*clientv3.Client).Close).
		Return(merr.Wrap(context.Canceled, "closing embedded etcd client")).Build()
	defer patch.UnPatch()
	require.NotPanics(t, CloseEtcdClient)
	require.Nil(t, clientCreator.client)
	require.Nil(t, clientCreator.rootpath)
}

func TestCloseEtcdClientUnexpectedError(t *testing.T) {
	for _, closeErr := range []error{context.DeadlineExceeded, merr.ErrServiceInternal} {
		t.Run(closeErr.Error(), func(t *testing.T) {
			previous := clientCreator
			client := clientv3.NewCtxClient(context.Background())
			clientCreator = &etcdClientCreator{client: client}
			t.Cleanup(func() {
				clientCreator = previous
				client.Close()
			})
			patch := mockey.Mock((*clientv3.Client).Close).Return(closeErr).Build()
			defer patch.UnPatch()
			require.PanicsWithValue(t, closeErr, CloseEtcdClient)
		})
	}
}
