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

package msgstream

import (
	"context"
	"errors"
	"runtime"

	"github.com/opentracing/opentracing-go"
	"github.com/opentracing/opentracing-go/ext"
	"github.com/opentracing/opentracing-go/log"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/util/trace"
)

// ExtractFromPulsarMsgProperties extracts trace span from msg.properties.
// And it will attach some default tags to the span.
func ExtractFromPulsarMsgProperties(msg TsMsg, properties map[string]string) (opentracing.Span, bool) {
	if !allowTrace(msg) {
		return trace.NoopSpan(), false
	}
	tracer := opentracing.GlobalTracer()
	sc, _ := tracer.Extract(opentracing.TextMap, trace.PropertiesReaderWriter{PpMap: properties})
	name := "receive pulsar msg"
	opts := []opentracing.StartSpanOption{
		ext.RPCServerOption(sc),
		opentracing.Tags{
			"ID":       msg.ID(),
			"Type":     msg.Type(),
			"HashKeys": msg.HashKeys(),
			"Position": msg.Position(),
		}}
	return opentracing.StartSpan(name, opts...), true
}

// MsgSpanFromCtx extracts the span from context.
// And it will attach some default tags to the span.
func MsgSpanFromCtx(ctx context.Context, msg TsMsg, opts ...opentracing.StartSpanOption) (opentracing.Span, context.Context) {
	if ctx == nil {
		return trace.NoopSpan(), ctx
	}
	if !allowTrace(msg) {
		return trace.NoopSpan(), ctx
	}
	operationName := "send pulsar msg"
	opts = append(opts, opentracing.Tags{
		"ID":       msg.ID(),
		"Type":     msg.Type(),
		"HashKeys": msg.HashKeys(),
		"Position": msg.Position(),
	})

	var pcs [1]uintptr
	n := runtime.Callers(2, pcs[:])
	if n < 1 {
		span, ctx := opentracing.StartSpanFromContext(ctx, operationName, opts...)
		span.LogFields(log.Error(errors.New("runtime.Callers failed")))
		return span, ctx
	}
	file, line := runtime.FuncForPC(pcs[0]).FileLine(pcs[0])

	if parentSpan := opentracing.SpanFromContext(ctx); parentSpan != nil {
		opts = append(opts, opentracing.ChildOf(parentSpan.Context()))
	}
	span := opentracing.StartSpan(operationName, opts...)
	ctx = opentracing.ContextWithSpan(ctx, span)

	span.LogFields(log.String("filename", file), log.Int("line", line))

	return span, ctx
}

func allowTrace(in interface{}) bool {
	if in == nil {
		return false
	}
	switch res := in.(type) {
	case TsMsg:
		return !(res.Type() == commonpb.MsgType_TimeTick ||
			res.Type() == commonpb.MsgType_LoadIndex)
	default:
		return false
	}
}
