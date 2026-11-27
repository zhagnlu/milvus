// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

package sqlquery

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSQLGroupKeyExactTuples(t *testing.T) {
	keys := [][]any{
		{nil}, {""}, {"null"}, {"a", "bc"}, {"ab", "c"}, {"a\x00b", "c"}, {"a", "\x00bc"},
		{`a","b`}, {"世界"}, {"\\\""}, {int64(9007199254740992)}, {int64(9007199254740993)},
		{int32(1)}, {int64(1)}, {true}, {false}, {float32(1)}, {float64(1)}, {float64(0)}, {math.Copysign(0, -1)},
	}
	seen := make(map[string]bool)
	buffer := make([]byte, 0, 128)
	for _, values := range keys {
		encoded, err := appendSQLGroupKey(buffer[:0], values)
		require.NoError(t, err)
		key := string(encoded)
		require.False(t, seen[key], "aliased tuple %v", values)
		seen[key] = true
		again, err := appendSQLGroupKey(nil, values)
		require.NoError(t, err)
		require.Equal(t, encoded, again)
		buffer = encoded
	}
	for _, value := range []any{math.NaN(), math.Inf(1), float32(math.Inf(-1))} {
		_, err := appendSQLGroupKey(nil, []any{value})
		require.Error(t, err)
	}
}
