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
	"encoding/binary"
	"encoding/json"
	"math"
)

// appendSQLGroupKey encodes the fixed-schema scalar tuple without reflection or
// JSON string escaping. Tags and string lengths make tuple boundaries exact.
// The buffer can be reused after its bytes have been copied into a map key.
func appendSQLGroupKey(dst []byte, values []any) ([]byte, error) {
	for _, value := range values {
		switch value := value.(type) {
		case nil:
			dst = append(dst, 0)
		case bool:
			if value {
				dst = append(dst, 1)
			} else {
				dst = append(dst, 2)
			}
		case int32:
			dst = binary.LittleEndian.AppendUint32(append(dst, 3), uint32(value))
		case int64:
			dst = binary.LittleEndian.AppendUint64(append(dst, 4), uint64(value))
		case float32:
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				_, err := json.Marshal(value)
				return nil, err
			}
			dst = binary.LittleEndian.AppendUint32(append(dst, 5), math.Float32bits(value))
		case float64:
			if math.IsNaN(value) || math.IsInf(value, 0) {
				_, err := json.Marshal(value)
				return nil, err
			}
			dst = binary.LittleEndian.AppendUint64(append(dst, 6), math.Float64bits(value))
		case string:
			dst = binary.AppendUvarint(append(dst, 7), uint64(len(value)))
			dst = append(dst, value...)
		default:
			// Retain the previous JSON contract for any future accessor types.
			encoded, err := json.Marshal(value)
			if err != nil {
				return nil, err
			}
			dst = binary.AppendUvarint(append(dst, 255), uint64(len(encoded)))
			dst = append(dst, encoded...)
		}
	}
	return dst, nil
}
