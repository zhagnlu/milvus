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

// Package e2e provides standalone E2E tests that connect to an external
// Milvus instance via the REST API. No embedded cluster is started.
//
// Usage:
//
//	# Start Milvus externally, then:
//	cd tests/e2e
//	go test -v -run TestSqlQuery -milvus-addr=localhost:19530
package e2e

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var milvusAddr = flag.String("milvus-addr", "", "Milvus REST API address (host:port); required to run SQL E2E")
var milvusToken = flag.String("milvus-token", "", "Milvus authentication token")

// ---------- helpers ----------

type apiResp struct {
	Code    int             `json:"code"`
	Data    json.RawMessage `json:"data"`
	Message string          `json:"message"`
}

func baseURL() string {
	return "http://" + *milvusAddr + "/v2/vectordb"
}

func postJSON(t *testing.T, path string, body any) apiResp {
	t.Helper()
	payload, err := json.Marshal(body)
	require.NoError(t, err)

	req, err := http.NewRequest(http.MethodPost, baseURL()+path, bytes.NewBuffer(payload))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	if *milvusToken != "" {
		req.Header.Set("Authorization", "Bearer "+*milvusToken)
	}
	req.Header.Set("Accept-Type-Allow-Int64", "true")

	client := http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "body: %s", raw)

	var result apiResp
	err = json.Unmarshal(raw, &result)
	require.NoError(t, err, "body: %s", string(raw))
	return result
}

func sqlQuery(t *testing.T, sql string) apiResp {
	t.Helper()
	return postJSON(t, "/entities/sql_query", map[string]any{"sql": sql, "consistencyLevel": "Strong", "timezone": "UTC"})
}

func sqlQueryRows(t *testing.T, sql string) []map[string]any {
	t.Helper()
	resp := sqlQuery(t, sql)
	require.Equal(t, 0, resp.Code, "sql query failed: %s (sql: %s)", resp.Message, sql)
	var rows []map[string]any
	require.NoError(t, json.Unmarshal(resp.Data, &rows))
	return rows
}

// ---------- test data setup ----------

var collName = fmt.Sprintf("test_sql_e2e_%d", time.Now().UnixNano())

const (
	dim      = 128
	rowCount = 100
)

// setupCollection creates the collection, inserts data, creates index, and loads.
// It uses the REST API only — no gRPC dependency.
func setupCollection(t *testing.T) {
	t.Helper()

	// Drop if exists (ignore error)
	postJSON(t, "/collections/drop", map[string]any{
		"collectionName": collName,
	})

	// Create collection
	resp := postJSON(t, "/collections/create", map[string]any{
		"collectionName": collName,
		"params":         map[string]any{"shardsNum": 2},
		"schema": map[string]any{
			"fields": []map[string]any{
				{"fieldName": "pk", "dataType": "Int64", "isPrimary": true},
				{"fieldName": "age", "dataType": "Int64"},
				{"fieldName": "name", "dataType": "VarChar", "elementTypeParams": map[string]any{"max_length": 256}},
				{"fieldName": "score", "dataType": "Float"},
				{"fieldName": "data", "dataType": "JSON"},
				{"fieldName": "embedding", "dataType": "FloatVector", "elementTypeParams": map[string]any{"dim": dim}},
			},
		},
	})
	require.Equal(t, 0, resp.Code, "create collection failed: %s", resp.Message)

	// Insert data
	rng := rand.New(rand.NewSource(42))
	categories := []string{"electronics", "books", "clothing", "food", "sports"}
	rows := make([]map[string]any, rowCount)
	for i := 0; i < rowCount; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()
		}
		rows[i] = map[string]any{
			"pk":    int64(i),
			"age":   20 + (i % 50),
			"name":  fmt.Sprintf("user_%d", i),
			"score": float64(i) * 1.5,
			"data": map[string]any{
				"category": categories[i%len(categories)],
				"kind":     "commit",
				"did":      fmt.Sprintf("user_%d", i%4),
				"time_us":  int64(1704067200+i*1800) * 1_000_000,
				"commit":   map[string]any{"operation": "create", "collection": []string{"app.bsky.feed.post", "app.bsky.feed.repost", "app.bsky.feed.like"}[i%3]},
				"tags":     []string{"tag_a", "tag_b"},
				"info":     map[string]any{"level": i % 5},
			},
			"embedding": vec,
		}
	}
	resp = postJSON(t, "/entities/insert", map[string]any{
		"collectionName": collName,
		"data":           rows,
	})
	require.Equal(t, 0, resp.Code, "insert failed: %s", resp.Message)

	// Create index
	resp = postJSON(t, "/indexes/create", map[string]any{
		"collectionName": collName,
		"indexParams": []map[string]any{
			{
				"fieldName":  "embedding",
				"indexName":  "vec_idx",
				"metricType": "L2",
				"indexType":  "IVF_FLAT",
				"params":     map[string]any{"nlist": 10},
			},
		},
	})
	require.Equal(t, 0, resp.Code, "create index failed: %s", resp.Message)

	// Load collection
	resp = postJSON(t, "/collections/load", map[string]any{
		"collectionName": collName,
	})
	require.Equal(t, 0, resp.Code, "load collection failed: %s", resp.Message)

	// Wait for load to complete
	for i := 0; i < 60; i++ {
		resp = postJSON(t, "/collections/get_load_state", map[string]any{
			"collectionName": collName,
		})
		if resp.Code == 0 {
			var state map[string]any
			if json.Unmarshal(resp.Data, &state) == nil {
				if s, ok := state["loadState"].(string); ok && s == "LoadStateLoaded" {
					t.Log("collection loaded")
					return
				}
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	t.Fatal("timeout waiting for collection to load")
}

// cleanupCollection drops the test collection.
func cleanupCollection(t *testing.T) {
	t.Helper()
	postJSON(t, "/collections/drop", map[string]any{
		"collectionName": collName,
	})
}

// ---------- test cases ----------

func TestSqlQuery(t *testing.T) {
	if *milvusAddr == "" {
		t.Skip("-milvus-addr is required")
	}
	setupCollection(t)
	defer cleanupCollection(t)

	t.Run("SimpleSelect", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT pk, age, name FROM %s WHERE age > 30 LIMIT 10", collName))
		assert.LessOrEqual(t, len(rows), 10)
		for _, row := range rows {
			age, _ := row["age"].(float64)
			assert.Greater(t, age, float64(30))
		}
	})

	t.Run("SelectStar", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT * FROM %s LIMIT 5", collName))
		assert.LessOrEqual(t, len(rows), 5)
		if len(rows) > 0 {
			_, hasPK := rows[0]["pk"]
			_, hasAge := rows[0]["age"]
			assert.True(t, hasPK, "SELECT * should include pk")
			assert.True(t, hasAge, "SELECT * should include age")
		}
	})

	t.Run("JSONFieldAccess", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT pk, data->>'category' AS category FROM %s WHERE data->>'category' = 'books' LIMIT 10`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
		for _, row := range rows {
			cat, _ := row["category"].(string)
			assert.Equal(t, "books", cat)
		}
	})

	t.Run("NestedJSONAccess", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT pk, data->'info'->>'level' AS level FROM %s WHERE data->'info'->>'level' = '0' LIMIT 10`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
	})

	t.Run("CountStar", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT count(*) AS n FROM %s", collName))
		require.Len(t, rows, 1)
		assert.Equal(t, float64(rowCount), rows[0]["n"])
	})

	t.Run("CountWithWhere", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT count(*) AS n FROM %s WHERE age >= 40", collName))
		require.Len(t, rows, 1)
		assert.Equal(t, float64(60), rows[0]["n"])
	})

	t.Run("SumAvg", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT sum(score) AS s, avg(score) AS a FROM %s", collName))
		require.Len(t, rows, 1)
		assert.Equal(t, float64(7425), rows[0]["s"])
		assert.Equal(t, float64(74.25), rows[0]["a"])
	})

	t.Run("MinMax", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT min(age) AS lo, max(age) AS hi FROM %s", collName))
		require.Len(t, rows, 1)
		assert.Equal(t, float64(20), rows[0]["lo"])
		assert.Equal(t, float64(69), rows[0]["hi"])
	})

	t.Run("GroupBy", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT data->>'category' AS category, count(*) FROM %s GROUP BY data->>'category'`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
		assert.LessOrEqual(t, len(rows), 5)
	})

	t.Run("OrderByLimit", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT pk, age, score FROM %s ORDER BY score DESC LIMIT 5", collName))
		assert.LessOrEqual(t, len(rows), 5)
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["score"].(float64)
			curr, _ := rows[i]["score"].(float64)
			assert.GreaterOrEqual(t, prev, curr, "should be ordered by score DESC")
		}
	})

	t.Run("OrderByAsc", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT pk, age FROM %s ORDER BY age ASC LIMIT 10", collName))
		assert.LessOrEqual(t, len(rows), 10)
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["age"].(float64)
			curr, _ := rows[i]["age"].(float64)
			assert.LessOrEqual(t, prev, curr, "should be ordered by age ASC")
		}
	})

	t.Run("OrderByWithWhere", func(t *testing.T) {
		sql := fmt.Sprintf("SELECT pk, age, score FROM %s WHERE age > 30 ORDER BY score DESC LIMIT 5", collName)
		rows := sqlQueryRows(t, sql)
		assert.LessOrEqual(t, len(rows), 5)
		for _, row := range rows {
			age, _ := row["age"].(float64)
			assert.Greater(t, age, float64(30))
		}
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["score"].(float64)
			curr, _ := rows[i]["score"].(float64)
			assert.GreaterOrEqual(t, prev, curr)
		}
	})

	t.Run("OrderByDefaultAsc", func(t *testing.T) {
		// No direction specified → default ASC
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT pk, score FROM %s ORDER BY score LIMIT 5", collName))
		assert.LessOrEqual(t, len(rows), 5)
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["score"].(float64)
			curr, _ := rows[i]["score"].(float64)
			assert.LessOrEqual(t, prev, curr, "default direction should be ASC")
		}
	})

	t.Run("GroupByOrderByCount", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT data->>'category' AS category, count(*) AS cnt FROM %s GROUP BY data->>'category' ORDER BY cnt DESC`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["cnt"].(float64)
			curr, _ := rows[i]["cnt"].(float64)
			assert.GreaterOrEqual(t, prev, curr, "should be ordered by count DESC")
		}
	})

	t.Run("GroupByPlainFieldOrderByCount", func(t *testing.T) {
		// Use age (Int64) as group key to avoid segcore's JSON-output limitation
		sql := fmt.Sprintf(`SELECT age, count(*) AS cnt FROM %s GROUP BY age ORDER BY cnt DESC LIMIT 10`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["cnt"].(float64)
			curr, _ := rows[i]["cnt"].(float64)
			assert.GreaterOrEqual(t, prev, curr, "should be ordered by count DESC")
		}
	})

	t.Run("GroupByPlainFieldOrderByKey", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT age, count(*) AS cnt FROM %s GROUP BY age ORDER BY age ASC LIMIT 10`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["age"].(float64)
			curr, _ := rows[i]["age"].(float64)
			assert.LessOrEqual(t, prev, curr, "should be ordered by age ASC")
		}
	})

	t.Run("GroupByOrderByGroupKey", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT data->>'category' AS category, count(*) AS cnt FROM %s GROUP BY data->>'category' ORDER BY category ASC`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
		// Categories should be sorted alphabetically
		for i := 1; i < len(rows); i++ {
			prev, _ := rows[i-1]["category"].(string)
			curr, _ := rows[i]["category"].(string)
			assert.LessOrEqual(t, prev, curr, "categories should be ASC")
		}
	})

	t.Run("WhereComplex", func(t *testing.T) {
		sql := fmt.Sprintf("SELECT pk, age, name FROM %s WHERE (age > 30 AND score < 100) OR name = 'user_0' LIMIT 20", collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
	})

	t.Run("WhereIn", func(t *testing.T) {
		sql := fmt.Sprintf("SELECT pk, name FROM %s WHERE name IN ('user_0', 'user_1', 'user_2') LIMIT 10", collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
	})

	t.Run("WhereBetween", func(t *testing.T) {
		sql := fmt.Sprintf("SELECT pk, age FROM %s WHERE age BETWEEN 25 AND 35 LIMIT 20", collName)
		rows := sqlQueryRows(t, sql)
		for _, row := range rows {
			age, _ := row["age"].(float64)
			assert.True(t, age >= 25 && age <= 35, "age %v not in BETWEEN range", age)
		}
	})

	t.Run("WhereLike", func(t *testing.T) {
		sql := fmt.Sprintf(`SELECT pk, name FROM %s WHERE name LIKE 'user_1%%' LIMIT 20`, collName)
		rows := sqlQueryRows(t, sql)
		assert.Greater(t, len(rows), 0)
	})

	t.Run("VectorSearch", func(t *testing.T) {
		rng := rand.New(rand.NewSource(99))
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = rng.Float32()
		}
		// Build vector JSON array for searchParams
		vecBytes, _ := json.Marshal(vec)

		body := map[string]any{
			"sql":              fmt.Sprintf("SELECT pk, l2_distance(embedding, $1) AS distance FROM %s ORDER BY distance LIMIT 5", collName),
			"consistencyLevel": "Strong",
			"searchParams": map[string]string{
				"$1":     string(vecBytes),
				"nprobe": "10",
			},
		}
		resp := postJSON(t, "/entities/sql_query", body)
		require.Equal(t, 0, resp.Code, "vector search failed: %s", resp.Message)
		var rows []map[string]any
		require.NoError(t, json.Unmarshal(resp.Data, &rows))
		require.Len(t, rows, 5)
		type hit struct {
			pk       int
			distance float64
		}
		want := make([]hit, rowCount)
		data := rand.New(rand.NewSource(42))
		for i := range want {
			var squared float64
			for _, query := range vec {
				delta := float64(data.Float32()) - float64(query)
				squared += delta * delta
			}
			want[i] = hit{i, math.Sqrt(squared)}
		}
		sort.Slice(want, func(i, j int) bool { return want[i].distance < want[j].distance })
		for i, row := range rows {
			assert.Equal(t, float64(want[i].pk), row["pk"])
			assert.InDelta(t, want[i].distance, row["distance"], 1e-5)
		}
	})

	t.Run("InvalidSQL", func(t *testing.T) {
		resp := sqlQuery(t, "THIS IS NOT SQL")
		assert.NotEqual(t, 0, resp.Code)
	})

	t.Run("EmptyResult", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT pk FROM %s WHERE age > 99999 LIMIT 10", collName))
		assert.Len(t, rows, 0)
	})

	t.Run("CountDistinct", func(t *testing.T) {
		rows := sqlQueryRows(t, fmt.Sprintf("SELECT count(DISTINCT age) AS n FROM %s", collName))
		require.Len(t, rows, 1)
		assert.Equal(t, float64(50), rows[0]["n"])
		assert.Len(t, rows, 1)
	})
	t.Run("JSONBenchBeforeFlush", checkJSONBenchOracle)
	t.Run("JSONBenchAfterFlushReload", func(t *testing.T) {
		for _, action := range []string{"flush", "release", "load"} {
			response := postJSON(t, "/collections/"+action, map[string]any{"collectionName": collName})
			require.Equal(t, 0, response.Code, response.Message)
		}
		require.Eventually(t, func() bool {
			response := postJSON(t, "/collections/get_load_state", map[string]any{"collectionName": collName})
			var state map[string]any
			return response.Code == 0 && json.Unmarshal(response.Data, &state) == nil && state["loadState"] == "LoadStateLoaded"
		}, time.Minute, time.Second)
		checkJSONBenchOracle(t)
	})
}

func checkJSONBenchOracle(t *testing.T) {
	t.Helper()
	query := func(sql string) []map[string]any {
		return sqlQueryRows(t, strings.ReplaceAll(sql, "bluesky", collName))
	}
	events := []string{"app.bsky.feed.post", "app.bsky.feed.repost", "app.bsky.feed.like"}
	wantEvents := map[string]float64{events[0]: 34, events[1]: 33, events[2]: 33}
	for _, sql := range []string{benchQ1, benchQ2} {
		rows := query(sql)
		require.Len(t, rows, 3)
		for _, row := range rows {
			require.Equal(t, wantEvents[row["event"].(string)], row["count"])
			if sql == benchQ2 {
				require.Equal(t, float64(4), row["users"])
			}
		}
	}
	wantHours := make(map[string]int)
	for i := 0; i < rowCount; i++ {
		wantHours[fmt.Sprintf("%d/%s", (i/2)%24, events[i%3])]++
	}
	rows := query(benchQ3)
	require.Len(t, rows, len(wantHours))
	for _, row := range rows {
		key := fmt.Sprintf("%.0f/%s", row["hour_of_day"], row["event"])
		require.Equal(t, float64(wantHours[key]), row["count"], key)
	}
	rows = query(benchQ4)
	require.Len(t, rows, 3)
	for i, user := range []string{"user_0", "user_3", "user_2"} {
		require.Equal(t, user, rows[i]["user_id"])
		require.Equal(t, time.Unix(int64(1704067200+i*3*1800), 0).UTC().Format(time.RFC3339), rows[i]["first_post_ts"])
	}
	rows = query(strings.Replace(benchQ5, "activity_span DESC LIMIT", "activity_span DESC, user_id ASC LIMIT", 1))
	require.Len(t, rows, 3)
	for i, user := range []string{"user_0", "user_3", "user_1"} {
		require.Equal(t, user, rows[i]["user_id"])
		span := float64(96 * 1800 * 1000)
		if i == 2 {
			span = 84 * 1800 * 1000
		}
		require.Equal(t, span, rows[i]["activity_span"])
	}
}
