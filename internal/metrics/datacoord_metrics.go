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

package metrics

import (
	"github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/prometheus/client_golang/prometheus"
)

const (
	CompactTypeI         = "compactTypeI"
	CompactTypeII        = "compactTypeII"
	CompactInputLabel    = "input"
	CompactInput2Label   = "input2"
	CompactOutputLabel   = "output"
	compactIOLabelName   = "IO"
	compactTypeLabelName = "compactType"
)

var (
	//DataCoordNumDataNodes records the num of data nodes managed by DataCoord.
	DataCoordNumDataNodes = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "datanode_num",
			Help:      "number of data nodes",
		}, []string{})

	DataCoordNumSegments = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "segment_num",
			Help:      "number of segments",
		}, []string{
			segmentStateLabelName,
		})

	//DataCoordCollectionNum records the num of collections managed by DataCoord.
	DataCoordNumCollections = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "collection_num",
			Help:      "number of collections",
		}, []string{})

	DataCoordNumStoredRows = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "stored_rows_num",
			Help:      "number of stored rows",
		}, []string{})

	DataCoordNumStoredRowsCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "stored_rows_count",
			Help:      "count of all stored rows ever",
		}, []string{})

	DataCoordSyncEpoch = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "sync_epoch_time",
			Help:      "synchronized unix epoch per physical channel",
		}, []string{channelNameLabelName})

	/* hard to implement, commented now
	DataCoordSegmentSizeRatio = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "segment_size_ratio",
			Help:      "size ratio compared to the configuration size",
			Buckets:   prometheus.LinearBuckets(0.0, 0.1, 15),
		}, []string{})

	DataCoordSegmentFlushDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "segment_flush_duration",
			Help:      "time spent on each segment flush",
			Buckets:   []float64{0.1, 0.5, 1, 5, 10, 20, 50, 100, 250, 500, 1000, 3600, 5000, 10000}, // unit seconds
		}, []string{})

	DataCoordCompactDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "segment_compact_duration",
			Help:      "time spent on each segment flush",
			Buckets:   []float64{0.1, 0.5, 1, 5, 10, 20, 50, 100, 250, 500, 1000, 3600, 5000, 10000}, // unit seconds
		}, []string{compactTypeLabelName})

	DataCoordCompactLoad = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "compaction_load",
			Help:      "Information on the input and output of compaction",
		}, []string{compactTypeLabelName, compactIOLabelName})

	DataCoordNumCompactionTask = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: milvusNamespace,
			Subsystem: typeutil.DataCoordRole,
			Name:      "num_compaction_tasks",
			Help:      "Number of compaction tasks currently",
		}, []string{statusLabelName})
	*/

)

//RegisterDataCoord registers DataCoord metrics
func RegisterDataCoord(registry *prometheus.Registry) {
	registry.MustRegister(DataCoordNumDataNodes)
	registry.MustRegister(DataCoordNumSegments)
	registry.MustRegister(DataCoordNumCollections)
	registry.MustRegister(DataCoordNumStoredRows)
	registry.MustRegister(DataCoordNumStoredRowsCounter)
	registry.MustRegister(DataCoordSyncEpoch)
}
