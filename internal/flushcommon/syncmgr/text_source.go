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

package syncmgr

import (
	"context"
	"fmt"
	"path"
	"sort"
	"sync"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/flushcommon/metacache"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/retry"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

type TextFlushConfig struct {
	SegmentBasePath   string
	PartitionBasePath string
	CollectionID      int64
	PartitionID       int64
	TextFieldIDs      []int64
	TextLobPaths      []string
	ReadVersion       int64
}

type TextFlushResult struct {
	ManifestPath string
	NumRows      int64
}

type TextFlushSource interface {
	CurrentOffset() int64
	FlushTextData(ctx context.Context, startOffset, endOffset int64, config *TextFlushConfig) (*TextFlushResult, error)
	Release()
}

type TextFlushSourceCommitter interface {
	CommitTextFlush(targetOffset int64)
}

type TextSourceState int

const (
	TextSourceUnavailable TextSourceState = iota
	TextSourcePending
	TextSourceUsable
)

type TextSourceProvider interface {
	GetTextFlushSource(segmentID int64, targetOffset int64, endPos *msgpb.MsgPosition) (TextFlushSource, TextSourceState)
}

type TextReleaseHandoffSegment struct {
	SegmentID    int64
	TargetOffset int64
}

type TextReleaseHandoffProvider interface {
	PrepareTextReleaseHandoff(ctx context.Context, fenceTs uint64, segments []TextReleaseHandoffSegment) error
	IsReleaseAllowed(segmentID int64, checkpointTs uint64) bool
	IsReleasePrepared(segmentID int64, checkpointTs uint64) bool
	ClearReleasePrepared(segmentID int64)
	ReleasePreparedSegments() []int64
}

type TextSourceRegistry struct {
	mu        sync.RWMutex
	nextToken uint64
	providers map[string]map[uint64]TextSourceProvider
}

type TextSourceRegistration struct {
	registry *TextSourceRegistry
	channel  string
	token    uint64
}

func NewTextSourceRegistry() *TextSourceRegistry {
	return &TextSourceRegistry{
		providers: make(map[string]map[uint64]TextSourceProvider),
	}
}

func (r *TextSourceRegistry) Register(channel string, provider TextSourceProvider) *TextSourceRegistration {
	if provider == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nextToken++
	token := r.nextToken
	if _, ok := r.providers[channel]; !ok {
		r.providers[channel] = make(map[uint64]TextSourceProvider)
	}
	r.providers[channel][token] = provider
	return &TextSourceRegistration{
		registry: r,
		channel:  channel,
		token:    token,
	}
}

func (r *TextSourceRegistry) Unregister(registration *TextSourceRegistration) {
	if registration == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	providers, ok := r.providers[registration.channel]
	if !ok {
		return
	}
	delete(providers, registration.token)
	if len(providers) == 0 {
		delete(r.providers, registration.channel)
	}
}

func (r *TextSourceRegistry) ProviderCount(channel string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.providers[channel])
}

func (r *TextSourceRegistry) getProviders(channel string) []TextSourceProvider {
	r.mu.RLock()
	channelProviders := r.providers[channel]
	tokens := make([]uint64, 0, len(channelProviders))
	for token := range channelProviders {
		tokens = append(tokens, token)
	}
	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i] < tokens[j]
	})
	providers := make([]TextSourceProvider, 0, len(tokens))
	for _, token := range tokens {
		providers = append(providers, channelProviders[token])
	}
	r.mu.RUnlock()
	return providers
}

func (r *TextSourceRegistry) PrepareTextReleaseHandoff(ctx context.Context, channel string, fenceTs uint64, segments []TextReleaseHandoffSegment) error {
	handoffProviders := make([]TextReleaseHandoffProvider, 0)
	for _, provider := range r.getProviders(channel) {
		handoffProvider, ok := provider.(TextReleaseHandoffProvider)
		if !ok {
			continue
		}
		handoffProviders = append(handoffProviders, handoffProvider)
	}
	if len(handoffProviders) == 0 {
		return merr.WrapErrChannelNotAvailable(channel, "no local TEXT release handoff provider")
	}

	for _, handoffProvider := range handoffProviders {
		if err := handoffProvider.PrepareTextReleaseHandoff(ctx, fenceTs, segments); err != nil {
			return err
		}
	}

	for _, segment := range segments {
		if segment.TargetOffset > 0 {
			if !r.IsReleasePrepared(channel, segment.SegmentID, fenceTs) {
				return merr.WrapErrSegmentNotFound(segment.SegmentID, "TEXT release handoff source is not prepared")
			}
			continue
		}
		if !r.IsReleaseAllowed(channel, segment.SegmentID, fenceTs) {
			return merr.WrapErrSegmentNotFound(segment.SegmentID, "TEXT release handoff is not allowed")
		}
	}
	return nil
}

func (r *TextSourceRegistry) IsReleaseAllowed(channel string, segmentID int64, checkpointTs uint64) bool {
	for _, provider := range r.getProviders(channel) {
		handoffProvider, ok := provider.(TextReleaseHandoffProvider)
		if !ok {
			continue
		}
		if handoffProvider.IsReleaseAllowed(segmentID, checkpointTs) {
			return true
		}
	}
	return false
}

func (r *TextSourceRegistry) IsReleasePrepared(channel string, segmentID int64, checkpointTs uint64) bool {
	for _, provider := range r.getProviders(channel) {
		handoffProvider, ok := provider.(TextReleaseHandoffProvider)
		if !ok {
			continue
		}
		if handoffProvider.IsReleasePrepared(segmentID, checkpointTs) {
			return true
		}
	}
	return false
}

func (r *TextSourceRegistry) ClearReleasePrepared(channel string, segmentID int64) {
	for _, provider := range r.getProviders(channel) {
		handoffProvider, ok := provider.(TextReleaseHandoffProvider)
		if !ok {
			continue
		}
		handoffProvider.ClearReleasePrepared(segmentID)
	}
}

func (r *TextSourceRegistry) ReleasePreparedSegments(channel string) []int64 {
	var segments []int64
	for _, provider := range r.getProviders(channel) {
		handoffProvider, ok := provider.(TextReleaseHandoffProvider)
		if !ok {
			continue
		}
		segments = append(segments, handoffProvider.ReleasePreparedSegments()...)
	}
	return lo.Uniq(segments)
}

func (r *TextSourceRegistry) Resolve(channel string, segmentID int64, targetOffset int64, endPos *msgpb.MsgPosition) (TextFlushSource, TextSourceState) {
	hasPending := false
	for _, provider := range r.getProviders(channel) {
		if provider == nil {
			continue
		}
		source, state := provider.GetTextFlushSource(segmentID, targetOffset, endPos)
		if source == nil {
			continue
		}
		switch state {
		case TextSourceUsable:
			return source, TextSourceUsable
		case TextSourcePending:
			hasPending = true
			source.Release()
		default:
			source.Release()
		}
	}
	if hasPending {
		return nil, TextSourcePending
	}
	return nil, TextSourceUnavailable
}

var defaultTextSourceRegistry = NewTextSourceRegistry()

func DefaultTextSourceRegistry() *TextSourceRegistry {
	return defaultTextSourceRegistry
}

type TextSourceSyncTask struct {
	collectionID  int64
	partitionID   int64
	segmentID     int64
	channelName   string
	startPosition *msgpb.MsgPosition
	checkpoint    *msgpb.MsgPosition
	batchRows     int64
	targetOffset  int64
	level         datapb.SegmentLevel
	isFlush       bool
	isDrop        bool

	metacache  metacache.MetaCache
	metaWriter MetaWriter
	schema     *schemapb.CollectionSchema
	source     TextFlushSource

	chunkManager storage.ChunkManager
	manifestPath string
	flushedSize  int64

	writeRetryOpts  []retry.Option
	failureCallback func(error)
	tr              *timerecord.TimeRecorder
}

func NewTextSourceSyncTask() *TextSourceSyncTask {
	return new(TextSourceSyncTask)
}

func (t *TextSourceSyncTask) WithCollectionID(collectionID int64) *TextSourceSyncTask {
	t.collectionID = collectionID
	return t
}

func (t *TextSourceSyncTask) WithPartitionID(partitionID int64) *TextSourceSyncTask {
	t.partitionID = partitionID
	return t
}

func (t *TextSourceSyncTask) WithSegmentID(segmentID int64) *TextSourceSyncTask {
	t.segmentID = segmentID
	return t
}

func (t *TextSourceSyncTask) WithChannelName(channelName string) *TextSourceSyncTask {
	t.channelName = channelName
	return t
}

func (t *TextSourceSyncTask) WithStartPosition(position *msgpb.MsgPosition) *TextSourceSyncTask {
	t.startPosition = position
	return t
}

func (t *TextSourceSyncTask) WithCheckpoint(position *msgpb.MsgPosition) *TextSourceSyncTask {
	t.checkpoint = position
	return t
}

func (t *TextSourceSyncTask) WithBatchRows(batchRows int64) *TextSourceSyncTask {
	t.batchRows = batchRows
	return t
}

func (t *TextSourceSyncTask) WithTargetOffset(targetOffset int64) *TextSourceSyncTask {
	t.targetOffset = targetOffset
	return t
}

func (t *TextSourceSyncTask) WithLevel(level datapb.SegmentLevel) *TextSourceSyncTask {
	t.level = level
	return t
}

func (t *TextSourceSyncTask) WithFlush() *TextSourceSyncTask {
	t.isFlush = true
	return t
}

func (t *TextSourceSyncTask) WithDrop() *TextSourceSyncTask {
	t.isDrop = true
	return t
}

func (t *TextSourceSyncTask) WithMetaCache(metacache metacache.MetaCache) *TextSourceSyncTask {
	t.metacache = metacache
	return t
}

func (t *TextSourceSyncTask) WithMetaWriter(metaWriter MetaWriter) *TextSourceSyncTask {
	t.metaWriter = metaWriter
	return t
}

func (t *TextSourceSyncTask) WithSchema(schema *schemapb.CollectionSchema) *TextSourceSyncTask {
	t.schema = schema
	return t
}

func (t *TextSourceSyncTask) WithSource(source TextFlushSource) *TextSourceSyncTask {
	t.source = source
	return t
}

func (t *TextSourceSyncTask) WithChunkManager(cm storage.ChunkManager) *TextSourceSyncTask {
	t.chunkManager = cm
	return t
}

func (t *TextSourceSyncTask) WithWriteRetryOptions(opts ...retry.Option) *TextSourceSyncTask {
	t.writeRetryOpts = opts
	return t
}

func (t *TextSourceSyncTask) WithFailureCallback(callback func(error)) *TextSourceSyncTask {
	t.failureCallback = callback
	return t
}

func (t *TextSourceSyncTask) SegmentID() int64 {
	return t.segmentID
}

func (t *TextSourceSyncTask) Checkpoint() *msgpb.MsgPosition {
	return t.checkpoint
}

func (t *TextSourceSyncTask) StartPosition() *msgpb.MsgPosition {
	return t.startPosition
}

func (t *TextSourceSyncTask) ChannelName() string {
	return t.channelName
}

func (t *TextSourceSyncTask) IsFlush() bool {
	return t.isFlush
}

func (t *TextSourceSyncTask) IsDrop() bool {
	return t.isDrop
}

func (t *TextSourceSyncTask) ManifestPath() string {
	return t.manifestPath
}

func (t *TextSourceSyncTask) BatchRows() int64 {
	return t.batchRows
}

func (t *TextSourceSyncTask) TargetOffset() int64 {
	return t.targetOffset
}

func (t *TextSourceSyncTask) HandleError(err error) {
	if t.failureCallback != nil {
		t.failureCallback(err)
	}
	metrics.DataNodeFlushBufferCount.WithLabelValues(paramtable.GetStringNodeID(), metrics.FailLabel, t.level.String()).Inc()
}

func (t *TextSourceSyncTask) ReleaseSource() {
	if t.source != nil {
		t.source.Release()
		t.source = nil
	}
}

func (t *TextSourceSyncTask) Run(ctx context.Context) (err error) {
	t.tr = timerecord.NewTimeRecorder("textSourceSyncTask")
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", t.collectionID),
		zap.Int64("partitionID", t.partitionID),
		zap.Int64("segmentID", t.segmentID),
		zap.String("channel", t.channelName),
	)
	commitSource := false
	defer func() {
		committer, shouldCommit := t.source.(TextFlushSourceCommitter)
		t.ReleaseSource()
		if commitSource && shouldCommit && (t.IsFlush() || t.IsDrop()) {
			committer.CommitTextFlush(t.targetOffset)
		}
		if err != nil {
			t.HandleError(err)
		}
	}()

	segment, ok := t.metacache.GetSegmentByID(t.segmentID)
	if !ok {
		if t.isDrop {
			log.Info("segment dropped, discard text source sync task")
			return nil
		}
		log.Warn("segment not found in metacache")
		return nil
	}
	if t.source == nil {
		return errors.New("text flush source is nil")
	}
	if t.source.CurrentOffset() < t.targetOffset {
		return errors.Errorf("text flush source is behind target offset, current=%d target=%d", t.source.CurrentOffset(), t.targetOffset)
	}

	expectedRows := t.targetOffset - segment.FlushedRows()
	if expectedRows < 0 {
		return errors.Errorf("text source target offset is behind flushed rows, flushedRows=%d targetOffset=%d segmentID=%d",
			segment.FlushedRows(), t.targetOffset, t.segmentID)
	}
	if expectedRows == 0 {
		t.manifestPath = segment.ManifestPath()
	} else {
		config := t.buildFlushConfig(segment)
		result, err := t.source.FlushTextData(ctx, segment.FlushedRows(), t.targetOffset, config)
		if err != nil {
			return errors.Wrap(err, "flush text source data")
		}
		if result == nil || result.ManifestPath == "" {
			return errors.New("text source flush returned empty manifest")
		}
		if result.NumRows != expectedRows {
			return errors.Errorf("text source flush row count mismatch, expected=%d actual=%d flushedRows=%d targetOffset=%d segmentID=%d",
				expectedRows, result.NumRows, segment.FlushedRows(), t.targetOffset, t.segmentID)
		}
		t.manifestPath = result.ManifestPath
	}
	t.flushedSize = expectedRows

	if t.metaWriter != nil {
		if err := t.metaWriter.UpdateTextSourceSync(ctx, t); err != nil {
			return err
		}
	}

	actions := make([]metacache.SegmentAction, 0, 3)
	if t.batchRows > 0 {
		actions = append(actions, metacache.FinishSyncing(t.batchRows))
	}
	if t.manifestPath != "" {
		actions = append(actions, metacache.UpdateManifestPath(t.manifestPath))
	}
	if t.IsFlush() {
		actions = append(actions, metacache.UpdateState(commonpb.SegmentState_Flushed))
	}
	t.metacache.UpdateSegments(metacache.MergeSegmentAction(actions...), metacache.WithSegmentIDs(t.segmentID))
	if t.isDrop {
		t.metacache.RemoveSegments(metacache.WithSegmentIDs(t.segmentID))
		log.Info("dropped text source segment removed")
	}
	commitSource = true

	metrics.DataNodeWriteDataCount.WithLabelValues(paramtable.GetStringNodeID(), metrics.StreamingDataSourceLabel, metrics.InsertLabel, fmt.Sprint(t.collectionID)).Add(float64(t.batchRows))
	metrics.DataNodeFlushedRows.WithLabelValues(paramtable.GetStringNodeID(), metrics.StreamingDataSourceLabel).Add(float64(t.batchRows))
	metrics.DataNodeFlushBufferCount.WithLabelValues(paramtable.GetStringNodeID(), metrics.SuccessLabel, t.level.String()).Inc()
	log.Info("text source sync task done",
		zap.Int64("targetOffset", t.targetOffset),
		zap.Int64("batchRows", t.batchRows),
		zap.String("manifestPath", t.manifestPath),
		zap.Duration("timeTaken", t.tr.ElapseSpan()))
	return nil
}

func (t *TextSourceSyncTask) buildFlushConfig(segment *metacache.SegmentInfo) *TextFlushConfig {
	segmentBasePath := path.Join(t.chunkManager.RootPath(), common.SegmentInsertLogPath,
		metautil.JoinIDPath(t.collectionID, t.partitionID, t.segmentID))
	partitionBasePath := path.Join(t.chunkManager.RootPath(), common.SegmentInsertLogPath,
		metautil.JoinIDPath(t.collectionID, t.partitionID))

	var textFieldIDs []int64
	var textLobPaths []string
	if t.schema != nil {
		for _, field := range t.schema.GetFields() {
			if field.GetDataType() == schemapb.DataType_Text {
				fieldID := field.GetFieldID()
				textFieldIDs = append(textFieldIDs, fieldID)
				textLobPaths = append(textLobPaths, fmt.Sprintf("%s/lobs/%d", partitionBasePath, fieldID))
			}
		}
	}

	return &TextFlushConfig{
		SegmentBasePath:   segmentBasePath,
		PartitionBasePath: partitionBasePath,
		CollectionID:      t.collectionID,
		PartitionID:       t.partitionID,
		TextFieldIDs:      textFieldIDs,
		TextLobPaths:      textLobPaths,
		ReadVersion:       manifestVersion(segment.ManifestPath()),
	}
}

func manifestVersion(manifestPath string) int64 {
	if manifestPath == "" {
		return -1
	}
	if _, version, err := packedManifestVersion(manifestPath); err == nil {
		return version
	}
	return -1
}

func packedManifestVersion(manifestPath string) (string, int64, error) {
	return packed.UnmarshalManifestPath(manifestPath)
}

func (t *TextSourceSyncTask) startPositions() []*datapb.SegmentStartPosition {
	startPos := lo.Map(t.metacache.GetSegmentsBy(
		metacache.WithSegmentState(commonpb.SegmentState_Growing, commonpb.SegmentState_Sealed, commonpb.SegmentState_Flushing),
		metacache.WithLevel(datapb.SegmentLevel_L1),
		metacache.WithStartPosNotRecorded(),
	), func(info *metacache.SegmentInfo, _ int) *datapb.SegmentStartPosition {
		return &datapb.SegmentStartPosition{
			SegmentID:     info.SegmentID(),
			StartPosition: info.StartPosition(),
		}
	})
	if t.level == datapb.SegmentLevel_L0 {
		startPos = append(startPos, &datapb.SegmentStartPosition{SegmentID: t.segmentID, StartPosition: t.startPosition})
	}
	return startPos
}
