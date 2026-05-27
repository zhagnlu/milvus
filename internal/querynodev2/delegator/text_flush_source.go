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

package delegator

import (
	"context"
	"sync"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/flushcommon/syncmgr"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

var errTextSourceProviderClosed = errors.New("text source provider is closed")

type delegatorTextSourceProvider struct {
	segmentManager  segments.SegmentManager
	waitFence       func(context.Context, uint64) error
	mu              sync.Mutex
	cond            *sync.Cond
	closing         bool
	deactivated     bool
	registration    *syncmgr.TextSourceRegistration
	active          int
	retained        map[int64]*retainedTextFlushSource
	releaseAllowed  map[int64]uint64
	releasePrepared map[int64]int64
}

func newDelegatorTextSourceProvider(segmentManager segments.SegmentManager, waitFence func(context.Context, uint64) error) *delegatorTextSourceProvider {
	provider := &delegatorTextSourceProvider{
		segmentManager:  segmentManager,
		waitFence:       waitFence,
		retained:        make(map[int64]*retainedTextFlushSource),
		releaseAllowed:  make(map[int64]uint64),
		releasePrepared: make(map[int64]int64),
	}
	provider.cond = sync.NewCond(&provider.mu)
	return provider
}

func (p *delegatorTextSourceProvider) SetRegistration(registration *syncmgr.TextSourceRegistration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.registration = registration
}

func (p *delegatorTextSourceProvider) GetTextFlushSource(segmentID int64, targetOffset int64, _ *msgpb.MsgPosition) (syncmgr.TextFlushSource, syncmgr.TextSourceState) {
	if !p.acquireLease() {
		return nil, syncmgr.TextSourceUnavailable
	}
	segment := p.segmentManager.GetGrowing(segmentID)
	retained := false
	if segment == nil {
		var ok bool
		segment, ok = p.getRetained(segmentID)
		if !ok {
			p.releaseLease()
			return nil, syncmgr.TextSourceUnavailable
		}
		retained = true
	} else if p.isDeactivated() {
		p.releaseLease()
		return nil, syncmgr.TextSourceUnavailable
	}
	if err := segment.PinIfNotReleased(); err != nil {
		p.releaseLease()
		return nil, syncmgr.TextSourceUnavailable
	}
	source := &delegatorTextFlushSource{segmentID: segmentID, segment: segment, provider: p, targetOffset: targetOffset, retained: retained}
	if p.currentOffset(segment) < targetOffset {
		return source, syncmgr.TextSourcePending
	}
	return source, syncmgr.TextSourceUsable
}

func (p *delegatorTextSourceProvider) PrepareTextReleaseHandoff(ctx context.Context, fenceTs uint64, segments []syncmgr.TextReleaseHandoffSegment) error {
	if p.isDeactivated() {
		return p.prepareDeactivatedTextReleaseHandoff(fenceTs, segments)
	}
	if p.waitFence != nil && fenceTs > 0 {
		if err := p.waitFence(ctx, fenceTs); err != nil {
			return err
		}
	}
	snapshot := p.snapshotRetained(segments)
	allowedSegments := make([]syncmgr.TextReleaseHandoffSegment, 0, len(segments))
	preparedSegments := make([]syncmgr.TextReleaseHandoffSegment, 0, len(segments))
	for _, segment := range segments {
		allowedSegments = append(allowedSegments, segment)
		if segment.TargetOffset <= 0 {
			continue
		}
		if err := p.registerRetained(segment.SegmentID, segment.TargetOffset); err != nil {
			if errors.Is(err, merr.ErrSegmentNotFound) {
				continue
			}
			p.rollbackRetained(snapshot)
			return err
		}
		preparedSegments = append(preparedSegments, segment)
	}
	p.markReleaseAllowed(fenceTs, allowedSegments)
	p.markReleasePrepared(fenceTs, preparedSegments)
	return nil
}

func (p *delegatorTextSourceProvider) prepareDeactivatedTextReleaseHandoff(fenceTs uint64, segments []syncmgr.TextReleaseHandoffSegment) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, segment := range segments {
		retained, ok := p.retained[segment.SegmentID]
		if !ok {
			continue
		}
		if segment.TargetOffset > retained.targetOffset {
			continue
		}
		if current, ok := p.releaseAllowed[segment.SegmentID]; !ok || current < fenceTs {
			p.releaseAllowed[segment.SegmentID] = fenceTs
		}
		if segment.TargetOffset <= 0 {
			continue
		}
		if current, ok := p.releasePrepared[segment.SegmentID]; !ok || current < segment.TargetOffset {
			p.releasePrepared[segment.SegmentID] = segment.TargetOffset
		}
	}
	return nil
}

func (p *delegatorTextSourceProvider) registerRetained(segmentID int64, targetOffset int64) error {
	p.mu.Lock()
	if p.closing {
		p.mu.Unlock()
		return errTextSourceProviderClosed
	}
	if retained, ok := p.retained[segmentID]; ok {
		if retained.targetOffset < targetOffset {
			retained.targetOffset = targetOffset
		}
		p.mu.Unlock()
		return nil
	}
	p.mu.Unlock()

	segment := p.segmentManager.GetGrowing(segmentID)
	if segment == nil {
		return merr.WrapErrSegmentNotFound(segmentID)
	}
	if err := segment.PinIfNotReleased(); err != nil {
		return err
	}
	currentOffset := p.currentOffset(segment)
	if currentOffset < targetOffset {
		segment.Unpin()
		return errors.Errorf("TEXT growing segment %d is behind target offset, current=%d target=%d", segmentID, currentOffset, targetOffset)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closing {
		segment.Unpin()
		return errTextSourceProviderClosed
	}
	if retained, ok := p.retained[segmentID]; ok {
		if retained.targetOffset < targetOffset {
			retained.targetOffset = targetOffset
		}
		segment.Unpin()
		return nil
	}
	p.retained[segmentID] = &retainedTextFlushSource{
		segment:      segment,
		targetOffset: targetOffset,
	}
	return nil
}

type retainedSnapshot struct {
	existed      bool
	source       *retainedTextFlushSource
	targetOffset int64
}

func (p *delegatorTextSourceProvider) snapshotRetained(segments []syncmgr.TextReleaseHandoffSegment) map[int64]retainedSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()

	snapshot := make(map[int64]retainedSnapshot, len(segments))
	for _, segment := range segments {
		if _, ok := snapshot[segment.SegmentID]; ok {
			continue
		}
		retained, existed := p.retained[segment.SegmentID]
		entry := retainedSnapshot{existed: existed, source: retained}
		if existed {
			entry.targetOffset = retained.targetOffset
		}
		snapshot[segment.SegmentID] = entry
	}
	return snapshot
}

func (p *delegatorTextSourceProvider) rollbackRetained(snapshot map[int64]retainedSnapshot) {
	var toUnpin []segments.Segment
	p.mu.Lock()
	for segmentID, entry := range snapshot {
		current, exists := p.retained[segmentID]
		if entry.existed {
			p.retained[segmentID] = entry.source
			entry.source.targetOffset = entry.targetOffset
			if exists && current != entry.source {
				toUnpin = append(toUnpin, current.segment)
			}
			continue
		}
		if exists {
			delete(p.retained, segmentID)
			toUnpin = append(toUnpin, current.segment)
		}
	}
	p.mu.Unlock()
	for _, segment := range toUnpin {
		segment.Unpin()
	}
}

func (p *delegatorTextSourceProvider) markReleaseAllowed(fenceTs uint64, segments []syncmgr.TextReleaseHandoffSegment) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, segment := range segments {
		if current, ok := p.releaseAllowed[segment.SegmentID]; !ok || current < fenceTs {
			p.releaseAllowed[segment.SegmentID] = fenceTs
		}
	}
}

func (p *delegatorTextSourceProvider) markReleasePrepared(fenceTs uint64, segments []syncmgr.TextReleaseHandoffSegment) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, segment := range segments {
		if current, ok := p.releaseAllowed[segment.SegmentID]; !ok || current < fenceTs {
			p.releaseAllowed[segment.SegmentID] = fenceTs
		}
		if current, ok := p.releasePrepared[segment.SegmentID]; !ok || current < segment.TargetOffset {
			p.releasePrepared[segment.SegmentID] = segment.TargetOffset
		}
	}
}

func (p *delegatorTextSourceProvider) IsReleaseAllowed(segmentID int64, checkpointTs uint64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.isReleaseAllowedLocked(segmentID, checkpointTs)
}

func (p *delegatorTextSourceProvider) IsReleasePrepared(segmentID int64, checkpointTs uint64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.releasePrepared[segmentID]; ok {
		return p.isReleaseAllowedLocked(segmentID, checkpointTs)
	}
	return false
}

func (p *delegatorTextSourceProvider) isReleaseAllowedLocked(segmentID int64, checkpointTs uint64) bool {
	fenceTs, ok := p.releaseAllowed[segmentID]
	if !ok {
		return false
	}
	return fenceTs == 0 || checkpointTs == 0 || checkpointTs <= fenceTs
}

func (p *delegatorTextSourceProvider) ClearReleasePrepared(segmentID int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.releasePrepared, segmentID)
	delete(p.releaseAllowed, segmentID)
}

func (p *delegatorTextSourceProvider) ReleasePreparedSegments() []int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	segments := make([]int64, 0, len(p.releasePrepared))
	for segmentID := range p.releasePrepared {
		segments = append(segments, segmentID)
	}
	return segments
}

func (p *delegatorTextSourceProvider) getRetained(segmentID int64) (segments.Segment, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	retained, ok := p.retained[segmentID]
	if !ok {
		return nil, false
	}
	return retained.segment, true
}

func (p *delegatorTextSourceProvider) releaseRetainedIfComplete(segmentID int64, targetOffset int64) {
	p.mu.Lock()
	retained, ok := p.retained[segmentID]
	if !ok || targetOffset < retained.targetOffset {
		p.mu.Unlock()
		return
	}
	delete(p.retained, segmentID)
	registration := p.unregisterIfInactiveLocked()
	p.mu.Unlock()
	retained.segment.Unpin()
	retained.segment.Release(context.Background())
	syncmgr.DefaultTextSourceRegistry().Unregister(registration)
}

func (p *delegatorTextSourceProvider) acquireLease() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closing {
		return false
	}
	p.active++
	return true
}

func (p *delegatorTextSourceProvider) releaseLease() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.active > 0 {
		p.active--
	}
	if p.closing && p.active == 0 {
		p.cond.Broadcast()
	}
}

func (p *delegatorTextSourceProvider) isDeactivated() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.deactivated
}

func (p *delegatorTextSourceProvider) Deactivate() {
	p.mu.Lock()
	p.deactivated = true
	registration := p.unregisterIfInactiveLocked()
	p.mu.Unlock()
	syncmgr.DefaultTextSourceRegistry().Unregister(registration)
}

func (p *delegatorTextSourceProvider) Close() {
	p.mu.Lock()
	p.closing = true
	for p.active > 0 {
		p.cond.Wait()
	}
	retained := p.retained
	p.retained = make(map[int64]*retainedTextFlushSource)
	p.releaseAllowed = make(map[int64]uint64)
	p.releasePrepared = make(map[int64]int64)
	registration := p.registration
	p.registration = nil
	p.mu.Unlock()
	for _, source := range retained {
		source.segment.Unpin()
	}
	syncmgr.DefaultTextSourceRegistry().Unregister(registration)
}

func (p *delegatorTextSourceProvider) unregisterIfInactiveLocked() *syncmgr.TextSourceRegistration {
	if !p.deactivated || len(p.retained) > 0 {
		return nil
	}
	registration := p.registration
	p.registration = nil
	p.releaseAllowed = make(map[int64]uint64)
	p.releasePrepared = make(map[int64]int64)
	return registration
}

func (p *delegatorTextSourceProvider) currentOffset(segment segments.Segment) int64 {
	if segment == nil {
		return 0
	}
	return segment.InsertCount()
}

type retainedTextFlushSource struct {
	segment      segments.Segment
	targetOffset int64
}

type delegatorTextFlushSource struct {
	segmentID    int64
	segment      segments.Segment
	provider     *delegatorTextSourceProvider
	targetOffset int64
	retained     bool
	once         sync.Once
}

func (s *delegatorTextFlushSource) CurrentOffset() int64 {
	if s.provider != nil {
		return s.provider.currentOffset(s.segment)
	}
	if s.segment == nil {
		return 0
	}
	return s.segment.InsertCount()
}

func (s *delegatorTextFlushSource) FlushTextData(ctx context.Context, startOffset, endOffset int64, config *syncmgr.TextFlushConfig) (*syncmgr.TextFlushResult, error) {
	result, err := s.segment.FlushData(ctx, startOffset, endOffset, &segments.FlushConfig{
		SegmentBasePath:   config.SegmentBasePath,
		PartitionBasePath: config.PartitionBasePath,
		CollectionID:      config.CollectionID,
		PartitionID:       config.PartitionID,
		TextFieldIDs:      config.TextFieldIDs,
		TextLobPaths:      config.TextLobPaths,
		ReadVersion:       config.ReadVersion,
	})
	if err != nil || result == nil {
		return nil, err
	}
	return &syncmgr.TextFlushResult{
		ManifestPath: result.ManifestPath,
		NumRows:      result.NumRows,
	}, nil
}

func (s *delegatorTextFlushSource) Release() {
	s.once.Do(func() {
		if s.segment != nil {
			s.segment.Unpin()
		}
		if s.provider != nil {
			s.provider.releaseLease()
		}
	})
}

func (s *delegatorTextFlushSource) CommitTextFlush(targetOffset int64) {
	if s.retained && s.provider != nil {
		s.provider.releaseRetainedIfComplete(s.segmentID, targetOffset)
	}
}
