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

package querycoordv2

import (
	"context"
	"sort"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus/internal/flushcommon/writebuffer"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/streamingcoord/server/broadcaster"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/streaming/util/message"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

var _ task.TextReleaseDrainer = (*textReleaseDrainer)(nil)

type broadcastStarter func(ctx context.Context, collectionID int64) (broadcaster.BroadcastAPI, error)

type textFlushProgressGetter interface {
	GetTextFlushProgress(ctx context.Context, vchannel string, segmentIDs []int64, fenceTs uint64) ([]writebuffer.TextFlushSegmentProgress, error)
}

type textFlushProgressGetterFunc func(ctx context.Context, vchannel string, segmentIDs []int64, fenceTs uint64) ([]writebuffer.TextFlushSegmentProgress, error)

func (f textFlushProgressGetterFunc) GetTextFlushProgress(ctx context.Context, vchannel string, segmentIDs []int64, fenceTs uint64) ([]writebuffer.TextFlushSegmentProgress, error) {
	return f(ctx, vchannel, segmentIDs, fenceTs)
}

// textReleaseDrainer runs WAL fence + same-process WAL flusher handoff before TEXT growing sources are released.
type textReleaseDrainer struct {
	broker                  meta.Broker
	targetMgr               meta.TargetManagerInterface
	startBroadcast          broadcastStarter
	textFlushProgressGetter textFlushProgressGetter
}

func newTextReleaseDrainer(
	broker meta.Broker,
	targetMgr meta.TargetManagerInterface,
	startBroadcast broadcastStarter,
	textFlushProgressGetter textFlushProgressGetter,
) *textReleaseDrainer {
	return &textReleaseDrainer{
		broker:                  broker,
		targetMgr:               targetMgr,
		startBroadcast:          startBroadcast,
		textFlushProgressGetter: textFlushProgressGetter,
	}
}

func (d *textReleaseDrainer) DrainTextReleaseChannels(ctx context.Context, collectionID int64, channels []string) (map[string]uint64, error) {
	coll, err := d.broker.DescribeCollection(ctx, collectionID)
	if err != nil {
		return nil, err
	}

	broadcastAPI, err := d.startBroadcast(ctx, collectionID)
	if err != nil {
		return nil, err
	}
	defer broadcastAPI.Close()

	return d.DrainChannels(ctx, broadcastAPI, coll, channels)
}

func (d *textReleaseDrainer) DrainTextReleaseSegments(ctx context.Context, collectionID int64, segmentsByChannel map[string][]int64) (map[string]uint64, error) {
	segmentsByChannel = normalizeTextReleaseSegments(segmentsByChannel)
	if len(segmentsByChannel) == 0 {
		return nil, nil
	}

	coll, err := d.broker.DescribeCollection(ctx, collectionID)
	if err != nil {
		return nil, err
	}

	channels := make([]string, 0, len(segmentsByChannel))
	for channel := range segmentsByChannel {
		channels = append(channels, channel)
	}

	broadcastAPI, err := d.startBroadcast(ctx, collectionID)
	if err != nil {
		return nil, err
	}
	defer broadcastAPI.Close()

	return d.drainChannels(ctx, broadcastAPI, coll, channels, segmentsByChannel)
}

func (d *textReleaseDrainer) DrainChannels(
	ctx context.Context,
	broadcastAPI broadcaster.BroadcastAPI,
	coll *milvuspb.DescribeCollectionResponse,
	vchannels []string,
) (map[string]uint64, error) {
	return d.drainChannels(ctx, broadcastAPI, coll, vchannels, nil)
}

func (d *textReleaseDrainer) drainChannels(
	ctx context.Context,
	broadcastAPI broadcaster.BroadcastAPI,
	coll *milvuspb.DescribeCollectionResponse,
	vchannels []string,
	releaseSegmentsByChannel map[string][]int64,
) (map[string]uint64, error) {
	if !hasTextField(coll.GetSchema()) {
		return nil, nil
	}

	vchannels = lo.Uniq(lo.Filter(vchannels, func(channel string, _ int) bool {
		return channel != ""
	}))
	if releaseSegmentsByChannel != nil {
		releaseSegmentsByChannel = normalizeTextReleaseSegments(releaseSegmentsByChannel)
		vchannels = lo.Filter(vchannels, func(channel string, _ int) bool {
			return len(releaseSegmentsByChannel[channel]) > 0
		})
	}
	sort.Strings(vchannels)
	if len(vchannels) == 0 {
		return nil, nil
	}

	log.Ctx(ctx).Info("append release fence for TEXT collection",
		zap.Int64("collectionID", coll.GetCollectionID()),
		zap.Strings("channels", vchannels))

	textReleaseFenceKey, textReleaseFenceValue := message.TextReleaseFenceProperty()
	msg := message.NewManualFlushMessageBuilderV2().
		WithHeader(&message.ManualFlushMessageHeader{
			CollectionId: coll.GetCollectionID(),
		}).
		WithBody(&message.ManualFlushMessageBody{}).
		WithProperty(textReleaseFenceKey, textReleaseFenceValue).
		WithBroadcast(vchannels).
		MustBuildBroadcast()
	result, err := broadcastAPI.Broadcast(ctx, msg)
	if err != nil {
		return nil, err
	}

	fenceTs := make(map[string]uint64, len(vchannels))
	fencedSegments := make(map[string][]int64, len(vchannels))
	for _, channel := range vchannels {
		appendResult := result.GetAppendResult(channel)
		if appendResult == nil || appendResult.TimeTick == 0 {
			return nil, errors.Errorf("release fence append result missing for channel %s", channel)
		}
		fenceTs[channel] = appendResult.TimeTick
		if appendResult.Extra != nil {
			var extra message.ManualFlushExtraResponse
			if err := appendResult.GetExtra(&extra); err != nil {
				return nil, errors.Wrapf(err, "failed to get release fence segment ids for channel %s", channel)
			}
			fencedSegments[channel] = extra.GetSegmentIds()
		}
	}

	prepareSegments := fencedSegments
	if releaseSegmentsByChannel != nil {
		prepareSegments = make(map[string][]int64, len(vchannels))
		for _, channel := range vchannels {
			prepareSegments[channel] = lo.Uniq(releaseSegmentsByChannel[channel])
		}
	}

	log.Ctx(ctx).Info("wait release handoff prepared for TEXT collection",
		zap.Int64("collectionID", coll.GetCollectionID()),
		zap.Any("fenceTs", fenceTs),
		zap.Any("fencedSegments", fencedSegments),
		zap.Any("prepareSegments", prepareSegments))
	waitCtx, cancel := context.WithTimeout(ctx, paramtable.Get().QueryCoordCfg.TextReleaseDrainTimeout.GetAsDuration(time.Millisecond))
	defer cancel()
	if err := d.prepareTextReleaseHandoff(waitCtx, coll.GetCollectionID(), vchannels, fenceTs, prepareSegments); err != nil {
		return nil, err
	}

	log.Ctx(ctx).Info("release handoff prepared for TEXT collection",
		zap.Int64("collectionID", coll.GetCollectionID()),
		zap.Any("fenceTs", fenceTs))
	return fenceTs, nil
}

func normalizeTextReleaseSegments(segmentsByChannel map[string][]int64) map[string][]int64 {
	normalized := make(map[string][]int64, len(segmentsByChannel))
	for channel, segmentIDs := range segmentsByChannel {
		if channel == "" {
			continue
		}
		segmentIDs = lo.Uniq(segmentIDs)
		if len(segmentIDs) == 0 {
			continue
		}
		normalized[channel] = segmentIDs
	}
	return normalized
}

func (d *textReleaseDrainer) prepareTextReleaseHandoff(
	ctx context.Context,
	collectionID int64,
	vchannels []string,
	fenceTs map[string]uint64,
	fencedSegments map[string][]int64,
) error {
	for _, channel := range vchannels {
		if err := d.prepareHandoffOnFlusher(ctx, channel, fencedSegments[channel], fenceTs[channel]); err != nil {
			return errors.Wrapf(err, "prepare TEXT release handoff on WAL flusher for channel %s collection %d", channel, collectionID)
		}
	}
	return nil
}

func (d *textReleaseDrainer) prepareHandoffOnFlusher(ctx context.Context, channel string, segmentIDs []int64, fenceTs uint64) error {
	if d.textFlushProgressGetter == nil {
		return errors.New("text flush progress getter is not initialized")
	}
	_, err := d.textFlushProgressGetter.GetTextFlushProgress(ctx, channel, segmentIDs, fenceTs)
	return err
}

func (d *textReleaseDrainer) ReleaseDrainChannels(ctx context.Context, collectionID int64) []string {
	channelSet := make(map[string]struct{})
	for _, scope := range []meta.TargetScope{meta.CurrentTarget, meta.NextTarget} {
		for channel := range d.targetMgr.GetDmChannelsByCollection(ctx, collectionID, scope) {
			channelSet[channel] = struct{}{}
		}
	}

	channels := make([]string, 0, len(channelSet))
	for channel := range channelSet {
		channels = append(channels, channel)
	}
	return channels
}
