template <typename Compare>
void
ReorderDeleteBatch(std::vector<PkType>& pks,
                   std::vector<Timestamp>& timestamps,
                   Compare compare) {
    std::vector<size_t> order(pks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), compare);
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] == i) {
            continue;
        }
        auto pk = std::move(pks[i]);
        auto ts = timestamps[i];
        size_t dst = i;
        while (order[dst] != i) {
            const auto src = order[dst];
            pks[dst] = std::move(pks[src]);
            timestamps[dst] = timestamps[src];
            order[dst] = dst;
            dst = src;
        }
        pks[dst] = std::move(pk);
        timestamps[dst] = ts;
        order[dst] = dst;
    }
}


template <typename Chunks, typename GetTimestamp, typename ReadTimestamp,
          typename Callback>
void SortedLookup(const std::vector<PkType>& pks, Chunks chunks,
                  GetTimestamp get_timestamp, ReadTimestamp read_ts,
                  Callback callback) {
    const auto timestamp_hit = [](Timestamp a, Timestamp b) { return a < b; };
    size_t first_pk = 0;
    auto search_sorted_chunk = [&](size_t rows,
                                   int64_t base,
                                   auto value_at,
                                   auto key_at) {
        if (rows == 0) {
            return;
        }
        // Segment PKs are globally sorted. Keep groups that can also occur in
        // the next chunk: advancing to the end of this chunk's matches would
        // lose duplicate PKs straddling a chunk boundary.
        first_pk =
            GallopingLowerBound(pks.size(), first_pk, value_at(0), key_at);
        size_t pos = 0;
        for (size_t j = first_pk;
             j < pks.size() && !(value_at(rows - 1) < key_at(j));) {
            const auto target = key_at(j);
            size_t next = j + 1;
            while (next < pks.size() && key_at(next) == target) {
                ++next;
            }
            pos = GallopingLowerBound(rows, pos, target, value_at);
            size_t end = pos;
            while (end < rows && value_at(end) == target) {
                ++end;
            }
            // Search each distinct PK once, but retain every delete event and
            // its order. Different insert versions can need different deletes.
            for (size_t k = j; k < next; ++k) {
                const auto timestamp = get_timestamp(k);
                for (size_t offset = pos; offset < end; ++offset) {
                    const auto segment_offset = base + offset;
                    if (timestamp_hit(read_ts(segment_offset), timestamp)) {
                        callback(SegOffset(segment_offset), timestamp);
                    }
                }
            }
            pos = end;
            j = next;
        }
    };

    chunks(search_sorted_chunk);
}
