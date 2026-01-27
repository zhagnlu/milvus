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

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <filesystem>

#include <fcntl.h>
#include <unistd.h>

#include "common/EasyAssert.h"
#include "common/FieldMeta.h"

namespace milvus::segcore {

/**
 * TextLobRef is a 16-byte fixed-size reference to TEXT data stored in a
 * temporary LOB file. It is stored in ConcurrentVector<std::string> as a
 * binary-encoded string.
 *
 * Layout:
 *   - offset (8 bytes): position in the LOB file
 *   - size (4 bytes): length of the text data
 *   - flags (4 bytes): reserved for future use (e.g., compression)
 */
struct TextLobRef {
    uint64_t offset;  // Position in LOB file
    uint32_t size;    // Length of text data
    uint32_t flags;   // Reserved (0 = uncompressed)

    static constexpr size_t kEncodedSize = 16;

    // Encode reference to binary string for storage in ConcurrentVector
    std::string
    Encode() const {
        static_assert(sizeof(TextLobRef) == kEncodedSize,
                      "TextLobRef must be 16 bytes");
        return std::string(reinterpret_cast<const char*>(this), kEncodedSize);
    }

    // Decode reference from binary string
    static TextLobRef
    Decode(std::string_view s) {
        AssertInfo(s.size() == kEncodedSize,
                   "Invalid TextLobRef encoding: expected {} bytes, got {}",
                   kEncodedSize,
                   s.size());
        TextLobRef ref;
        std::memcpy(&ref, s.data(), kEncodedSize);
        return ref;
    }

    // Check if a string looks like a TextLobRef (for debugging)
    static bool
    IsValidEncoding(std::string_view s) {
        return s.size() == kEncodedSize;
    }
};

/**
 * TextLobWriter handles writing TEXT data to a temporary LOB file.
 * Thread-safe: uses mutex to protect concurrent writes.
 *
 * File format: simple binary append (raw text bytes concatenated)
 *   [text1_bytes][text2_bytes][text3_bytes]...
 */
class TextLobWriter {
 public:
    explicit TextLobWriter(const std::string& path)
        : path_(path), current_offset_(0) {
        // Create parent directories if needed
        auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        // Open file for writing (truncate if exists)
        file_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
        AssertInfo(file_.is_open(),
                   "Failed to open temp LOB file for writing: {}",
                   path);
    }

    ~TextLobWriter() {
        Close();
    }

    // Non-copyable, non-movable
    TextLobWriter(const TextLobWriter&) = delete;
    TextLobWriter&
    operator=(const TextLobWriter&) = delete;
    TextLobWriter(TextLobWriter&&) = delete;
    TextLobWriter&
    operator=(TextLobWriter&&) = delete;

    /**
     * Write text data to the LOB file and return a reference.
     * Thread-safe.
     */
    TextLobRef
    Write(const std::string& text) {
        return Write(text.data(), text.size());
    }

    TextLobRef
    Write(const char* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        AssertInfo(size <= std::numeric_limits<uint32_t>::max(),
                   "TEXT value too large for LOB spillover: {} bytes (max 4GB)",
                   size);

        TextLobRef ref;
        ref.offset = current_offset_;
        ref.size = static_cast<uint32_t>(size);
        ref.flags = 0;

        file_.write(data, size);
        AssertInfo(file_.good(), "Failed to write to temp LOB file: {}", path_);

        current_offset_ += size;
        return ref;
    }

    /**
     * Flush buffered data to disk without closing the file.
     * Thread-safe.
     */
    void
    Flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.flush();
        }
    }

    void
    Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.close();
        }
    }

    const std::string&
    GetPath() const {
        return path_;
    }

    uint64_t
    GetCurrentOffset() const {
        return current_offset_;
    }

 private:
    std::string path_;
    std::ofstream file_;
    std::mutex mutex_;
    uint64_t current_offset_;
};

/**
 * TextLobReader handles reading TEXT data from a temporary LOB file
 * using pread() for lock-free, thread-safe concurrent reads.
 *
 * pread() reads at a given offset without modifying the file descriptor's
 * position, so multiple threads can read concurrently without any locking.
 * This also allows the writer to keep appending while reads are in progress.
 */
class TextLobReader {
 public:
    explicit TextLobReader(const std::string& path) : path_(path), fd_(-1) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        AssertInfo(
            fd_ >= 0, "Failed to open temp LOB file for reading: {}", path);
    }

    ~TextLobReader() {
        Close();
    }

    // Non-copyable, non-movable
    TextLobReader(const TextLobReader&) = delete;
    TextLobReader&
    operator=(const TextLobReader&) = delete;
    TextLobReader(TextLobReader&&) = delete;
    TextLobReader&
    operator=(TextLobReader&&) = delete;

    /**
     * Read text data from the LOB file using a reference.
     * Thread-safe without locking (pread is atomic w.r.t. offset).
     */
    std::string
    Read(const TextLobRef& ref) {
        return Read(ref.offset, ref.size);
    }

    std::string
    Read(uint64_t offset, uint32_t size) {
        std::string result(size, '\0');
        size_t total_read = 0;
        while (total_read < size) {
            ssize_t n = ::pread(fd_,
                                result.data() + total_read,
                                size - total_read,
                                offset + total_read);
            AssertInfo(n > 0,
                       "Failed to pread from temp LOB file: {} at offset {}, "
                       "size {}, read so far {}",
                       path_,
                       offset + total_read,
                       size - total_read,
                       total_read);
            total_read += n;
        }
        return result;
    }

    void
    Close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    const std::string&
    GetPath() const {
        return path_;
    }

 private:
    std::string path_;
    int fd_;
};

/**
 * TextLobSpillover combines writer and reader for a single TEXT field.
 * Manages the lifecycle of the temporary LOB file.
 *
 * Design: Writer stays open for the entire lifetime of the growing segment.
 * Reader is opened lazily on the first read and uses pread() for lock-free
 * concurrent reads. Before each read, the writer is flushed to ensure all
 * written data is visible. This allows insert and query to interleave freely.
 */
class TextLobSpillover {
 public:
    TextLobSpillover(int64_t segment_id,
                     FieldId field_id,
                     const std::string& base_path)
        : segment_id_(segment_id),
          field_id_(field_id),
          path_(BuildPath(base_path, segment_id, field_id)) {
        writer_ = std::make_unique<TextLobWriter>(path_);
    }

    ~TextLobSpillover() {
        // Close writer and reader, then delete the temp file
        writer_.reset();
        reader_.reset();
        if (std::filesystem::exists(path_)) {
            std::filesystem::remove(path_);
        }
    }

    // Non-copyable, non-movable
    TextLobSpillover(const TextLobSpillover&) = delete;
    TextLobSpillover&
    operator=(const TextLobSpillover&) = delete;
    TextLobSpillover(TextLobSpillover&&) = delete;
    TextLobSpillover&
    operator=(TextLobSpillover&&) = delete;

    /**
     * Write text and return encoded reference string.
     * The returned string can be stored in ConcurrentVector<std::string>.
     * Safe to call concurrently with DecodeAndRead().
     */
    std::string
    WriteAndEncode(const std::string& text) {
        TextLobRef ref = writer_->Write(text);
        return ref.Encode();
    }

    std::string
    WriteAndEncode(const char* data, size_t size) {
        TextLobRef ref = writer_->Write(data, size);
        return ref.Encode();
    }

    /**
     * Read text using encoded reference string.
     * Flushes writer first to ensure all data is on disk,
     * then reads via pread() (lock-free, concurrent-safe).
     * Safe to call concurrently with WriteAndEncode().
     */
    std::string
    DecodeAndRead(std::string_view ref_str) {
        // Flush writer to ensure recently written data is visible on disk
        writer_->Flush();
        EnsureReaderOpen();
        TextLobRef ref = TextLobRef::Decode(ref_str);
        return reader_->Read(ref);
    }

    /**
     * Get reader for use in flush path.
     * Flushes writer first.
     * Caller must not delete the returned pointer.
     */
    TextLobReader*
    GetReader() {
        writer_->Flush();
        EnsureReaderOpen();
        return reader_.get();
    }

    const std::string&
    GetPath() const {
        return path_;
    }

    FieldId
    GetFieldId() const {
        return field_id_;
    }

 private:
    static std::string
    BuildPath(const std::string& base_path,
              int64_t segment_id,
              FieldId field_id) {
        std::filesystem::path p(base_path);
        p /= "growing_lob";
        p /= std::to_string(segment_id);
        p /= std::to_string(field_id.get()) + ".lob";
        return p.string();
    }

    void
    EnsureReaderOpen() {
        // reader uses pread (stateless), so once opened it's always valid.
        // use call_once for lock-free lazy init — safe because reader
        // creation is idempotent and pread-based reader has no mutable state.
        std::call_once(reader_init_flag_, [this]() {
            reader_ = std::make_unique<TextLobReader>(path_);
        });
    }

 private:
    int64_t segment_id_;
    FieldId field_id_;
    std::string path_;

    std::unique_ptr<TextLobWriter> writer_;  // always open, append-only
    std::once_flag reader_init_flag_;
    std::unique_ptr<TextLobReader> reader_;  // lazily opened, pread-based
};

}  // namespace milvus::segcore
