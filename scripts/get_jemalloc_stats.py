#!/usr/bin/env python3
# Copyright 2025 Zilliz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Get jemalloc memory statistics from Milvus cluster.

Usage:
    python get_jemalloc_stats.py [--host HOST] [--port PORT] [--all]

Examples:
    python get_jemalloc_stats.py
    python get_jemalloc_stats.py --host 192.168.1.100 --port 19530
    python get_jemalloc_stats.py --all  # Print full system info
"""

import argparse
import json
import sys

from pymilvus import connections
from pymilvus.grpc_gen import milvus_pb2 as milvus_types


def format_bytes(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_system_metrics(host, port):
    """Get system metrics from Milvus."""
    connections.connect(host=host, port=port)
    handler = connections._fetch_handler(alias='default')

    req = milvus_types.GetMetricsRequest(
        request=json.dumps({"metric_type": "system_info"})
    )
    response = handler._stub.GetMetrics(req, wait_for_ready=True, timeout=60)

    return json.loads(response.response)


def print_jemalloc_stats(data):
    """Print jemalloc statistics for each node."""
    nodes_info = data.get('nodes_info', [])

    if not nodes_info:
        print("No nodes found in cluster.")
        return

    print("=" * 70)
    print("Jemalloc Memory Statistics")
    print("=" * 70)

    for node in nodes_info:
        infos = node.get('infos', {})
        node_type = infos.get('type', 'unknown')
        node_name = infos.get('name', 'unknown')
        hw = infos.get('hardware_infos', {})

        # Only show nodes with jemalloc stats (typically querynode)
        jemalloc_available = hw.get('jemalloc_available', False)

        print(f"\nNode: {node_name} (type: {node_type})")
        print("-" * 50)

        if jemalloc_available:
            allocated = hw.get('jemalloc_allocated', 0)
            resident = hw.get('jemalloc_resident', 0)
            cached = hw.get('jemalloc_cached', 0)

            print(f"  Jemalloc Allocated: {format_bytes(allocated):>12}  (actual app usage)")
            print(f"  Jemalloc Resident:  {format_bytes(resident):>12}  (physical memory)")
            print(f"  Jemalloc Cached:    {format_bytes(cached):>12}  (unreturned to OS)")
            print(f"  Jemalloc Available: {jemalloc_available}")

            if resident > 0:
                cache_ratio = (cached / resident) * 100
                print(f"  Cache Ratio:        {cache_ratio:>11.1f}%  (cached/resident)")
        else:
            print(f"  Jemalloc stats not available for this node")
            print(f"  Memory Usage: {format_bytes(hw.get('memory_usage', 0))}")
            print(f"  Total Memory: {format_bytes(hw.get('memory', 0))}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Get jemalloc memory statistics from Milvus cluster'
    )
    parser.add_argument('--host', default='127.0.0.1', help='Milvus host (default: 127.0.0.1)')
    parser.add_argument('--port', default='19530', help='Milvus port (default: 19530)')
    parser.add_argument('--all', action='store_true', help='Print full system info JSON')

    args = parser.parse_args()

    try:
        data = get_system_metrics(args.host, args.port)

        if args.all:
            print(json.dumps(data, indent=2))
        else:
            print_jemalloc_stats(data)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        connections.disconnect(alias='default')


if __name__ == '__main__':
    main()
