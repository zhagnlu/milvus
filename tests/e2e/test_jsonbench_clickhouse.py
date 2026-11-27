# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-accounting regressions using real subprocesses; no database or sockets."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jsonbench_clickhouse as runner


def round_with(before, after):
    return {"engines": {"clickhouse": {"Q2": {"cpu_counters": [
        {"before": before, "after": after}]}}}}


class ClickHouseCpuTest(unittest.TestCase):
    def test_live_server_and_reaped_worker_cpu(self):
        worker = """
import sys, time
print('ready', flush=True)
sys.stdin.readline()
begin = time.process_time()
while time.process_time() - begin < 0.35:
    pass
print('done', flush=True)
sys.stdin.readline()
"""
        watchdog = f"""
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', {worker!r}], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
try:
    print(child.stdout.readline().strip(), flush=True)
    sys.stdin.readline()
    child.stdin.write('go\\n'); child.stdin.flush()
    print(child.stdout.readline().strip(), flush=True)
    sys.stdin.readline()
    child.stdin.close()
    child.wait()
    print('reaped', flush=True)
    sys.stdin.readline()
finally:
    if child.poll() is None:
        child.kill()
        child.wait()
"""
        child = subprocess.Popen([sys.executable, "-c", watchdog], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            before = runner.cpu_state(child.pid)
            child.stdin.write("go\n")
            child.stdin.flush()
            self.assertEqual(child.stdout.readline().strip(), "done")
            after = runner.cpu_state(child.pid)
            self.assertEqual(before["pids"], after["pids"])
            self.assertEqual(len(after["pids"]), 2)
            self.assertGreaterEqual(after["cpu_seconds"] - before["cpu_seconds"], 0.30)
            self.assertLess(after["root_cpu_seconds"] - before["root_cpu_seconds"], 0.10)
            runner.validate_cpu([round_with(before, after)])
            child.stdin.write("reap\n")
            child.stdin.flush()
            self.assertEqual(child.stdout.readline().strip(), "reaped")
            reaped = runner.cpu_state(child.pid)
            self.assertEqual(reaped["pids"], [child.pid])
            self.assertGreaterEqual(reaped["cpu_seconds"], after["cpu_seconds"] - 0.02)
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            child.stdout.close()

    def test_reject_watchdog_only_counters_even_when_nonzero(self):
        before = {"cpu_seconds": 1.0, "root_cpu_seconds": 1.0, "pids": [1, 2]}
        after = {"cpu_seconds": 1.1, "root_cpu_seconds": 1.1, "pids": [1, 2]}
        with self.assertRaisesRegex(RuntimeError, "child server CPU"):
            runner.validate_cpu([round_with(before, after)])

    def test_each_round_must_capture_cpu(self):
        before = {"cpu_seconds": 1.0, "root_cpu_seconds": 0.0, "pids": [1, 2]}
        after = {"cpu_seconds": 1.4, "root_cpu_seconds": 0.0, "pids": [1, 2]}
        with self.assertRaisesRegex(RuntimeError, "no server work"):
            runner.validate_cpu([round_with(before, after), round_with(before, before)])

    def test_single_process_server(self):
        before = {"cpu_seconds": 1.0, "root_cpu_seconds": 1.0, "pids": [1]}
        after = {"cpu_seconds": 1.4, "root_cpu_seconds": 1.4, "pids": [1]}
        runner.validate_cpu([round_with(before, after)])

    def test_preserve_existing_report_before_starting_databases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical-comparison.json"
            path.write_text(json.dumps({"historical": True}))
            args = SimpleNamespace(samples=10, warmups=2, output_prefix="historical",
                                   run_dir=Path(directory))
            with self.assertRaises(FileExistsError):
                runner.main(args)
            self.assertEqual(json.loads(path.read_text()), {"historical": True})

    def test_unstable_process_tree_and_counter_reset_abort_measurement(self):
        before = {"cpu_seconds": 1.0, "root_cpu_seconds": 0.0, "pids": [1, 2]}
        for after, message in [
            ({"cpu_seconds": 1.1, "root_cpu_seconds": 0.0, "pids": [1, 3]}, "process tree changed"),
            ({"cpu_seconds": 0.9, "root_cpu_seconds": 0.0, "pids": [1, 2]}, "counter moved backwards"),
        ]:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                session = mock.Mock()
                session.post.return_value = SimpleNamespace(content=b'{"data": []}', ok=True)
                args = SimpleNamespace(samples=1, warmups=0, run_dir=Path(directory))
                with mock.patch.object(runner, "cpu_state", side_effect=[before, after]):
                    with self.assertRaisesRegex(RuntimeError, message):
                        runner.measure(args, SimpleNamespace(pid=1), session,
                                       {"Q1": "SELECT 1"}, {"Q1": []}, "attempt")
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
