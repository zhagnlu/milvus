"""Verify archived inputs, gate statistics, links, and generated table stability."""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / name).open() as source:
        return list(csv.DictReader(source))


def main():
    manifest = json.loads((ROOT / 'archive-manifest.json').read_text())
    for entry in manifest:
        data = (ROOT / entry['file']).read_bytes()
        assert len(data) == entry['bytes'], entry['file']
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['file']

    counts = {
        'initial/delete-threshold-all.csv': 1600,
        'layouts/delete-layout-results.csv': 2786,
        'delete-size-grid-4MiB/matrix.csv': 136,
        'delete-size-grid-1MiB/matrix.csv': 32,
        'delete-size-grid-bulk/matrix.csv': 12,
        'sizes/delete-size-gate-bulk.csv': 12,
        'sizes/delete-size-results.csv': 192,
        'delete-size-midpoints-20261027/matrix.csv': 192,
        'delete-size-midpoints-20261027/results.csv': 192,
    }
    for name, count in counts.items():
        assert len(rows(name)) == count, name
    midpoint = rows('delete-size-midpoints-20261027/results.csv')
    assert sum(r['retested'] == '1' for r in midpoint) == 12

    summary = json.loads((ROOT / 'recommended-gate-summary.json').read_text())
    expected = []
    for name in summary['inputs']:
        for row in rows(name):
            if int(row['batch']) >= 1024 and int(row['chunks']) >= (32 if row['mode'] == 'load' else 16):
                row['archive_source'] = name
                expected.append(row)
    selected = rows('recommended-gate-selected.csv')
    assert len(expected) == len(selected) == 639
    for a, b in zip(expected, selected):
        assert all(b[k] == v for k, v in a.items())
        assert all(k in a or v == '' for k, v in b.items())
    for mode in ['stream', 'load']:
        subset = [r for r in selected if (r['mode'] == 'load') == (mode == 'load')]
        actual = {'records': len(subset),
                  'min_median_speedup': min(float(r['speedup']) for r in subset),
                  'min_round_speedup': min(float(r['round_min_speedup']) for r in subset),
                  'regressions': sum(float(r['speedup']) < 1 for r in subset),
                  'batch_min': min(int(r['batch']) for r in subset),
                  'batch_max': max(int(r['batch']) for r in subset)}
        assert actual == summary['paths'][mode]

    report = ROOT.parent / (ROOT.name + '.md')
    before = report.read_bytes()
    subprocess.run([sys.executable, str(ROOT / 'render_report_tables.py')], check=True)
    assert report.read_bytes() == before, 'Tables differed from archived CSV values'
    links = 0
    for path in [report, ROOT / 'README.md']:
        text = path.read_text()
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if link.startswith(('http:', 'https:', '#')):
                continue
            assert (path.parent / link.split('#')[0]).exists(), (path, link)
            links += 1
    print(f'OK: {len(manifest)} archived hashes; 9 dataset counts; 639 gate records; '
          f'12 retests; generated tables; {links} local links.')


if __name__ == '__main__':
    main()
