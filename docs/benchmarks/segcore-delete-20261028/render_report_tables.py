"""Regenerate the report tables from archived CSVs; no benchmarks are run."""

import csv
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parent / (ROOT.name + '.md')


def read(name):
    with (ROOT / name).open() as source:
        return list(csv.DictReader(source))


def matrix(rows, batches, retests=False):
    sections = []
    for kind in ['int64', 'varchar16']:
        for mode in ['stream_same', 'load']:
            subset = [r for r in rows if r['type'] == kind and r['mode'] == mode]
            cells = {(int(r['rows']), int(r['batch'])): r for r in subset}
            assert len(cells) == len(subset), 'duplicate matrix cells'
            label = '同 TS 流式' if mode == 'stream_same' else '回放'
            lines = [f'**{kind}，{label}；加速比 > 1 表示新模式更快。**', '',
                     '| 段行数 | chunk 数 | ' + ' | '.join(f'{b:,} pairs' for b in batches) + ' |',
                     '|---:|---:|' + '---:|' * len(batches)]
            for n in sorted({int(r['rows']) for r in subset}):
                chunks = {r['chunks'] for r in subset if int(r['rows']) == n}
                assert len(chunks) == 1
                vals = []
                for b in batches:
                    r = cells.get((n, b))
                    if r is None:
                        assert b > n
                        vals.append('—')
                    else:
                        star = '*' if retests and r['retested'] == '1' else ''
                        vals.append(f"{float(r['speedup']):.2f}{star}")
                lines.append(f'| {n:,} | {next(iter(chunks))} | ' + ' | '.join(vals) + ' |')
            sections.append('\n'.join(lines))
    return '\n\n'.join(sections)


def replace(text, tag, content):
    pattern = rf'<!-- {tag} -->(?:.*?<!-- END {tag} -->)?'
    replacement = f'<!-- {tag} -->\n\n{content}\n\n<!-- END {tag} -->'
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.S)
    assert count == 1, tag
    return text


def main():
    size = read('sizes/delete-size-results.csv')
    midpoint = read('delete-size-midpoints-20261027/results.csv')
    assert len(size) == 192 and len(midpoint) == 192
    assert sum(r['retested'] == '1' for r in midpoint) == 12
    text = REPORT.read_text()
    grid = [r for r in size if r['cell_mib'] == '4']
    assert len(grid) == 148
    text = replace(text, 'SIZE_TABLES', matrix(grid, [128, 1024, 8192, 65536, 262144, 440000, 1500000]))
    text = replace(text, 'MIDPOINT_TABLES', matrix(midpoint, [1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536], True))
    keys = [(2000000, 12, 65536), (2000000, 12, 440000), (2000000, 12, 1500000),
            (2000000, 16, 1500000), (2000000, 32, 1500000), (500000, 3, 440000)]
    table = ['| 段行数 | chunks | pairs | 二分总耗时 ms | 新模式总耗时 ms | 加速比 |',
             '|---:|---:|---:|---:|---:|---:|']
    for n, c, b in keys:
        matches = [r for r in size if r['type'] == 'int64' and r['mode'] == 'load' and
                   (int(r['rows']), int(r['chunks']), int(r['batch'])) == (n, c, b)]
        assert len(matches) == 1
        r = matches[0]
        table.append(f"| {n:,} | {c} | {b:,} | {float(r['old_total_us']) / 1000:.3f} | "
                     f"{float(r['new_total_us']) / 1000:.3f} | {float(r['speedup']):.4f} |")
    text = replace(text, 'REGRESSION_TABLE', '\n'.join(table))
    REPORT.write_text(text)
    print('Rendered 148 size cells, 192 midpoint cells, and 6 cost examples.')


if __name__ == '__main__':
    main()
