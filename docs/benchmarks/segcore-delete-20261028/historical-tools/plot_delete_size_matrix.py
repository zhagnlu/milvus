import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

base = Path('/tmp')
records = []
for name in ['delete-size-grid-4MiB', 'delete-size-grid-1MiB', 'delete-size-grid-bulk']:
    path = base / name / 'matrix.csv'
    if path.exists():
        with path.open() as stream:
            for row in csv.DictReader(stream):
                row['run'] = name
                records.append(row)

control = base / 'delete-size-gate-bulk.csv'
if control.exists():
    with control.open() as stream:
        for row in csv.DictReader(stream):
            row.update(cell_mib='', group_row_bytes=24,
                       batch_fraction=int(row['batch']) / int(row['rows']),
                       run='gate-control')
            records.append(row)

with (base / 'delete-size-results.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

row_counts = [100000, 500000, 1000000, 2000000, 8000000, 44000000]
batches = [128, 1024, 8192, 65536, 262144, 440000, 1500000]
row_labels = ['100k', '500k', '1M', '2M', '8M', '44M']
batch_labels = ['128', '1,024', '8,192', '65,536', '262,144', '440,000', '1,500,000']
lookup = {(r['type'], r['mode'], int(r['rows']), int(r['batch'])): r
          for r in records if r['cell_mib'] == '4'}
logs = [np.log2(float(r['speedup'])) for r in lookup.values()]
norm = colors.DivergingNorm(vmin=min(-1.0, min(logs)), vcenter=0,
                           vmax=max(1.0, max(logs)))
cmap = plt.get_cmap('RdBu').copy() if hasattr(plt.get_cmap('RdBu'), 'copy') else plt.get_cmap('RdBu')
cmap.set_bad('#eeeeee')
fig, axes = plt.subplots(2, 2, figsize=(17, 10))
for ix, mode in enumerate(['stream_same', 'load']):
    for iy, kind in enumerate(['int64', 'varchar16']):
        ax = axes[ix, iy]
        values = np.full((len(row_counts), len(batches)), np.nan)
        for i, rows in enumerate(row_counts):
            for j, batch in enumerate(batches):
                r = lookup.get((kind, mode, rows, batch))
                if r:
                    values[i, j] = np.log2(float(r['speedup']))
        im = ax.imshow(values, norm=norm, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels(batch_labels, fontsize=9)
        ax.set_yticks(range(len(row_counts)))
        row_bytes = 24 if kind == 'int64' else 36
        chunks = [(n * row_bytes + 4194303) // 4194304 for n in row_counts]
        ax.set_yticklabels(['{} / C={}'.format(n, c) for n, c in zip(row_labels, chunks)])
        ax.set_xlabel('Delete pairs (unique PKs)', fontsize=10)
        ax.set_ylabel('Segment rows / modeled chunk count', fontsize=10)
        ax.set_title('{} | {}'.format('INT64' if kind == 'int64' else 'VARCHAR(16)',
                                     'stream, one delete timestamp' if mode == 'stream_same' else 'load replay'),
                     fontsize=13)
        for i in range(len(row_counts)):
            for j in range(len(batches)):
                value = values[i, j]
                if np.isnan(value):
                    label, color = '--', '#888888'
                else:
                    label = '{:.2f}x'.format(2 ** value)
                    color = 'white' if norm(value) < .15 or norm(value) > .85 else '#111111'
                ax.text(j, i, label, ha='center', va='center', color=color, fontsize=10)

fig.suptitle('Delete preparation + lookup: old/new runtime', fontsize=19, y=.98)
fig.text(.5, .935, '4 MiB modeled system-column groups | random unique PKs | i7-8700 | both paths forced',
         ha='center', fontsize=12)
fig.subplots_adjust(left=.09, right=.89, top=.88, bottom=.11, hspace=.32, wspace=.32)
cax = fig.add_axes([.92, .15, .018, .68])
bar = fig.colorbar(im, cax=cax)
ticks = [v for v in [.5, 1, 2, 4, 8, 16, 32, 64] if norm.vmin <= np.log2(v) <= norm.vmax]
bar.set_ticks(np.log2(ticks))
bar.set_ticklabels(['{}x'.format(v) for v in ticks])
bar.set_label('Speedup: >1 new faster; <1 new slower', fontsize=11)
fig.text(.5, .025, 'Median paired ratios. Main grid: 5 rounds; 1.5M-pair extension: 3 rounds. '
         'Excludes storage pins, cgo, skiplist, snapshots and tSafe. -- means unmeasured / batch > rows.',
         ha='center', fontsize=10)
fig.savefig(str(base / 'delete-size-heatmap.png'), dpi=160)
fig.savefig(str(base / 'delete-size-heatmap.svg'))
plt.close(fig)
print('records', len(records), '4MiB cells', len(lookup))

for kind in ['int64', 'varchar16']:
    for mode in ['stream_same', 'load']:
        print(kind, mode)
        for rows in row_counts:
            print(rows, ' '.join(lookup.get((kind, mode, rows, b), {}).get('speedup', '--') for b in batches))
