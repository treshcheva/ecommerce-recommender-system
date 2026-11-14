"""Report completed full results without fitting or selecting any model."""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from src.recommendation import select_benchmark_users, select_catalog_items


METRICS = ['ndcg@10', 'recall@10', 'precision@10', 'coverage@10']


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/full'
    metrics, figures = output / 'metrics', output / 'figures'
    status = json.loads((metrics / 'full_execution_manifest.json').read_text())
    names = ['02_modeling', '03_evaluation', '04_learning_to_rank']
    if not all(name in status['completed'] for name in names):
        raise RuntimeError('All full notebooks must finish before final reporting.')
    if not status['dev_artifacts_unchanged']:
        raise RuntimeError('Dev artifacts unexpectedly changed.')
    comparisons = []
    for split in ['validation', 'test']:
        dev = pd.read_csv(root / f'outputs/metrics/ltr_{split}_metrics.csv').set_index('model')
        full = pd.read_csv(metrics / f'ltr_{split}_metrics.csv').set_index('model')
        for mode, frame in [('dev', dev), ('full', full)]:
            comparisons.append(frame[METRICS + ['evaluated_users']].assign(mode=mode, split=split).reset_index())
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(metrics / 'dev_full_comparison.csv', index=False)
    for split in ['validation', 'test']:
        frame = comparison[comparison.split == split]
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        for metric, ax in zip(METRICS, axes.flat):
            frame.pivot(index='model', columns='mode', values=metric).plot.bar(ax=ax)
            ax.set(title=metric, xlabel='', ylabel='Macro score' if metric != 'coverage@10' else 'Catalog fraction')
            ax.tick_params(axis='x', rotation=35, labelsize=8)
        fig.suptitle(f'Dev vs full: {split}; different cohorts/catalogs')
        fig.savefig(figures / f'dev_full_{split}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    intervals = pd.read_csv(metrics / 'ltr_test_paired_intervals.csv')
    cb_rows = []
    for model in ['implicit ALS', 'ItemKNN']:
        row = intervals.set_index('model').loc[model]
        cb_rows.append({'contrast':'CatBoost minus ' + model, 'users':int(row['users']),
                        'delta':-row['delta'], 'ci_low':-row['ci_high'], 'ci_high':-row['ci_low'],
                        'confidence':row['confidence'], 'n_bootstrap':int(row['n_bootstrap']),
                        'seed':int(row['seed']), 'metric':row['metric']})
    contrasts = pd.DataFrame(cb_rows)
    contrasts.to_csv(metrics / 'catboost_test_contrasts.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.errorbar(contrasts.delta, contrasts.contrast,
                xerr=[contrasts.delta-contrasts.ci_low, contrasts.ci_high-contrasts.delta], fmt='o')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set(title='Full test: paired user bootstrap', xlabel='CatBoost minus generator NDCG@10; 95% CI')
    fig.savefig(figures / 'catboost_test_contrasts.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Count actual selected events without reloading all event columns or fitting.
    train = pd.read_csv(root / 'data/processed/train.csv', usecols=['visitorid','itemid'])
    cohort = select_benchmark_users(train, 5, 100000, 42)
    catalog = select_catalog_items(train, 10000)
    manifest = json.loads((metrics / 'benchmark_manifest.json').read_text())
    assert hashlib.sha256(json.dumps(cohort).encode()).hexdigest() == manifest['cohort_sha256']
    assert hashlib.sha256(json.dumps(catalog).encode()).hexdigest() == manifest['catalog_sha256']
    counts = {'cohort_users':len(cohort), 'catalog_items':len(catalog)}
    counts['train_events'] = int((train.visitorid.isin(cohort) & train.itemid.isin(catalog)).sum())
    del train
    for split in ['validation','test']:
        count = 0
        for frame in pd.read_csv(root / f'data/processed/{split}.csv', usecols=['visitorid','itemid'], chunksize=100000):
            count += int((frame.visitorid.isin(cohort) & frame.itemid.isin(catalog)).sum())
        counts[split + '_events'] = count
        counts[split + '_evaluated_users'] = int(comparison[(comparison['mode']=='full') & (comparison['split']==split)].evaluated_users.iloc[0])
    (metrics / 'dataset_counts.json').write_text(json.dumps(counts, indent=2))

    # Preserve every inline graph as a separate PNG, including plots without
    # an explicit savefig call. Figures were rendered at 300 dpi by the runner.
    pngs = []
    for name in names:
        notebook = nbformat.read(output / 'notebooks' / (name + '.ipynb'), as_version=4)
        for index, cell in enumerate(notebook.cells):
            for position, entry in enumerate(cell.get('outputs', [])):
                data = entry.get('data', {}).get('image/png')
                if data:
                    image = Image.open(BytesIO(base64.b64decode(data)))
                    target = figures / f'{name}_cell_{index}_{position}.png'
                    image.save(target, dpi=(300,300))
                    image.close()
                    pngs.append(target.name)
    for path in figures.glob('*.png'):
        with Image.open(path) as image:
            dpi = image.info.get('dpi', (0,0))
            if not all(abs(value-300) < 0.1 for value in dpi):
                raise AssertionError(f'{path.name}: expected dpi=300, found {dpi}')

    peaks = [attempt['resources']['sampled_rss_peak_gib_by_stage'] for attempt in status['attempts'] if 'resources' in attempt]
    peak = max(value for stage in peaks for value in stage.values())
    segments, current = [], []
    for line in (output / 'logs/resource_samples.jsonl').read_text().splitlines():
        sample = json.loads(line)
        if current and sample['elapsed_seconds'] < current[-1]['elapsed_seconds']:
            segments.append(current)
            current = []
        current.append(sample)
    if current:
        segments.append(current)
    sampled_active_seconds = sum(segment[-1]['elapsed_seconds'] for segment in segments)
    interrupted_04_seconds = max((sample['stage_seconds'] for sample in segments[0]
                                 if sample['stage']=='04_learning_to_rank'), default=0)
    seconds = {name:status['completed'][name]['seconds'] for name in names}
    report = {'counts':counts, 'notebook_seconds':seconds,
              'total_successful_notebook_seconds':sum(seconds.values()),
              'sampled_runner_and_kernel_peak_rss_gib':peak,
              'sampled_active_runner_seconds_including_attempts':sampled_active_seconds,
              'interrupted_04_seconds_at_stop':interrupted_04_seconds,
              'interruption':'First 04 attempt stopped at swap growth >0.25 GiB. User explicitly authorized continuation without that swap threshold; only selected CatBoost was reconstructed, cached search/ablations reused.',
              'rss_definition':'250 ms sampled process-tree RSS sum; shared pages may be double counted; not exact OS high-water mark.',
              'catboost_test_contrasts':cb_rows, 'all_figures_dpi':300,
              'dev_artifacts_unchanged':status['dev_artifacts_unchanged'],
              'limitations':['Dev/full use different cohorts and catalogs: raw score differences are not a controlled model improvement.',
                             'Binary implicit events include views, not only purchases; only known eligible users/new catalog targets are evaluated.',
                             'Single temporal split and training seed; percentile user bootstrap is conditional on fixed models and ignores shared-item dependencies.',
                             'Validation search/early stopping/ablations share validation and produce optimistic diagnostic intervals; intervals are not multiplicity adjusted.']}
    (metrics / 'full_summary.json').write_text(json.dumps(report, indent=2))
    lines = ['# Completed full experiment', '',
             f"Cohort: {counts['cohort_users']:,} users; catalog: {counts['catalog_items']:,} items.",
             f"Events: train {counts['train_events']:,}, validation {counts['validation_events']:,}, test {counts['test_events']:,}.",
             f"Evaluated users: validation {counts['validation_evaluated_users']:,}, test {counts['test_evaluated_users']:,}.", '',
             '## Test metrics', '', '| Model | NDCG@10 | Recall@10 | Precision@10 | Coverage@10 |', '|---|---:|---:|---:|---:|']
    full_test = comparison[(comparison['mode']=='full') & (comparison['split']=='test')]
    for _, row in full_test.iterrows():
        lines.append('| {} | {} |'.format(row['model'], ' | '.join(f'{row[m]:.5f}' for m in METRICS)))
    lines += ['', '## Paired bootstrap, full test', '']
    for row in cb_rows:
        verdict = 'includes zero; improvement not established' if row['ci_low'] <= 0 <= row['ci_high'] else ('positive interval; supports higher NDCG on this cohort' if row['ci_low'] > 0 else 'negative interval; supports lower NDCG on this cohort')
        lines.append(f"- {row['contrast']}: delta {row['delta']:+.5f}, 95% CI [{row['ci_low']:+.5f}, {row['ci_high']:+.5f}]; {verdict}.")
    lines += ['', '## Resources', '',
              *[f'- {name}: {value:.1f} s.' for name,value in seconds.items()],
              f'- Successful notebook total: {sum(seconds.values()):.1f} s.',
              f'- Interrupted first 04 attempt: {interrupted_04_seconds:.1f} s to stop.',
              f'- Active runner time including both attempts/preflight, sampled: {sampled_active_seconds:.1f} s.',
              '- First attempt stopped on swap growth; user authorized continuation without that threshold. Cached search/ablations reused; selected model reconstruction matched saved validation scores.',
              f'- Sampled peak runner + kernel RSS: {peak:.3f} GiB (sum, shared pages may be double counted).',
              '- Resource monitoring interval: 250 ms; sampling may miss shorter peaks.',
              '- Models/grids and full benchmark unchanged. All PNG figures exported at 300 dpi.',
              '- Dev notebooks, metrics, figures and models verified unchanged by SHA-256.',
              '', '## Limits', '', *['- '+x for x in report['limitations']]]
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
