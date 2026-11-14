"""Run approved memory-optimized full notebooks sequentially, with checkpoints."""

import argparse
import contextlib
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import nbformat
from nbclient import NotebookClient

from src.notebook_execution import (
    prepare_notebook, notebook_code_hash, save_executed_notebook, validate_completed_source,
)
from src.benchmark import estimate_benchmark_cost
from src.resource_guard import ResourceGuard
from src.retailrocket import load_processed_events


def prepare_full_notebook(source):
    """Compatibility wrapper: execute canonical source without cell rewrites."""
    return prepare_notebook(source, mode='full', resume=True)


def dev_hashes(root):
    paths = [*root.joinpath('notebooks').glob('*.ipynb')]
    for directory in ['metrics', 'figures', 'models']:
        paths += list(root.joinpath('outputs', directory).glob('*'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = root / 'outputs/full'
    for folder in ['metrics', 'logs', 'notebooks', 'figures']:
        (output / folder).mkdir(exist_ok=True, parents=True)
    status_path = output / 'metrics/full_execution_manifest.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        'mode': 'full', 'completed': {}, 'attempts': [], 'packages': {
            name: version(name) for name in ['numpy','pandas','scipy','implicit','catboost','matplotlib','nbclient','psutil']}}
    before = dev_hashes(root)
    os.environ.update(BENCHMARK_MODE='full', RESUME_EXPERIMENTS='1', MPLBACKEND='module://matplotlib_inline.backend_inline',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    continued = os.environ.get('FULL_ALLOW_SWAP_GROWTH') == '1'
    guard = ResourceGuard(output / 'logs/resource_samples.jsonl',
                          max_rss_gib=1.75 if continued else 1.25,
                          min_available_gib=0.25 if continued else 0.5,
                          max_swap_growth_gib=None if continued else 0.25)
    attempt = {'initial_resources': guard.initial, 'status': 'starting',
               'user_authorized_swap_limit_override': continued}
    status['attempts'].append(attempt)
    def save():
        status_path.write_text(json.dumps(status, indent=2))
    save()
    try:
        with guard:
            train = load_processed_events(root / 'data/processed/train.csv')
            estimate = estimate_benchmark_cost(train)
            estimate.to_csv(output / 'metrics/benchmark_cost_estimate.csv', index=False)
            print(estimate.to_string(index=False), flush=True)
            del train, estimate
            gc.collect()
            guard.check()
            with tempfile.TemporaryDirectory(prefix='retailrocket-full-') as directory:
                kernel = Path(directory) / 'kernels/full'
                kernel.mkdir(parents=True)
                (kernel / 'kernel.json').write_text(json.dumps({
                    'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                    'display_name':'Full experiment','language':'python'}))
                os.environ['JUPYTER_PATH'] = directory + (os.pathsep + os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else '')
                for name in ['02_modeling', '03_evaluation', '04_learning_to_rank']:
                    source = root / 'notebooks' / (name + '.ipynb')
                    target = output / 'notebooks' / (name + '.ipynb')
                    if name in status['completed']:
                        validate_completed_source(source, target, status['completed'][name])
                        print('Skip successfully completed', name, flush=True)
                        continue
                    guard.set_stage(name)
                    notebook = prepare_full_notebook(source)
                    started = time.monotonic()
                    print('Starting', name, flush=True)
                    def progress(cell, cell_index, **kwargs):
                        guard.check()
                        print('{} cell {}: {}'.format(name, cell_index, cell.source[:90].replace('\n',' ')), flush=True)
                    client = NotebookClient(notebook, timeout=1200, kernel_name='full',
                        resources={'metadata':{'path':str(root)}}, on_cell_start=progress)
                    try:
                        client.execute()
                        guard.check()
                    except Exception:
                        save_executed_notebook(notebook, source, target.with_suffix('.partial.ipynb'))
                        raise
                    save_executed_notebook(notebook, source, target)
                    status['completed'][name] = {'seconds':time.monotonic()-started,
                        'sampled_process_tree_peak_rss_gib':guard.peaks.get(name),
                        'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'source_code_sha256':notebook_code_hash(notebook)}
                    save()
                    print('Completed', name, status['completed'][name], flush=True)
                    del client, notebook
                    gc.collect()
            attempt['status'] = 'completed'
    except Exception as error:
        attempt['status'] = 'stopped'
        attempt['error'] = guard.reason or str(error)
        (output / 'logs/last_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        attempt['resources'] = guard.summary()
        status['dev_artifacts_unchanged'] = before == dev_hashes(root)
        save()


def run_logged():
    root = Path(__file__).resolve().parents[1]
    logs = root / 'outputs/full/logs'
    logs.mkdir(parents=True, exist_ok=True)
    # Console retains a concise progress stream; all stdout/stderr is also saved.
    class Tee:
        def __init__(self, stream, file): self.stream, self.file = stream, file
        def write(self, data): self.stream.write(data); self.file.write(data); return len(data)
        def flush(self): self.stream.flush(); self.file.flush()
    with (logs / 'execution.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-swap-growth', action='store_true',
                        help='User-authorized override: monitor swap without stopping on its growth.')
    args = parser.parse_args()
    if args.allow_swap_growth:
        os.environ['FULL_ALLOW_SWAP_GROWTH'] = '1'
    run_logged()
