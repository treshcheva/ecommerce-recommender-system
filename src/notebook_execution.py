"""Prepare execution artifacts without rewriting canonical notebook sources."""

import hashlib
import json
from pathlib import Path

import nbformat


NOTEBOOK_NAMES = ('02_modeling', '03_evaluation', '04_learning_to_rank')


def code_sources(notebook):
    """Exact ordered code-cell sources; outputs and run metadata are excluded."""
    return [cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            for cell in notebook['cells'] if cell['cell_type'] == 'code']


def notebook_code_hash(notebook):
    return hashlib.sha256(json.dumps(code_sources(notebook), ensure_ascii=False).encode()).hexdigest()


def prepare_notebook(source: Path, mode: str, resume: bool = False):
    """Copy source/narrative unchanged and clear only execution state.

    Mode and checkpoint policy are environment parameters; no cell source is
    injected, replaced, regex-transformed, or loaded from an old run artifact.
    """
    if mode not in {'dev', 'full'}:
        raise ValueError('mode must be dev or full')
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop('role', None)
    notebook.metadata['experiment_mode'] = mode
    notebook.metadata['source_notebook'] = 'notebooks/' + source.name
    notebook.metadata['source_code_sha256'] = notebook_code_hash(notebook)
    notebook.metadata['execution_parameters'] = {
        'BENCHMARK_MODE': mode, 'RESUME_EXPERIMENTS': '1' if resume else '0',
    }
    notebook.metadata['figure_dpi'] = 300
    return notebook


def save_executed_notebook(notebook, source: Path, target: Path):
    """Never overwrite the canonical source, even if a runner passes a bad path."""
    if source.resolve() == target.resolve():
        raise ValueError('Executed artifacts must be saved separately from source notebooks.')
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, target)


def validate_completed_source(source: Path, target: Path, checkpoint):
    """Refuse to silently skip changed code or a missing/damaged run artifact."""
    if not target.is_file():
        raise ValueError('Completed notebook artifact is missing: {}'.format(target))
    source_hash = notebook_code_hash(nbformat.read(source, as_version=4))
    artifact = nbformat.read(target, as_version=4)
    if source_hash != checkpoint.get('source_code_sha256') or source_hash != notebook_code_hash(artifact):
        raise ValueError('Canonical code changed for {}; preserved results were not rerun. '
                         'Review the checkpoint before explicitly starting a new experiment.'.format(source.name))
    if hashlib.sha256(target.read_bytes()).hexdigest() != checkpoint['notebook_sha256']:
        raise ValueError('Completed artifact checksum changed: {}'.format(target))
