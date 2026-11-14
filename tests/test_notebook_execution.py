import hashlib
from pathlib import Path
import sys

import nbformat
import pandas as pd
import pytest

from scripts.check_notebook_sources import audit_sources
from src.notebook_execution import (
    code_sources, notebook_code_hash, prepare_notebook,
    save_executed_notebook, validate_completed_source,
)


def test_repository_notebooks_share_code_and_preserve_historical_outputs():
    report = audit_sources(Path(__file__).resolve().parents[1])
    assert len(report['notebooks']) == 7
    assert all(row['code_exactly_equal'] for row in report['notebooks'])
    assert all(row['all_cell_sources_exactly_equal'] for row in report['notebooks'])
    assert all(row['historical_outputs_preserved'] for row in report['notebooks'])


def source_fixture(tmp_path):
    source = tmp_path / 'notebooks' / 'example.ipynb'
    source.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print("canonical")')])
    nbformat.write(notebook, source)
    return source


@pytest.mark.parametrize('mode,resume', [('dev', False), ('full', True)])
def test_preparation_changes_metadata_and_execution_state_only(tmp_path, mode, resume):
    source = source_fixture(tmp_path)
    original = source.read_bytes()
    expected = nbformat.read(source, as_version=4)
    prepared = prepare_notebook(source, mode, resume)
    assert code_sources(prepared) == code_sources(expected)
    assert source.read_bytes() == original
    assert prepared.metadata.execution_parameters == {
        'BENCHMARK_MODE':mode, 'RESUME_EXPERIMENTS':'1' if resume else '0'}


def test_writer_cannot_overwrite_source(tmp_path):
    source = source_fixture(tmp_path)
    original = source.read_bytes()
    with pytest.raises(ValueError, match='separately'):
        save_executed_notebook(prepare_notebook(source, 'dev'), source, source)
    assert source.read_bytes() == original


def test_checkpoint_rejects_changed_canonical_code_without_execution(tmp_path):
    source = source_fixture(tmp_path)
    artifact = tmp_path / 'outputs/full/notebooks/example.ipynb'
    notebook = prepare_notebook(source, 'full', True)
    save_executed_notebook(notebook, source, artifact)
    checkpoint = {'source_code_sha256':notebook_code_hash(notebook),
                  'notebook_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    validate_completed_source(source, artifact, checkpoint)
    notebook.cells[0].source = 'print("changed source")'
    nbformat.write(notebook, source)
    with pytest.raises(ValueError, match='Canonical code changed'):
        validate_completed_source(source, artifact, checkpoint)


def test_dev_runner_executes_sources_and_saves_separate_artifacts(tmp_path, monkeypatch):
    import os
    import scripts.run_experiments as runner
    for key in ['BENCHMARK_MODE', 'RESUME_EXPERIMENTS', 'MPLBACKEND', 'JUPYTER_PATH']:
        monkeypatch.setenv(key, os.environ.get(key, ''))
    monkeypatch.setattr(runner, '__file__', str(tmp_path / 'scripts/run_experiments.py'))
    monkeypatch.setattr(sys, 'argv', ['run_experiments', '--mode', 'dev'])
    monkeypatch.setattr(runner, 'load_processed_events', lambda path: pd.DataFrame())
    monkeypatch.setattr(runner, 'estimate_benchmark_cost', lambda frame: pd.DataFrame({'mode':['dev']}))
    monkeypatch.setattr(runner, 'version', lambda name: 'test')
    originals = {}
    names = ['02_modeling', '03_evaluation', '04_learning_to_rank']
    (tmp_path / 'notebooks').mkdir()
    for name in names:
        path = tmp_path / 'notebooks' / (name + '.ipynb')
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f'print("{name}")')]), path)
        originals[name] = path.read_bytes()
    executed = []
    class FakeClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook
            assert kwargs['resources']['metadata']['path'] == str(tmp_path)
        def execute(self):
            executed.append(self.notebook.cells[0].source)
            self.notebook.cells[0].execution_count = 1
            self.notebook.cells[0].outputs = [nbformat.v4.new_output('stream', text='executed\n')]
    monkeypatch.setattr('nbclient.NotebookClient', FakeClient)
    runner.main()
    assert executed == [f'print("{name}")' for name in names]
    for name in names:
        source = tmp_path / 'notebooks' / (name + '.ipynb')
        assert source.read_bytes() == originals[name]
        artifact = nbformat.read(tmp_path / 'outputs/dev/notebooks' / source.name, as_version=4)
        assert artifact.cells[0].outputs[0].text == 'executed\n'
