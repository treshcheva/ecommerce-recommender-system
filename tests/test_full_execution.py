import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_full import prepare_full_notebook
from src.resource_guard import ResourceGuard


def test_full_copies_use_300_dpi_and_keep_sources_unchanged():
    root = Path(__file__).resolve().parents[1]
    for name in ['02_modeling', '03_evaluation', '04_learning_to_rank']:
        path = root / 'notebooks' / (name + '.ipynb')
        original = path.read_bytes()
        notebook = prepare_full_notebook(path)
        assert path.read_bytes() == original
        assert notebook.metadata.figure_dpi == 300
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                assert cell.execution_count is None
                assert not cell.outputs
                assert 'dpi=160' not in cell.source
                assert 'dpi=180' not in cell.source


def test_guard_stops_before_launch_when_reserve_is_low(monkeypatch, tmp_path):
    monkeypatch.setattr('src.resource_guard.resource_snapshot', lambda: {
        'ram_available_gib': 0.7, 'disk_free_gib': 13, 'logical_cpus': 8, 'load_1m': 2,
        'swap_used_gib': 1.5})
    with pytest.raises(RuntimeError, match='before launch'):
        with ResourceGuard(tmp_path / 'samples.jsonl'):
            pytest.fail('Must not start work')


def test_guard_records_and_stops_on_process_memory_limit(monkeypatch, tmp_path):
    monkeypatch.setattr('src.resource_guard.resource_snapshot', lambda: {
        'ram_available_gib': 2, 'disk_free_gib': 13, 'logical_cpus': 8, 'load_1m': 2,
        'swap_used_gib': 1.5})
    guard = ResourceGuard(tmp_path / 'samples.jsonl')
    guard.parent = SimpleNamespace(children=lambda recursive: [],
                                   memory_info=lambda: SimpleNamespace(rss=2*2**30))
    guard._watch()
    assert 'RSS exceeded' in guard.reason
    assert guard.peaks['preflight'] == 2
    with pytest.raises(RuntimeError, match='RSS exceeded'):
        guard.check()
