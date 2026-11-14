"""Audit notebook code parity and output preservation without executing cells."""

import json
from pathlib import Path

import nbformat

from src.notebook_execution import code_sources, notebook_code_hash, validate_completed_source


def output_state(path):
    notebook = json.loads(path.read_text())
    return [(cell.get('execution_count'), [output for output in cell.get('outputs', []) if not output.get('metadata', {}).get('generated_full_report')])
            for cell in notebook['cells'] if cell['cell_type'] == 'code']


def audit_sources(root):
    rows = []
    backup = root / 'outputs/full/logs/source_unification_before'
    for source in sorted((root / 'notebooks').glob('*.ipynb')):
        main = nbformat.read(source, as_version=4)
        nbformat.validate(main)
        if any(cell.get('outputs') or cell.get('execution_count') is not None
               for cell in main.cells if cell.cell_type == 'code'):
            raise ValueError('Canonical source contains execution state: {}'.format(source))
        for folder, old_folder in [('outputs/dev/notebooks', 'main'), ('outputs/full/notebooks', 'full')]:
            artifact_path = root / folder / source.name
            if not artifact_path.exists():
                continue
            artifact = nbformat.read(artifact_path, as_version=4)
            nbformat.validate(artifact)
            if [(c.cell_type, c.source) for c in main.cells] != [(c.cell_type, c.source) for c in artifact.cells]:
                raise ValueError('Notebook cell sources differ: {}'.format(artifact_path))
            if code_sources(main) != code_sources(artifact):
                raise ValueError('Notebook code differs: {}'.format(artifact_path))
            old = backup / old_folder / source.name
            historical = artifact.metadata.get('source_synchronization', {}).get('executed_again') is False
            preserved = output_state(old) == output_state(artifact_path) if old.exists() and historical else None
            if old.exists() and historical:
                old_notebook = json.loads(old.read_text())
                notes = [''.join(c['source']) for c in old_notebook['cells'] if c.get('metadata', {}).get('generated_full_report')]
                displayed = [o.get('data', {}).get('text/markdown') for c in artifact.cells for o in c.get('outputs', []) if o.get('metadata', {}).get('generated_full_report')]
                if notes != displayed:
                    raise ValueError('Generated run conclusions changed: {}'.format(artifact_path))
            if preserved is False:
                raise ValueError('Historical outputs changed: {}'.format(artifact_path))
            if folder == 'outputs/full/notebooks':
                manifest = json.loads((root / 'outputs/full/metrics/full_execution_manifest.json').read_text())
                validate_completed_source(source, artifact_path, manifest['completed'][source.stem])
            rows.append({'source':str(source.relative_to(root)),
                         'artifact':str(artifact_path.relative_to(root)),
                         'code_cells':len(code_sources(main)),
                         'source_code_sha256':notebook_code_hash(main),
                         'code_exactly_equal':True,
                         'all_cell_sources_exactly_equal':True,
                         'historical_outputs_preserved':preserved})
    return {'status':'passed', 'models_executed':False, 'notebooks':rows}


def main():
    root = Path(__file__).resolve().parents[1]
    report = audit_sources(root)
    target = root / 'outputs/full/logs/notebook_source_alignment.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print('Source parity and historical outputs verified for {} artifact copies; no cells executed.'.format(len(report['notebooks'])))


if __name__ == '__main__':
    main()
