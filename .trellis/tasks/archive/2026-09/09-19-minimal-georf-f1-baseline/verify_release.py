"""Build and independently exercise the local source release using Windows Python."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

task = Path(__file__).resolve().parent
repo = next(parent for parent in task.parents if (parent / 'GeoRFBaseline').is_dir())
package = repo / 'GeoRFBaseline'
evidence = task / 'validation'
evidence.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
provenance = json.loads((package / 'SOURCE_PROVENANCE.json').read_text())
for path, digest in provenance['copied_sources_sha256'].items():
    assert sha(repo / path) == digest, f'Parent source changed: {path}'
files = sorted(p for p in package.rglob('*') if p.is_file()
               and not {'releases', '__pycache__'}.intersection(p.relative_to(package).parts)
               and p.name != 'MANIFEST.json')
for path in files:
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
manifest = {'version': '0.1.0-f1-nosmote', 'source_commit': provenance['source_commit'],
            'files_sha256': {p.relative_to(package).as_posix(): sha(p) for p in files}}
(package / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
release = package / 'releases'
release.mkdir(exist_ok=True)
archive = release / 'georf-baseline-v0.1.0.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in files + [package / 'MANIFEST.json']:
        z.write(path, 'GeoRFBaseline/' + path.relative_to(package).as_posix())
(release / 'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8')
results = []
with tempfile.TemporaryDirectory(prefix='georf-release-check-') as tmp:
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp)
    extracted = Path(tmp) / 'GeoRFBaseline'
    for path, digest in manifest['files_sha256'].items():
        assert sha(extracted / path) == digest, path
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    env.pop('PYTHONPATH', None)
    def run(name, args):
        completed = subprocess.run([sys.executable, '-B', *args], cwd=extracted,
                                   env=env, capture_output=True, text=True, encoding='utf-8')
        (evidence / f'{name}.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        results.append({'check': name, 'returncode': completed.returncode})
        print(name, completed.returncode, flush=True)
        if completed.returncode:
            raise RuntimeError(f'{name} failed; see {evidence}')
    run('focused-tests', ['tests/test_baseline.py'])
    entries = ['app/main_model_GF.py', 'scripts/run_stage1.py',
               'scripts/step1_merge_results.py', 'scripts/step3_create_linked_tables.py',
               'scripts/step4_similarity_matrix.py', 'scripts/step5_sparsification.py',
               'scripts/step6_complete_clustering_pipeline.py',
               'scripts/compare_partitioned_vs_pooled_rf_k40_nc4.py']
    for entry in entries:
        run('help-' + Path(entry).stem, [entry, '--help'])
    # Exercise the actual Stage 2 file pipeline on deliberately synthetic inputs.
    import numpy as np
    import pandas as pd
    experiment = extracted / 'work' / 'synthetic'
    results_dir = experiment / 'GeoRFResults'
    for month in (2, 6):
        model = results_dir / f'result_GeoRF_2018_fs1_2018-{month:02d}_visual'
        model.mkdir(parents=True)
        labels = [f'{(i + (month == 6)) // 20:02d}' for i in range(60)]
        pd.DataFrame({'FEWSNET_admin_code': np.arange(60), 'partition_id': labels}).to_csv(
            model / f'correspondence_table_2018-{month:02d}.csv', index=False)
        pd.DataFrame({'year': [2018], 'month': [month], 'f1(1)': [.8], 'f1_base(1)': [.6]}).to_csv(
            results_dir / f'results_df_gp_fs1_2018_2018_m{month:02d}.csv', index=False)
    pd.DataFrame({'FEWSNET_admin_code': np.arange(60),
                  'lat': np.arange(60) / 10, 'lon': np.arange(60) / 20}).to_csv(
        experiment / 'FEWSNET_admin_code_lat_lon.csv', index=False)
    for stage in ('step1_merge_results', 'step3_create_linked_tables',
                  'step4_similarity_matrix', 'step5_sparsification',
                  'step6_complete_clustering_pipeline'):
        args = [f'scripts/{stage}.py', '--experiment-dir', str(experiment)]
        if stage == 'step1_merge_results':
            args += ['--model-type', 'georf']
        run('synthetic-' + stage, args)
    maps = list((experiment / 'knn_sparsification_results').glob('cluster_mapping_k40_nc*_general.csv'))
    assert len(maps) == 1
    mapping = pd.read_csv(maps[0])
    assert set(mapping['FEWSNET_admin_code']) == set(range(60))
    assert mapping.notna().all().all()
    results.append({'check': 'synthetic-stage2-mapping', 'rows': len(mapping), 'file': maps[0].name})
    run('isolated-import-paths', ['-c',
        'from pathlib import Path; import src.model.GeoRF as m; import config; '
        'root=Path.cwd(); assert Path(m.__file__).is_relative_to(root); '
        'assert Path(config.__file__).is_relative_to(root); print(m.__file__); print(config.__file__)'])
report = {'status': 'passed', 'archive_sha256': sha(archive),
          'archive_bytes': archive.stat().st_size, 'payload_files': len(manifest['files_sha256']),
          'parent_source_hashes_verified': len(provenance['copied_sources_sha256']),
          'checks': results, 'scientific_rerun': False}
(evidence / 'release-verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
