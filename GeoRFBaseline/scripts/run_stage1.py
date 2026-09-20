"""Run one month/scope and collect the legacy Stage 2 input layout."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def collect_stage1(run_dir: Path, results_dir: Path, year: int, month: int, scope: int) -> None:
    """Copy one complete monthly run; never replace an existing handoff."""
    term = f'{year}-{month:02d}'
    archive = results_dir / f'result_GeoRF_{year}_fs{scope}_{term}_visual'
    metrics = run_dir / f'results_df_gp_fs{scope}_{year}_{year}.csv'
    predictions = run_dir / f'y_pred_test_gp_fs{scope}_{year}_{year}.csv'
    tables = list(run_dir.glob(f'result_GeoRF*/correspondence_table_{term}.csv'))
    if len(tables) != 1 or not metrics.is_file() or not predictions.is_file():
        raise RuntimeError(f'Incomplete Stage 1 handoff in {run_dir}; inspect run.log')
    destinations = [results_dir / f'{p.stem}_m{month:02d}.csv' for p in (metrics, predictions)]
    if archive.exists() or any(p.exists() for p in destinations):
        raise FileExistsError(f'Stage 1 handoff already exists for {term}, fs{scope}')
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tables[0].parent, archive)
    for source, target in zip((metrics, predictions), destinations):
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--polygons', type=Path, required=True)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, choices=range(1, 13), required=True)
    parser.add_argument('--scope', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    data, polygons = args.data.resolve(), args.polygons.resolve()
    for path in (data, polygons):
        if not path.is_file():
            parser.error(f'Input does not exist: {path}')
    experiment = args.experiment_dir.resolve()
    term = f'{args.year}-{args.month:02d}'
    run_dir = experiment / 'runs' / f'fs{args.scope}_{term}'
    run_dir.mkdir(parents=True, exist_ok=False)
    entry = Path(__file__).resolve().parents[1] / 'app' / 'main_model_GF.py'
    command = [sys.executable, '-B', str(entry), '--data', str(data),
               '--start_year', str(args.year), '--end_year', str(args.year),
               '--forecasting_scope', str(args.scope), '--desired_terms', term]
    environment = dict(os.environ, GEORF_POLYGONS=str(polygons), PYTHONIOENCODING='utf-8')
    (run_dir / 'command.json').write_text(json.dumps({
        'command': command, 'GEORF_POLYGONS': str(polygons),
        'baseline_version': '0.1.0-f1-nosmote',
    }, indent=2), encoding='utf-8')
    with (run_dir / 'run.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=run_dir, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    collect_stage1(run_dir, experiment / 'GeoRFResults', args.year, args.month, args.scope)
    print(f'Stage 1 handoff ready: {experiment / "GeoRFResults"}')


if __name__ == '__main__':
    main()
