"""Run scheduled Stage 1 folds and collect the Stage 2 input layout.

python scripts/run_stage1.py --run-dir runs/<id> [--workers 3] [--only fs1:2018-02,...]
    [--retain fs1:2018-02,...]

Reads ``<run>/prepared`` (schedule, snapshots, geometry). Each scheduled fold runs in
its own process and a scratch working directory outside the synced tree
(``<tmp>/fourclass_stage1/<run>/fs{N}_{YYYY-MM}``, holding bulky checkpoints); the
retained evidence is copied into ``<run>/stage1/folds/fs{N}_{YYYY-MM}`` and the
Stage 2 handoff into ``<run>/stage1/GeoRFResults``. Completed folds are skipped,
never refitted; a fold directory that exists without a completion record is an error.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCHEMA = PACKAGE.parent / ".trellis" / "tasks" / "09-28-fewsnet-four-class-perturbation" / "feature-schema.json"
HORIZON_OF = {1: 4, 2: 8, 3: 12}


def collect_stage1(work: Path, results_dir: Path, fold_dir: Path, year: int, month: int, scope: int) -> None:
    """Copy one complete monthly run; never replace an existing handoff."""
    term = f'{year}-{month:02d}'
    archive = results_dir / f'result_GeoRF_{year}_fs{scope}_{term}_visual'
    metrics = work / f'results_df_gp_fs{scope}_{year}_{year}.csv'
    predictions = work / f'y_pred_test_gp_fs{scope}_{year}_{year}.csv'
    tables = list(work.glob(f'result_GeoRF*/correspondence_table_{term}.csv'))
    if len(tables) != 1 or not metrics.is_file() or not predictions.is_file():
        raise RuntimeError(f'Incomplete Stage 1 handoff in {work}; inspect run.log')
    destinations = [results_dir / f'{p.stem}_m{month:02d}.csv' for p in (metrics, predictions)]
    if archive.exists() or any(p.exists() for p in destinations):
        raise FileExistsError(f'Stage 1 handoff already exists for {term}, fs{scope}')
    results_dir.mkdir(parents=True, exist_ok=True)
    archive.mkdir()
    shutil.copy2(tables[0], archive / tables[0].name)
    for source, target in zip((metrics, predictions), destinations):
        shutil.copy2(source, target)
    fold_dir.mkdir(parents=True)
    for name in ('candidate.json', 'fold_membership.csv.gz', 'command.json', 'run.log'):
        shutil.copy2(work / name, fold_dir / name)
    shutil.copy2(tables[0], fold_dir / 'correspondence_table.csv')
    shutil.copy2(predictions, fold_dir / 'target_predictions.csv')
    shutil.copy2(metrics, fold_dir / 'heldout_scores.csv')
    model_dir = tables[0].parent
    for name in ('log_print.txt', 'val_coverage_by_group.csv', 'feature_name_reference.csv'):
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, fold_dir / name)
    for name in ('s_branch.pkl', 'branch_table.npy', 'X_branch_id.npy'):
        shutil.copy2(model_dir / 'space_partitions' / name, fold_dir / name)


def run_fold(run: Path, fold: dict, python: str, retain: bool) -> dict:
    scope, term = fold['scope'], fold['target_month']
    name = f'fs{scope}_{term}'
    fold_dir = run / 'stage1' / 'folds' / name
    if (fold_dir / 'candidate.json').exists():
        return {'fold': name, 'status': 'already_completed'}
    work = Path(tempfile.gettempdir()) / 'fourclass_stage1' / run.name / name
    if work.exists():
        shutil.rmtree(work)  # scratch from an interrupted attempt; nothing was collected
    if fold_dir.exists():
        raise FileExistsError(f'{name}: partial output exists without a completion record')
    work.mkdir(parents=True)
    prepared = run / 'prepared'
    command = [python, '-B', str(PACKAGE / 'app' / 'main_model_GF.py'),
               '--data', str(prepared / f'snapshot_h{HORIZON_OF[scope]}.parquet'),
               '--geometry-dir', str(prepared / 'geometry'), '--schema', str(SCHEMA),
               '--forecasting_scope', str(scope), '--desired_terms', term]
    if retain:
        command += ['--retain-checkpoints', str(run / 'stage1' / 'retained' / name)]
    (work / 'command.json').write_text(json.dumps({'command': command, 'cwd': str(work)}, indent=2),
                                       encoding='utf-8')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED='5')
    env.pop('PYTHONPATH', None)
    started = time.time()
    with (work / 'run.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {'fold': name, 'status': 'failed', 'returncode': result.returncode,
                'log': str(work / 'run.log')}
    year, month = map(int, term.split('-'))
    collect_stage1(work, run / 'stage1' / 'GeoRFResults', fold_dir, year, month, scope)
    shutil.rmtree(work, ignore_errors=True)
    return {'fold': name, 'status': 'completed', 'seconds': round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--only', default='', help='comma list of fs{N}:{YYYY-MM}')
    parser.add_argument('--retain', default='', help='comma list of fs{N}:{YYYY-MM} keeping checkpoints')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    schedule = json.loads((run / 'prepared' / 'manifests' / 'schedule.json').read_text(encoding='utf-8'))
    folds = [f for f in schedule['stage1'] if f['status'] == 'scheduled']
    wanted = {s for s in args.only.split(',') if s}
    if wanted:
        folds = [f for f in folds if f"fs{f['scope']}:{f['target_month']}" in wanted]
        if len(folds) != len(wanted):
            raise SystemExit(f'--only names unscheduled folds: {wanted}')
    retain = {s for s in args.retain.split(',') if s}
    ledger_path = run / 'stage1' / 'ledger.jsonl'
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_fold, run, f, python, f"fs{f['scope']}:{f['target_month']}" in retain)
                for f in folds]
        failures = 0
        for job in jobs:
            outcome = job.result()
            print(json.dumps(outcome), flush=True)
            with ledger_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(outcome) + '\n')
            failures += outcome['status'] == 'failed'
    if failures:
        raise SystemExit(f'{failures} Stage 1 folds failed')


if __name__ == '__main__':
    main()
