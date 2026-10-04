"""Reproduce the paper end to end: from the raw ledgers to every table, figure and number.

    python run_all.py                 everything the available data allow (see below), then the check
    python run_all.py --quick         smoke test of every stage in a few minutes (writes to output_quick/)
    python run_all.py --only public   selected stages (comma-separated): test, public, client, paper, check, or the
                                      halves public-benchmark, public-analyses, client-benchmark, client-analyses
    python run_all.py --parallel      run the ledger-D part of the client benchmark next to ledgers A, B and C

Stages
  test    unit tests of the type-aware measures (and of the client harness when the client ledgers are present)
  public  public-ledger benchmark (detectors, supervised reference, DeepSAD review protocol and its no-feedback
          control, about 3 h on 2 cores) and the public-ledger analyses
  client  client-ledger benchmark with the frozen protocol (injection, detectors, review protocol, supervised
          reference; about 8 h for ledgers A, B, C and 12 h for ledger D on 2 cores) and the client-ledger analyses;
          runs only when tableA.csv to tableD.csv are found (JET_CLIENT_DATA, default data/client/)
  paper   every table and figure of the paper and its supplement, the assembled manuscript and its anonymized
          versions (LaTeX compiled when pdflatex is installed); needs the public and client stages
  check   compares what was produced with the values reported in the paper (reference/) and writes
          output/check_report.md

Every step logs to output/logs/ and is skipped when its outputs exist, so an interrupted run resumes where it stopped
(--force reruns everything).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--quick', action='store_true', help='smoke test of every stage (few minutes), output in output_quick/')
ap.add_argument('--only', default='', help='comma-separated stages: test, public, client, paper, check')
ap.add_argument('--parallel', action='store_true', help='run ledger D of the client benchmark in parallel to A, B, C')
ap.add_argument('--force', action='store_true', help='rerun steps whose outputs already exist')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
if args.quick and not os.environ.get('JET_OUTPUT'):
    os.environ['JET_OUTPUT'] = str(ROOT / 'output_quick')
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402

PY = sys.executable
LOGS = paths.OUTPUT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
ONE_THREAD = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
summary = []


def log(msg):
    print(msg, flush=True)


def step(name, cmd, cwd=ROOT, outputs=(), env=None, background=False):
    """Run one step; skip it when all its outputs exist (unless --force)."""
    outs = [Path(o) for o in outputs]
    if outs and not args.force and all(o.exists() for o in outs):
        log(f'  [skip] {name} (outputs exist)'); summary.append((name, 'skipped')); return None
    e = dict(os.environ); e.update(env or {}); e['PYTHONPATH'] = str(ROOT) + os.pathsep + e.get('PYTHONPATH', '')
    lf = LOGS / (name.replace(' ', '_').replace('/', '_') + '.log')
    log(f'  [run ] {name}  (log: {lf.relative_to(paths.OUTPUT.parent) if paths.OUTPUT.parent in lf.parents else lf})')
    fh = open(lf, 'w')
    p = subprocess.Popen(cmd, cwd=str(cwd), env=e, stdout=fh, stderr=subprocess.STDOUT)
    if background:
        return (name, p, fh, time.time())
    return finish((name, p, fh, time.time()))


def finish(job):
    name, p, fh, t0 = job
    rc = p.wait(); fh.close()
    dt = time.time() - t0
    summary.append((name, f'ok ({dt / 60:.1f} min)' if rc == 0 else f'FAILED (exit {rc})'))
    if rc != 0:
        log(f'  [FAIL] {name}: see {LOGS / (name.replace(" ", "_").replace("/", "_") + ".log")}')
        report(); sys.exit(rc)
    return rc


def report():
    log('\nSummary')
    for n, s in summary:
        log(f'  {n:<48} {s}')


def verify_hash(f):
    import hashlib
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    ok = h.hexdigest() == paths.SHA256.get(Path(f).name)
    if not ok:
        log(f'  WARNING: {f} differs from the file used for the paper (SHA-256 {h.hexdigest()[:16]}...); '
            'results will not match the published values.')
    return ok


def py(script, *a):
    return [PY, str(ROOT / script), *map(str, a)]


# ------------------------------------------------------------------------------------------------ stages
def stage_test():
    log('== test')
    step('test measures', py('typeaware/test_measures.py'), cwd=ROOT / 'typeaware')


def stage_public():
    log('== public ledger')
    if not paths.PUBLIC_CSV.exists():
        sys.exit(f'public ledger not found: {paths.PUBLIC_CSV}\nDownload journal_entries_4203.csv from '
                 'https://doi.org/10.5281/zenodo.22014653 into data/ (see data/README.md) or set JET_PUBLIC_CSV.')
    verify_hash(paths.PUBLIC_CSV)
    S, R = paths.SCORES, paths.RESULTS_RUN
    seeds = paths.PUBLIC_SEEDS
    if args.quick:
        step('public detectors deterministic (quick: without kNN and LOF)', py('public_ledger/run_dets.py', 'deterministic'),
             outputs=[S / 'HBOS_s0.npy', S / 'ECOD_s0.npy', S / 'PCA_s0.npy'], env={**ONE_THREAD, 'DIB_SKIP_SLOW': '1'})
        step('public quick check', py('tools/quick_check.py', 'public'))
        return
    step('public detectors deterministic', py('public_ledger/run_dets.py', 'deterministic'),
         outputs=[S / f'{d}_s0.npy' for d in ('HBOS', 'ECOD', 'PCA', 'kNN', 'LOF')], env=ONE_THREAD)
    for s in seeds:
        step(f'public detectors stochastic seed {s}', py('public_ledger/run_dets.py', 'stochastic', s),
             outputs=[S / f'{d}_s{s}.npy' for d in ('IF', 'AE-pyod', 'VAE-pyod', 'AE-emb')], env=ONE_THREAD)
        step(f'public supervised reference seed {s}', py('public_ledger/run_dets.py', 'ceiling', s),
             outputs=[S / f'LGBM-ceiling_s{s}.npy'], env=ONE_THREAD)
    for s in seeds:
        step(f'public DeepSAD review protocol seed {s}', py('public_ledger/feedback.py', 'deepsad', s),
             outputs=[R / f'feedback_deepsad_s{s}.json'], env=ONE_THREAD)
        step(f'public no-feedback control seed {s}', py('public_ledger/feedback_nolabel_control.py', 'deepsad', s),
             outputs=[R / f'feedback_deepsad-nolabel-control_s{s}.json'], env=ONE_THREAD)
    step('public metrics', py('public_ledger/analyze.py'), outputs=[R / 'metrics_mean.csv'], env=ONE_THREAD)


SLOW = {'a5_bootstrap': 'a5_bootstrap_ci.csv', 'a10b_weights_random': 'a10b_client_random_bounds.csv'}


def stage_public_analyses():
    log('== public-ledger analyses')
    if not (paths.RESULTS_RUN / 'metrics_mean.csv').exists():
        log('  no public-ledger results yet (run the public stage first): skipped.')
        summary.append(('public analyses', 'skipped (no public-ledger results)')); return
    for a in ['dib_typeaware', 'a4_budget_profile', 'a5_bootstrap', 'a6_public_tables', 'a10c_public_weights',
              'a11_public_macro', 'deepsad_public', 'deepsad_control']:
        step(f'analysis public/{a}', py(f'analysis/public/{a}.py'), env=ONE_THREAD,
             outputs=[paths.OUT_PUBLIC / SLOW[a]] if a in SLOW else ())


def harness_copy():
    """Working copy of the harness in output/client_harness with the raw ledgers linked into data/raw.
    The harness code itself is never changed, so its code fingerprint stays that of the paper's runs."""
    W = paths.CLIENT_RUN
    for sub in ('src', 'configs', 'tests'):
        if (W / sub).exists():
            shutil.rmtree(W / sub)
        shutil.copytree(paths.HARNESS / sub, W / sub, ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('protocol.md', 'requirements.txt'):
        shutil.copy(paths.HARNESS / f, W / f)
    raw = W / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for f in paths.CLIENT_FILES:
        dst = raw / f.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(f.resolve())
        except OSError:            # e.g. Windows without symlink rights
            shutil.copy(f, dst)
    return W


def stage_client():
    log('== client ledgers')
    if not paths.has_client_data():
        log(f'  client ledgers not found in {paths.CLIENT_DATA} (tableA.csv to tableD.csv): client stage skipped.')
        summary.append(('client stage', 'skipped (no client ledgers)')); return False
    for f in paths.CLIENT_FILES:
        verify_hash(f)
    W = harness_copy()
    M = W / 'results' / 'metrics'
    m = lambda cells: [M / f'{c}.parquet' for c in cells]  # noqa: E731
    try:
        import pytest  # noqa: F401
        step('client harness tests', [PY, '-m', 'pytest', '-q', 'tests'], cwd=W)
    except ImportError:
        summary.append(('client harness tests', 'skipped (pytest not installed; pip install -r requirements.txt)'))
    step('client data audit', [PY, '-m', 'src.audit'], cwd=W, outputs=[paths.CLIENT_AUDIT])
    if args.quick:
        cell = ['--table', 'C', '--c', '0.015', '--d', 'medium', '--seed', '0']
        step('client quick cell C seed 0', [PY, '-m', 'src.run', 'cell', *cell, '--ceiling', '--save-model'], cwd=W,
             outputs=m(['C_c0.015_medium_s0']))
        step('client quick review protocol C seed 0', [PY, '-m', 'src.run_feedback', 'cell', *cell], cwd=W)
        step('client quick duplicates', py('analysis/client/duplicates.py'))
        return False
    abc = [('client core grid A, B, C', ['src.run', 'grid', '--part', 'core_ABC']),
           ('client review protocol A, B, C', ['src.run_feedback', 'grid', '--part', 'core_ABC']),
           ('client sweep A, B, C', ['src.run', 'grid', '--part', 'sweep'])]
    d = [('client core grid D', ['src.run', 'grid', '--part', 'core_D']),
         ('client review protocol D', ['src.run_feedback', 'grid', '--part', 'core_D'])]
    if args.parallel:
        # two processes, as the protocol recommends: ledger D in the background, A, B, C with one thread each
        job = step(d[0][0], [PY, '-m', *d[0][1]], cwd=W, background=True)
        for n, c in abc:
            step(n, [PY, '-m', *c], cwd=W, env=ONE_THREAD)
        if job:
            finish(job)
        step(d[1][0], [PY, '-m', *d[1][1]], cwd=W)
    else:
        for n, c in abc + d:
            step(n, [PY, '-m', *c], cwd=W)
    step('client collect metrics', [PY, '-m', 'src.run', 'collect'], cwd=W)
    step('client collect review protocol', [PY, '-m', 'src.run_feedback', 'collect'], cwd=W)
    step('client result tables', [PY, '-m', 'src.analyze'], cwd=W)
    return True


def stage_client_analyses():
    log('== client-ledger analyses')
    if not (paths.CLIENT_RESULTS.exists() and paths.CLIENT_AUDIT.exists() and paths.has_client_data()):
        log('  no client-ledger results yet (run the client stage first): skipped.')
        summary.append(('client analyses', 'skipped (no client-ledger results)')); return
    for a in ['duplicates', 'datev_typeaware', 'a2_bounds', 'a3_orderings', 'a7_datev_tables', 'a8_friedman_check',
              'a9_fig3_data', 'a10_weights', 'a10b_weights_random', 'a11_client_macro', 'deepsad_client', 'a1_check']:
        step(f'analysis client/{a}', py(f'analysis/client/{a}.py'), env=ONE_THREAD,
             outputs=[paths.OUT_CLIENT / SLOW[a]] if a in SLOW else ())


def latex(folder, doc):
    if not shutil.which('pdflatex'):
        summary.append((f'latex {doc}', 'skipped (pdflatex not installed)')); return
    for i, c in enumerate([['pdflatex', '-interaction=nonstopmode', doc], ['bibtex', doc],
                           ['pdflatex', '-interaction=nonstopmode', doc], ['pdflatex', '-interaction=nonstopmode', doc]]):
        subprocess.run(c, cwd=str(folder), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logf = folder / f'{doc}.log'
    errors = logf.read_text(errors='ignore').count('\n!') if logf.exists() else -1
    ok = (folder / f'{doc}.pdf').exists() and errors == 0
    summary.append((f'latex {doc}', 'ok' if ok else f'FAILED ({errors} errors, see {logf})'))


def stage_paper():
    log('== paper')
    if not paths.CLIENT_RESULTS.exists() or not (paths.OUT_CLIENT / 'a2_found_bounds.csv').exists():
        log('  the paper stage needs the client-ledger results: skipped.')
        summary.append(('paper stage', 'skipped (no client-ledger results)')); return
    step('tables', py('paper/make_tables.py'))
    step('figures', py('paper/make_figures.py'))
    if not (paths.MANUSCRIPT / 'preamble.tex').exists():
        summary.append(('manuscript', 'skipped (manuscript sources not in this copy)')); return
    step('assemble manuscript and supplement', py('paper/assemble.py'))
    latex(paths.BUILD, 'main'); latex(paths.BUILD, 'supplement')
    step('anonymized versions', py('paper/make_variants.py'))
    for d in ('main_anon', 'title_page', 'supplement_anon'):
        latex(paths.ANON, d)


def stage_check():
    log('== check')
    step('check against the paper', py('tools/check_reproduction.py'))


# ------------------------------------------------------------------------------------------------ main
t0 = time.time()
ALIASES = {'public': ['public-benchmark', 'public-analyses'], 'client': ['client-benchmark', 'client-analyses']}
stages = [s.strip() for s in args.only.split(',') if s.strip()] or ['test', 'public', 'client', 'paper', 'check']
stages = [x for s in stages for x in ALIASES.get(s, [s])]
log(f'output folder: {paths.OUTPUT}')
for s in stages:
    if s == 'test':
        stage_test()
    elif s == 'public-benchmark':
        stage_public()
    elif s == 'public-analyses':
        if not args.quick:
            stage_public_analyses()
    elif s == 'client-benchmark':
        stage_client()
    elif s == 'client-analyses':
        if not args.quick:
            stage_client_analyses()
    elif s == 'paper':
        if not args.quick:
            stage_paper()
    elif s == 'check':
        if not args.quick:
            stage_check()
    else:
        sys.exit(f'unknown stage {s}')
report()
log(f'\ndone in {(time.time() - t0) / 60:.1f} min; outputs in {paths.OUTPUT}')
