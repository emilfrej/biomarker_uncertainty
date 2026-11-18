"""Fit a parameter recovery cohort from fit_routine.py (synth_patients.csv, true_params.csv) with Gallagher et al.'s
full-history routine: cell 9 of mathematical_biomarkers/paper_figures/Fig2_Probing_Cycle.ipynb (settings, run as it is)
and cell 10's lmfit call (least squares on PSA / first PSA, one fit per patient from lmfit's default starts).

lmfit is not in the at-modelling venv, so run it in uv's throwaway environment, from biomarker_uncertainty/python:
  uv run --no-project --python 3.12 --with lmfit --with seaborn --with tqdm --with pandas --with scipy --with matplotlib \
      python gallagher_full_history_fit.py --cohort-dir ../param_recov_u_ad95

Writes <cohort-dir>/gallagher_full_history/fits.csv (estimates, truth, and the fitted values on our scale) and run_info.json.
Their model has n0 and K free with data scaled by the first PSA; ours fixes K = 1, so their n0 / K is our n0 and
(1 - turnover / (1 - cost)) * K / n0 is our u."""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / 'mathematical_biomarkers'
sys.path.insert(0, str(REPO / 'utils'))
from lmfit import minimize, Parameters  # noqa: E402
from odeModels import LotkaVolterraModel  # noqa: E402
from fittingUtils import residual  # noqa: E402

NAMES = ['n0', 'fR', 'cost', 'turnover', 'K', 'rS']
CELL9 = ''.join(json.load(open(REPO / 'paper_figures' / 'Fig2_Probing_Cycle.ipynb'))['cells'][9]['source'])


def fit_patient(training_data_df):
    """Their settings (cell 9, run as it is) and fit call (cell 10) for one patient's PSA record"""
    t0 = time.time()
    pid = int(training_data_df.PatientId.iloc[0])
    try:
        ns = {'Parameters': Parameters, 'os': os, 'out_dir': '', 'patientIds': []}
        exec(CELL9, ns)
        params, solver_kws, optimiser_kws = ns['params'], ns['solver_kws'], ns['optimiser_kws']
        model = LotkaVolterraModel()
        model.SetParams(**params)
        fit = minimize(residual, params, args=(training_data_df, model, 'PSA', solver_kws), **optimiser_kws)
        est = {k: fit.params[k].value for k in NAMES}
        at_bound = {f'{k}_at_bound': bool(np.isclose(est[k], [fit.params[k].min, fit.params[k].max], rtol=1e-4, atol=1e-8).any())
                    for k in NAMES}
        out = {'success': bool(fit.success), 'nfev': int(fit.nfev), 'message': fit.message, **est, **at_bound}
    except Exception as e:  # keep going, record the failure
        out = {'success': False, 'message': f'error: {e}'}
    return {'PatientId': pid, **out, 'seconds': round(time.time() - t0, 2)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit a fit_routine.py cohort with Gallagher et al.'s full-history routine")
    parser.add_argument('--cohort-dir', type=Path, required=True, help='folder with synth_patients.csv and true_params.csv')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-patients', type=int, default=None, help='only fit the first N patients (for testing)')
    args = parser.parse_args()

    cohort = args.cohort_dir.resolve()
    out_dir = cohort / 'gallagher_full_history'
    out_dir.mkdir(exist_ok=True)
    synth = pd.read_csv(cohort / 'synth_patients.csv')
    truth = pd.read_csv(cohort / 'true_params.csv')
    if args.max_patients is not None:
        synth = synth[synth.PatientId < args.max_patients]
    #their data format: one row per PSA reading, DrugConcentration = dose from that time on (same convention as ours)
    obs = synth.dropna(subset=['Observation']).rename(columns={'Observation': 'PSA'})[['PatientId', 'Time', 'DrugConcentration', 'PSA']]
    records = [g.reset_index(drop=True) for _, g in obs.groupby('PatientId')]

    run_info = {'started': datetime.now().astimezone().isoformat(timespec='seconds'), 'cohort_dir': str(cohort),
                'n_patients': len(records), 'workers': args.workers, 'routine': 'Fig2_Probing_Cycle.ipynb cells 9-10',
                'cell9_settings': CELL9}
    t0 = time.time()
    with get_context('fork').Pool(args.workers) as pool:
        est = pd.DataFrame(pool.map(fit_patient, records))
    run_info.update(finished=datetime.now().astimezone().isoformat(timespec='seconds'), total_seconds=round(time.time() - t0, 2),
                    n_success=int(est.success.sum()))

    #their estimates on our scale (K = 1, sizes relative to K), next to the truth
    est['n0_ours'] = est.n0 / est.K
    est['u_ours'] = (1 - est.turnover / (1 - est.cost)) * est.K / est.n0
    est['rFrac_ours'] = est.fR
    t = truth.set_index('PatientId')
    for k in ['cost', 'turnover', 'rS', 'n0', 'u', 'rFrac']:
        est[f'{k}_true'] = est.PatientId.map(t[k])
    est.to_csv(out_dir / 'fits.csv', index=False)
    (out_dir / 'run_info.json').write_text(json.dumps(run_info, indent=2))
    print(f"{len(est)} fits in {run_info['total_seconds'] / 60:.1f} min, {run_info['n_success']} converged -> {out_dir / 'fits.csv'}")
