import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd
from scipy import stats

from utils.stan_data import to_stan_prior, make_stan_data, u_init, raw_init, rates_init

ROOT = Path(__file__).resolve().parents[1]
SAMPLER_SEED = 4  #seed for pathfinder and sampling in fit_patient
#Stan file and pathfinder init per --model; all take make_stan_data's data and have the same posterior
MODELS = {'u': ('model_switchable_u.stan', u_init), 'raw_jac': ('model_switchable_raw_jac.stan', raw_init),
          'rates': ('model_switchable_rates.stan', rates_init)}

priorDic = {
    "cost":     stats.beta(2, 5),
    "u":        stats.lognorm(s=.5, scale=3),
    "n0":       stats.beta(2, 2),
    "rFrac":    stats.beta(1, 30),
    "rS":       stats.norm(loc=.027, scale=0.005), 
    "dD":       stats.lognorm(s=0.3, scale=1.5),
    'sigma':    stats.lognorm(s=.25, scale=.25),
}


def fit_patient(synth_df, patient_id, stan_model, priorDic, rng, out_dir, adapt_delta=0.8, init=u_init, timeout=300):
    """Fit one patient, return the posterior as arviz InferenceData, while saving unconstrained chains.
    init(priorDic, rng) gives pathfinder's start in the model's parameters (raw_init, rates_init for the other models in MODELS)"""
    priors = {f"{k}_prior": to_stan_prior(d) for k, d in priorDic.items()}
    stan_data = make_stan_data(synth_df, patient_id=patient_id, priors=priors)
    pathfinder_fit = stan_model.pathfinder(data=stan_data, seed=SAMPLER_SEED, inits=init(priorDic, rng))
    post_fit = stan_model.sample(data=stan_data, chains=4, seed=SAMPLER_SEED, inits=pathfinder_fit.create_inits(seed=SAMPLER_SEED, chains=4),
                                 adapt_delta=adapt_delta, show_progress=True, output_dir=out_dir / f'patient_{patient_id}_csv',
                                 save_warmup=True, save_latent_dynamics=True, timeout=timeout)
    idata = az.from_cmdstanpy(posterior=post_fit) 
    idata['sample_stats'] = idata.sample_stats.to_dataset().isel(draw=slice(-idata.posterior.sizes['draw'], None)).assign_coords(draw=idata.posterior.draw)
    return idata


def fit_all(synth_df, out_dir, stan_model, priorDic, seed=1, overwrite=True, adapt_delta=0.8, init=u_init):
    """Fit every patient in synth_df, save patient_<id>.nc to out_dir and the failed fits to out_dir/errors.txt.
    Also writes a copy of the Stan file and run_info.json (settings, priors, time per fit and in total) to out_dir"""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = synth_df['PatientId'].unique()
    errors = []

    #record what went into the fits. The Stan file is copied as it is on disk now; if it changed after the model was
    #compiled, the fits used the older compiled version, which stan_file_newer_than_exe flags
    stan_file = Path(stan_model.stan_file)
    shutil.copy(stan_file, out_dir / stan_file.name)
    run_info = {
        'started': datetime.now().astimezone().isoformat(timespec='seconds'),
        'stan_file': str(stan_file),
        'stan_file_newer_than_exe': os.path.getmtime(stan_file) > os.path.getmtime(stan_model.exe_file),
        'adapt_delta': adapt_delta, 'init_seed': seed, 'sampler_seed': SAMPLER_SEED, 'overwrite': overwrite,
        'priors': {k: {'scipy': f'{d.dist.name}(args={list(d.args)}, kwds={d.kwds})', 'stan': [float(x) for x in to_stan_prior(d)]}
                   for k, d in priorDic.items()},
        'fits': [], 'total_seconds': 0.0,
    }
    def save_run_info():
        (out_dir / 'run_info.json').write_text(json.dumps(run_info, indent=2))
    save_run_info()
    t_run = time.time()

    #fit patients
    for patient_id in patient_ids:
        if (out_dir / f'patient_{patient_id}.nc').exists() and not overwrite:
            continue
        t_fit = time.time()
        try:
            idata = fit_patient(synth_df, patient_id, stan_model, priorDic, rng, out_dir, adapt_delta=adapt_delta, init=init)
            idata.to_netcdf(out_dir / f'patient_{patient_id}.nc')
            print(f'Fitted patient {patient_id} out of {len(patient_ids)}')
            status = 'ok'
        except Exception as e:
            #catch error and continue loop.
            errors.append(f'Patient {patient_id}: {e}')
            status = f'error: {e}'
        #wall time per fit (pathfinder + sampling + saving), written after each fit so an interrupted run keeps it
        run_info['fits'].append({'patient': int(patient_id), 'seconds': round(time.time() - t_fit, 2), 'status': status})
        run_info['total_seconds'] = round(time.time() - t_run, 2)
        save_run_info()
    run_info['finished'] = datetime.now().astimezone().isoformat(timespec='seconds')
    save_run_info()

    #save errors to file
    with open(out_dir / 'errors.txt', 'w') as f:
        for error in errors:
            f.write(error + '\n')
    return errors


if __name__ == '__main__':
    #e.g. python fit_routine.py --data-dir ../param_recov_u --adapt-delta 0.95 --fits-dir ../param_recov_u/fits_ad95
    #--model raw_jac / rates fit the same model in the raw ODE parameters / the net growth rates (see MODELS)
    #the data come from generate_synth.py
    parser = argparse.ArgumentParser(description='Fit every patient in a synthetic trial with model_switchable_u.stan')
    parser.add_argument('--model', choices=list(MODELS), default='u', help='u: model_switchable_u.stan, raw_jac: model_switchable_raw_jac.stan, rates: model_switchable_rates.stan')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'synth_data_u', help='folder with synth_patients.csv')
    parser.add_argument('--adapt-delta', type=float, default=0.8)
    parser.add_argument('--fits-dir', type=Path, default=None, help='folder for the fits, default <data-dir>/fits')
    parser.add_argument('--overwrite', action='store_true', help='fit into --fits-dir even if it already exists')
    args = parser.parse_args()

    fits_dir = (args.fits_dir or args.data_dir / 'fits').resolve()
    #fit_all overwrites patient_<id>.nc one by one, so an old run's fits would be mixed with the new ones
    if fits_dir.exists() and not args.overwrite:
        parser.error(f'{fits_dir} already exists; pass --overwrite to fit into it anyway')
    synth_df = pd.read_csv(args.data_dir / 'synth_patients.csv')
    stan_file, init = MODELS[args.model]
    stan_model = cmdstanpy.CmdStanModel(stan_file=ROOT / 'stan' / stan_file)
    fit_all(synth_df, fits_dir, stan_model, priorDic, seed=1, adapt_delta=args.adapt_delta, init=init)
