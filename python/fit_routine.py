import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import arviz as az
import cmdstanpy
import numpy as np
from scipy import stats

from utils.stan_data import to_stan_prior, make_stan_data, u_init, raw_init, rates_init, rstar_init, turnover_init

ROOT = Path(__file__).resolve().parents[1]
SAMPLER_SEED = 4  
MODELS = {'u': ('model_switchable_u.stan', u_init), 'raw_jac': ('model_switchable_raw_jac.stan', raw_init),
          'rates': ('model_switchable_rates.stan', rates_init), 'logstate': ('model_switchable_u_logstate.stan', u_init),
          'rstar': ('model_switchable_u_rstar.stan', rstar_init), 'turnover': ('model_switchable_u_turnover.stan', turnover_init)}

priorDic = {
    "cost":     stats.beta(2, 5),
    "u":        stats.lognorm(s=.25, scale=2),
    "n0":       stats.beta(4, 5), 
    "rFrac":    stats.beta(1, 30),
    "rS":       stats.lognorm(s=0.3, scale=0.027),
    "dD":       stats.lognorm(s=0.1, scale=1.5),
    'sigma':    stats.lognorm(s=.25, scale=.25),
}


def fit_patient(synth_df, patient_id, stan_model, priorDic, rng, out_dir, adapt_delta=0.8, init=u_init, timeout=300, show_progress=True,
                iter_warmup=1000, iter_sampling=1000, *, save_warmup, save_latent_dynamics):
    """Fit one patient, return the posterior as arviz InferenceData, while saving unconstrained chains (save_latent_dynamics).
    init(priorDic, rng) gives pathfinder's start in the model's parameters (raw_init, rates_init for the other models in MODELS)"""
    priors = {f"{k}_prior": to_stan_prior(d) for k, d in priorDic.items()}
    stan_data = make_stan_data(synth_df, patient_id=patient_id, priors=priors)
    pathfinder_fit = stan_model.pathfinder(data=stan_data, seed=SAMPLER_SEED, inits=init(priorDic, rng))
    post_fit = stan_model.sample(data=stan_data, chains=4, seed=SAMPLER_SEED, inits=pathfinder_fit.create_inits(seed=SAMPLER_SEED, chains=4),
                                 adapt_delta=adapt_delta, show_progress=show_progress, output_dir=out_dir / f'patient_{patient_id}_csv',
                                 iter_warmup=iter_warmup, iter_sampling=iter_sampling, save_warmup=save_warmup,
                                 save_latent_dynamics=save_latent_dynamics, timeout=timeout)
    idata = az.from_cmdstanpy(posterior=post_fit) 
    idata['sample_stats'] = idata.sample_stats.to_dataset().isel(draw=slice(-idata.posterior.sizes['draw'], None)).assign_coords(draw=idata.posterior.draw)
    return idata


def fit_all(synth_df, out_dir, stan_model, priorDic, save_warmup, save_latent_dynamics, seed=1, overwrite=True, adapt_delta=0.8, init=u_init, n_workers=1,
            iter_warmup=1000, iter_sampling=1000,):
    """Fit every patient in synth_df, save patient_<id>.nc to out_dir and the failed fits to out_dir/errors.txt.
    Also writes a copy of the Stan file and run_info.json (settings, priors, time per fit and in total) to out_dir.
    n_workers: patients fitted at once. Each fit runs its 4 chains as separate CmdStan processes, so about cores / 4.
    Each patient's Pathfinder start comes from its own rng (seed, patient id), so a fit does not depend on n_workers or order.
    iter_warmup, iter_sampling, save_warmup, save_latent_dynamics go to fit_patient; diagnose_fits needs save_latent_dynamics"""
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
        'adapt_delta': adapt_delta, 'init_seed': seed, 'sampler_seed': SAMPLER_SEED, 'overwrite': overwrite, 'n_workers': n_workers,
        'iter_warmup': iter_warmup, 'iter_sampling': iter_sampling, 'save_warmup': save_warmup, 'save_latent_dynamics': save_latent_dynamics,
        'priors': {k: {'scipy': f'{d.dist.name}(args={list(d.args)}, kwds={d.kwds})', 'stan': [float(x) for x in to_stan_prior(d)]}
                   for k, d in priorDic.items()},
        'fits': [], 'total_seconds': 0.0,
    }
    def save_run_info():
        (out_dir / 'run_info.json').write_text(json.dumps(run_info, indent=2))
    save_run_info()
    t_run = time.time()

    def fit_one(patient_id):
        """Fit one patient in a worker thread. Returns the posterior, or the exception if the fit failed, and the wall time"""
        t_fit = time.time()
        rng = np.random.default_rng([seed, int(patient_id)])
        try:
            result = fit_patient(synth_df, patient_id, stan_model, priorDic, rng, out_dir, adapt_delta=adapt_delta, init=init,
                                 show_progress=n_workers == 1, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
                                 save_warmup=save_warmup, save_latent_dynamics=save_latent_dynamics)
        except Exception as e:
            result = e
        return result, time.time() - t_fit

    #fit patients, n_workers at a time. Threads are enough, as the sampling runs in CmdStan processes; the netCDF writes stay
    #in this thread, as HDF5 is not thread-safe. Results come in as fits finish, so run_info['fits'] is in finishing order
    todo = [p for p in patient_ids if overwrite or not (out_dir / f'patient_{p}.nc').exists()]
    pool = ThreadPoolExecutor(max_workers=n_workers)
    try:
        futures = {pool.submit(fit_one, p): p for p in todo}
        for future in as_completed(futures):
            patient_id = futures[future]
            result, seconds = future.result()
            t_save = time.time()
            try:
                if isinstance(result, Exception):
                    raise result
                result.to_netcdf(out_dir / f'patient_{patient_id}.nc')
                print(f'Fitted patient {patient_id} out of {len(patient_ids)}')
                status = 'ok'
            except Exception as e:
                #catch error and continue loop.
                errors.append(f'Patient {patient_id}: {e}')
                status = f'error: {e}'
            #wall time per fit (pathfinder + sampling + saving), written after each fit so an interrupted run keeps it.
            #With n_workers > 1 the fits share the cores, so compare times only between runs with the same n_workers
            run_info['fits'].append({'patient': int(patient_id), 'seconds': round(seconds + time.time() - t_save, 2), 'status': status})
            run_info['total_seconds'] = round(time.time() - t_run, 2)
            save_run_info()
    finally:
        #on an error or Ctrl-C, drop the fits not yet started; the running ones finish (sample's timeout caps them)
        pool.shutdown(cancel_futures=True)
    run_info['finished'] = datetime.now().astimezone().isoformat(timespec='seconds')
    save_run_info()

    #save errors to file
    with open(out_dir / 'errors.txt', 'w') as f:
        for error in errors:
            f.write(error + '\n')
    return errors
