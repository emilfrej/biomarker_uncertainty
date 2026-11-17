from pathlib import Path

import arviz as az
import cmdstanpy
import numpy as np
from scipy import stats

from utils.stan_data import to_stan_prior, make_stan_data, u_init
from utils.synth_data import generate_u_patients

ROOT = Path(__file__).resolve().parents[1]

priorDic = {
    "cost":     stats.beta(1.5, 4),
    "u":        stats.lognorm(s=0.5, scale=2.5),
    "n0":       stats.beta(2, 3),
    "rFrac":    stats.beta(1, 30),
    "rS":       stats.lognorm(s=0.3, scale=0.027),
    "dD":       stats.lognorm(s=0.05, scale=1.5),
    'sigma':    stats.lognorm(s=.2, scale=.2),
}


def fit_patient(synth_df, patient_id, stan_model, priorDic, rng, out_dir):
    """Fit one patient, return the posterior as arviz InferenceData, while saving unconstrained chains"""
    priors = {f"{k}_prior": to_stan_prior(d) for k, d in priorDic.items()}
    stan_data = make_stan_data(synth_df, patient_id=patient_id, priors=priors)
    pathfinder_fit = stan_model.pathfinder(data=stan_data, seed=4, inits=u_init(priorDic, rng))
    post_fit = stan_model.sample(data=stan_data, chains=4, seed=4, inits=pathfinder_fit.create_inits(seed=4, chains=4),
                                 adapt_delta=0.8, show_progress=True, output_dir=out_dir / f'patient_{patient_id}_csv',
                                 save_warmup=True, save_latent_dynamics=True, timeout=300)
    idata = az.from_cmdstanpy(posterior=post_fit) 
    idata['sample_stats'] = idata.sample_stats.to_dataset().isel(draw=slice(-idata.posterior.sizes['draw'], None)).assign_coords(draw=idata.posterior.draw)
    return idata


def fit_all(synth_df, out_dir, stan_model, priorDic, seed=1, overwrite=True):
    """Fit every patient in synth_df, save patient_<id>.nc to out_dir and the failed fits to out_dir/errors.txt"""
    rng = np.random.default_rng(seed)  
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_ids = synth_df['PatientId'].unique()
    errors = []

    #fit patients
    for patient_id in patient_ids:
        if (out_dir / f'patient_{patient_id}.nc').exists() and not overwrite:
            continue
        try:
            idata = fit_patient(synth_df, patient_id, stan_model, priorDic, rng, out_dir)
            idata.to_netcdf(out_dir / f'patient_{patient_id}.nc')
            print(f'Fitted patient {patient_id} out of {len(patient_ids)}')
        except Exception as e:
            #catch error and continue loop.
            errors.append(f'Patient {patient_id}: {e}')

    #save errors to file
    with open(out_dir / 'errors.txt', 'w') as f:
        for error in errors:
            f.write(error + '\n')
    return errors


if __name__ == '__main__':
    synth_dir = ROOT / 'synth_data_u'
    synth_df, _ = generate_u_patients(priorDic, n_patients=10, out_dir=synth_dir, seed=1)
    stan_model = cmdstanpy.CmdStanModel(stan_file=ROOT / 'stan' / 'model_switchable_u.stan')
    fit_all(synth_df, synth_dir / 'fits', stan_model, priorDic, seed=1)
