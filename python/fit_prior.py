
import shutil
import sys

import cmdstanpy
import numpy as np
import pandas as pd

from fit_routine import ROOT, MODELS, priorDic, fit_all
from utils.synth_data import generate_u_patients
from utils.fit_diagnostics import diagnose_fits

N = 1000
FIT = ['logstate']  
REGENERATE = False  
REFIT = False  
ADAPT_DELTA = 0.95  
DIAGNOSE = False 
PAIRS = False
N_WORKERS = 2  #patients fitted at once
ITER_WARMUP = 1000
ITER_SAMPLING = 1000
SAVE_WARMUP = True  #warmup rows in the chain CSVs
SAVE_LATENT_DYNAMICS = True  #the diagnostic_ CSVs (unconstrained draws)
OUT = ROOT / 'experiments' / 'logstate_vs_u' / 'sbc' 

if __name__ == '__main__':
    if (DIAGNOSE or PAIRS) and not SAVE_LATENT_DYNAMICS:
        sys.exit('DIAGNOSE and PAIRS read the diagnostic CSVs; set SAVE_LATENT_DYNAMICS = True or turn them off')
    print(f'div_experiment: N={N}, REGENERATE={REGENERATE}, models={FIT}, REFIT={REFIT}, ADAPT_DELTA={ADAPT_DELTA}, DIAGNOSE={DIAGNOSE}, PAIRS={PAIRS}, output in {OUT}')
    if REGENERATE or not (OUT / 'true_params.csv').exists():
        np.random.seed(1)
        generate_u_patients(priorDic, n_patients=N, out_dir=OUT, check_progression=False, progression='no_effect')
    else:
        n_old = len(pd.read_csv(OUT / 'true_params.csv'))
        if n_old != N:
            sys.exit(f'out/ has a cohort of {n_old} patients, not N={N}; set REGENERATE = True to replace it')
    synth_df = pd.read_csv(OUT / 'synth_patients.csv')

    for model in FIT:
        fits_dir = OUT / f'fits_{model}'
        if REFIT:
            shutil.rmtree(fits_dir, ignore_errors=True)
        if fits_dir.exists():
            print(f'skipping {model}: {fits_dir} exists')
        else:
            stan_file, init = MODELS[model]
            fit_all(synth_df, fits_dir, cmdstanpy.CmdStanModel(stan_file=ROOT / 'stan' / stan_file), priorDic, seed=1, init=init,
                    adapt_delta=ADAPT_DELTA, n_workers=N_WORKERS, iter_warmup=ITER_WARMUP, iter_sampling=ITER_SAMPLING,
                    save_warmup=SAVE_WARMUP, save_latent_dynamics=SAVE_LATENT_DYNAMICS)
        #also for fits that existed: diagnostics.csv, diagnostics/ and pairs/ are deleted and redrawn
        if DIAGNOSE or PAIRS:
            print(f'diagnosing {model}')
            diagnose_fits(fits_dir, diagnose=DIAGNOSE, pairs=PAIRS)
