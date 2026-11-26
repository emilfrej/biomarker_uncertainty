
import shutil
import sys

import cmdstanpy
import numpy as np
import pandas as pd

from fit_routine import ROOT, MODELS, priorDic, fit_all
from utils.synth_data import generate_u_patients
from utils.fit_diagnostics import diagnose_fits

N = 100
FIT = ['logstate', 'turnover']  
REGENERATE = False  
REFIT = False  
ADAPT_DELTA = 0.95  
DIAGNOSE = True  
PAIRS = True  
OUT = ROOT / 'experiments' / 'logstate_vs_u' / 'out' 

if __name__ == '__main__':
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
                    adapt_delta=ADAPT_DELTA, save_warmup=True, save_latent_dynamics=True) 
        #also for fits that existed: diagnostics.csv, diagnostics/ and pairs/ are deleted and redrawn
        if DIAGNOSE or PAIRS:
            print(f'diagnosing {model}')
            diagnose_fits(fits_dir, diagnose=DIAGNOSE, pairs=PAIRS)
