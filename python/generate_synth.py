"""Generate a synthetic AT trial for model_switchable_u.stan: synth_patients.csv and true_params.csv in --out-dir.
Fit it afterwards with fit_routine.py --data-dir <out-dir>"""
import argparse
from pathlib import Path

from fit_routine import ROOT, priorDic
from utils.synth_data import generate_u_patients

if __name__ == '__main__':
    #e.g. python generate_synth.py --n 200 --out-dir ../param_recov_u
    parser = argparse.ArgumentParser(description='Generate synthetic patients from fit_routine.priorDic')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'synth_data_u', help='folder for synth_patients.csv and true_params.csv')
    parser.add_argument('--overwrite', action='store_true', help='delete and recreate --out-dir if it already exists')
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    #generate_u_patients deletes out_dir first (fits included), so only allow an existing folder with --overwrite
    if out_dir.exists() and not args.overwrite:
        parser.error(f'{out_dir} already exists; pass --overwrite to delete and recreate it')
    generate_u_patients(priorDic, n_patients=args.n, out_dir=out_dir, seed=args.seed)
