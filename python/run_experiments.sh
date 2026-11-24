#!/bin/bash
#Generate one synthetic cohort and fit it five ways: NUTS (adapt_delta 0.8), NUTS on the same model in raw ODE parameters
#(model_switchable_raw_jac.stan) and in net growth rates (model_switchable_rates.stan), WALNUTS and Gallagher et al.'s full-history least squares. Run from biomarker_uncertainty/python:  bash run_experiments.sh [out-dir] [n-patients]
#Output: <out-dir>/{synth_patients.csv, true_params.csv, fits_ad80/, fits_ad80_rawjac/, fits_ad80_rates/, fits_walnuts/, gallagher_full_history/}
set -euo pipefail
OUT=${1:-../param_recov_u_exp100}
N=${2:-100}
PY=../../at-modelling/.venv/bin/python

$PY generate_synth.py --n "$N" --out-dir "$OUT"
$PY fit_routine.py --data-dir "$OUT" --adapt-delta 0.8 --fits-dir "$OUT/fits_ad80"
$PY fit_routine.py --data-dir "$OUT" --adapt-delta 0.8 --model raw_jac --fits-dir "$OUT/fits_ad80_rawjac"
$PY fit_routine.py --data-dir "$OUT" --adapt-delta 0.8 --model rates --fits-dir "$OUT/fits_ad80_rates"
$PY fit_walnuts.py --data-dir "$OUT" --fits-dir "$OUT/fits_walnuts"
#lmfit is not in the at-modelling venv (see gallagher_full_history_fit.py)
uv run --no-project --python 3.12 --with lmfit --with seaborn --with tqdm --with pandas --with scipy --with matplotlib \
    python gallagher_full_history_fit.py --cohort-dir "$OUT"
