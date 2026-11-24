#tests for python/utils/synth_data.py. Run from biomarker_uncertainty/: ../at-modelling/.venv/bin/python -m pytest
import numpy as np
import pytest
from scipy import stats

from utils.synth_data import generate_u_patients


def test_generated_patients_are_inside_the_model_bounds(tmp_path):
    #u prior with about half its mass below 1, so the u >= 1 rejection is exercised (BUGS.md D27)
    priorDic = {'cost': stats.beta(1.5, 4), 'u': stats.lognorm(s=0.5, scale=1.0), 'n0': stats.beta(2, 3), 'rFrac': stats.beta(1, 30),
                'rS': stats.lognorm(s=0.3, scale=0.027), 'dD': stats.lognorm(s=0.05, scale=1.5), 'sigma': stats.lognorm(s=.2, scale=.2)}
    np.random.seed(1)
    _, true_params = generate_u_patients(priorDic, n_patients=15, out_dir=tmp_path / 'synth')
    assert (true_params.u >= 1).all()
    assert (true_params.u * true_params.n0 < 1).all()


@pytest.mark.slow
def test_resistant_non_progressors_match_u_below_threshold(tmp_path):
    """Under progression='no_effect' a fully resistant tumour levels off at u x start, so it cannot progress when u <= 1.2.
    The number of 1000 patients that end mostly resistant (R / (S + R) > 0.9) without progressing should match the expected
    number of u <= 1.2 draws from fit_routine.priorDic within the generator's bounds. About a minute"""
    from fit_routine import priorDic
    #expected share of u <= 1.2: priorDic draws inside in_stan_bounds (u >= 1, u * n0 < K = 1, rS > 0), vectorised
    rng = np.random.default_rng(1)
    d = {k: dist.rvs(1_000_000, random_state=rng) for k, dist in priorDic.items()}
    in_bounds = (d['u'] >= 1) & (d['u'] * d['n0'] < 1) & (d['rS'] > 0)
    p_low_u = ((d['u'] <= 1.2) & in_bounds).sum() / in_bounds.sum()

    np.random.seed(1)
    synth_df, _ = generate_u_patients(priorDic, n_patients=1000, out_dir=tmp_path / 'synth', check_progression=False, progression='no_effect')
    last = synth_df.sort_values('Time').groupby('PatientId').tail(1)
    n_resistant = int((~last['Progressed'].astype(bool) & (last['R'] / (last['S'] + last['R']) > 0.9)).sum())
    assert stats.binomtest(n_resistant, 1000, p_low_u).pvalue > 0.01, f'{n_resistant} resistant non-progressors, expected {1000 * p_low_u:.1f}'
