#tests for python/utils/stan_data.py. Run from biomarker_uncertainty/: ../at-modelling/.venv/bin/python -m pytest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from utils.stan_data import make_stan_data, to_stan_prior

priors = {f'{p}_prior': [1, 1] for p in ['cost', 'turnover', 'n0', 'rFrac', 'rS', 'dD', 'sigma']}


def test_to_stan_prior():
    assert to_stan_prior(stats.beta(2, 3)) == [2, 3]
    assert to_stan_prior(stats.lognorm(s=0.6, scale=0.012)) == pytest.approx([np.log(0.012), 0.6])
    with pytest.raises(ValueError):
        to_stan_prior(stats.norm(0, 1))




def test_raw_init_maps_back_to_u_init():
    from utils.stan_data import raw_init, u_init
    priorDic = {'cost': stats.beta(1.5, 4), 'u': stats.lognorm(s=0.5, scale=2.5), 'n0': stats.beta(2, 3), 'rFrac': stats.beta(1, 30),
                'rS': stats.lognorm(s=0.3, scale=0.027), 'dD': stats.lognorm(s=0.05, scale=1.5), 'sigma': stats.lognorm(s=.2, scale=.2)}
    for seed in range(20):
        u, r = u_init(priorDic, np.random.default_rng(seed)), raw_init(priorDic, np.random.default_rng(seed))
        n0 = r['S0'] + r['R0']
        #inside model_switchable_raw_jac.stan's bounds
        assert 0 < r['rR'] < r['rS'] and n0 < 1 and 0 < r['d'] < r['rR'] * (1 - n0)
        #and the same point as u_init's draw
        assert 1 - r['rR'] / r['rS'] == pytest.approx(u['cost_free'][0])
        assert (1 - r['d'] / r['rR']) / n0 == pytest.approx(u['u_free'][0])
        assert n0 == pytest.approx(u['n0_free'][0]) and r['R0'] / n0 == pytest.approx(u['rFrac_free'][0])


def test_rates_init_maps_back_to_u_init():
    from utils.stan_data import rates_init, u_init
    priorDic = {'cost': stats.beta(1.5, 4), 'u': stats.lognorm(s=0.5, scale=2.5), 'n0': stats.beta(2, 3), 'rFrac': stats.beta(1, 30),
                'rS': stats.lognorm(s=0.3, scale=0.027), 'dD': stats.lognorm(s=0.05, scale=1.5), 'sigma': stats.lognorm(s=.2, scale=.2)}
    for seed in range(20):
        u, r = u_init(priorDic, np.random.default_rng(seed)), rates_init(priorDic, np.random.default_rng(seed))
        #inside model_switchable_rates.stan's bounds
        assert 0 < r['g_R'] < r['g_off'] and r['g_on'] < r['g_off']
        assert r['g_off'] * r['n0'] < r['a'] < r['g_off'] * r['n0'] + r['g_R'] * (1 - r['n0'])
        #and back to u_init's draw, with the model's map: rS = a / n0, d = rS - g_off, rR = d + g_R, dD = (g_off - g_on) / rS
        rS = r['a'] / r['n0']; d = rS - r['g_off']; rR = d + r['g_R']
        assert rS == pytest.approx(u['rS_free'][0]) and (r['g_off'] - r['g_on']) / rS == pytest.approx(u['dD_free'][0])
        assert 1 - rR / rS == pytest.approx(u['cost_free'][0]) and (1 - d / rR) / r['n0'] == pytest.approx(u['u_free'][0])
