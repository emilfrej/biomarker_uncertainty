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


