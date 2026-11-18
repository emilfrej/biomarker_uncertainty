"""One simulated patient for simulation-based calibration with the R SBC package (called through reticulate from
R/sbc_minimal.R).

True values are drawn from fit_routine.priorDic within model_switchable_u.stan's own bounds only (1 < u < 1 / n0,
K = 1), with no further rejection, so they follow the fit's prior as SBC requires (generate_u_patients also rejects
patients that do not progress, BUGS.md D5). The record is simulated with the at_modelling simulator (adaptive therapy,
multiplicative noise; it stops on the true size, BUGS.md D26) and turned into Stan data with make_stan_data."""
import numpy as np
from at_modelling.LotkaVolterraModel import LotkaVolterraModel

from fit_routine import priorDic
from utils.stan_data import to_stan_prior, make_stan_data
from utils.synth_data import complete_param_dict, add_cycle_col

#all floats: reticulate turns a mixed int/float list (e.g. beta(1.5, 4)) into an R list, which Stan reads as 2 x 1
PRIORS = {f'{k}_prior': [float(x) for x in to_stan_prior(d)] for k, d in priorDic.items()}


def simulate_one(seed, t_end=5000, interval=28, at_threshold=0.5):
    """True values (dict) and Stan data (dict) for one patient. seed makes both reproducible"""
    seed = int(seed)
    rng = np.random.default_rng(seed)
    while True:
        truth = {k: float(d.rvs(random_state=rng)) for k, d in priorDic.items()}
        if 1 < truth['u'] < 1 / truth['n0']:  #the model's bounds on u
            break
    turnover = (1 - truth['cost']) * (1 - truth['u'] * truth['n0'])
    params = complete_param_dict(**{k: v for k, v in truth.items() if k != 'u'}, turnover=turnover, K=1)
    np.random.seed(seed % 2**32)  #the simulator's observation noise uses numpy's global RNG
    model = LotkaVolterraModel(method='RK45', dt=1)
    model.set_params(**{k: v for k, v in params.items() if k not in ('Cost', 'Turnover', 'n0')})
    model.simulate_at(at_threshold=at_threshold, interval_length=interval, t_end=t_end)
    patient = add_cycle_col(model.results_df(), 'DrugConcentration').assign(PatientId=0)
    return {'truth': truth, 'stan_data': make_stan_data(patient, patient_id=0, priors=PRIORS)}
