# Forward simulation wrapper around Kit Gallagher's Lotka-Volterra model (AT_Model_Comparison).
# Callable from Python directly, or from R via reticulate::source_python("python/forward_sim.py").
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this file lives in python/
sys.path.append(os.path.join(_PROJECT_ROOT, "AT_Model_Comparison", "utils"))
from odeModels import LotkaVolterraModel


def run_forward_sim(params, schedule, dt=0.1):
    """Simulate the LV model under a fixed on/off dosing schedule.

    params: dict of model parameters, e.g. {'rS': 0.027, 'rR': 0.027, 'K': 1, 'dD': 1.5, 'dS': 0,
            'dR': 0, 'S0': 0.74, 'R0': 0.01, 'DMax': 1}. Missing keys keep the model defaults.
    schedule: DataFrame with columns start, end, dose (one row per interval),
              or a list of [start, end, dose] lists.
    dt: output time resolution in days.

    Returns a DataFrame with columns Time, DrugConcentration, S, R, TumourSize.
    """
    if isinstance(schedule, pd.DataFrame):
        schedule = schedule[["start", "end", "dose"]].to_numpy(dtype=float).tolist()

    model = LotkaVolterraModel()
    model.SetParams(**params)
    model.Simulate(schedule, dt=dt)
    if not model.successB:
        raise RuntimeError("ODE solver reported a problem: %s" % model.errMessage)

    out = model.resultsDf.reset_index(drop=True)  # index restarts per interval otherwise
    out["Time"] = out["Time"].round(8)  # np.arange float error (e.g. 249.9999999) breaks Time == 250
    return out
