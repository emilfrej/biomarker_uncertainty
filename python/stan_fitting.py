"""Helpers for fitting the synthetic AT patients made in genertate_synth_AT_trial_data.ipynb with stan/model_switchable.stan.
Python ports of load_synth_patient and make_switchable_stan_data in R/synth_functions.R"""
import numpy as np
import pandas as pd

switchable_params = ['cost', 'turnover', 'n0', 'rFrac', 'rS', 'K', 'dD']

def load_synth_visits(synth_df: pd.DataFrame, patient_id: int) -> pd.DataFrame:
    """One patient's visits with CycleId and dose segments, as load_synth_patient in R/synth_functions.R.
    A cycle is a treatment period and the holiday after it. The visit where treatment restarts still belongs to the old cycle,
    and the baseline has no dose before it, so it is never a restart"""
    visits = synth_df[(synth_df['PatientId'] == patient_id) & synth_df['Observation'].notna()].sort_values('Time')
    if visits.empty:
        raise ValueError(f"no synthetic patient {patient_id}")
    dose = visits['DrugConcentration']
    dose_before = dose.shift()
    restart = (dose_before > 0) & (dose_before.shift() == 0)
    return visits.assign(CycleId=1 + restart.cumsum(), segment=dose.ne(dose.shift()).cumsum())

def make_stan_data(synth_df: pd.DataFrame, patient_id: int, priors: dict, n_cycles: float = np.inf, fixed_vals: dict = {},
                   prior_only: int = 0, run_diagnostics: int = 1) -> dict:
    """Stan data for model_switchable.stan from one patient's first n_cycles cycles, as make_switchable_stan_data in R"""
    unknown = set(fixed_vals) - set(switchable_params)
    if unknown:
        raise ValueError(f"not switchable: {unknown}")
    missing = [p for p in switchable_params + ['sigma'] if p not in fixed_vals and f"{p}_prior" not in priors]
    if missing:
        raise ValueError(f"no prior for fitted params: {missing}")

    visits = load_synth_visits(synth_df, patient_id)
    visits = visits[visits['CycleId'] <= n_cycles]

    #segments of constant dose, from their first visit to the next segment's first visit. The last ends at the last visit
    segments = visits.groupby('segment').agg(start=('Time', 'first'), treatment=('DrugConcentration', 'first'))
    segments['end'] = segments['start'].shift(-1, fill_value=visits['Time'].max())
    segments = segments[segments['end'] > segments['start']]  #a dose change at the last visit has no follow up
    #times the solver reports: every visit and every segment end, sorted and unique. Indices are 1-based for Stan
    output_times = np.unique(np.concatenate([visits['Time'], segments['end']]))
    to_idx = np.searchsorted(output_times, segments['end']) + 1
    from_idx = np.concatenate([[1], to_idx[:-1] + 1])

    return {'n_segments':             len(segments),
            'starting_segment_times': segments['start'].astype(int).tolist(),
            'end_segment_times':      segments['end'].astype(int).tolist(),
            'treatment_indicator':    segments['treatment'].astype(int).tolist(),
            'n_out':                  len(output_times),
            'output_times':           output_times.tolist(),
            'segment_from_idx':       from_idx.tolist(),
            'segment_to_idx':         to_idx.tolist(),
            'n_obs':                  len(visits),
            'obs_idx':                (np.searchsorted(output_times, visits['Time']) + 1).tolist(),
            'PSA_norm':               visits['Observation'].tolist(),
            **{f"fit_{p}": int(p not in fixed_vals) for p in switchable_params},
            **{f"{p}_fixed_val": fixed_vals.get(p, 0) for p in switchable_params},
            **{f"{p}_prior": [1, 1] for p in fixed_vals},  #placeholders, Stan needs every prior but ignores those of fixed params
            **priors,
            'prior_only':             prior_only,
            'run_diagnostics':        run_diagnostics}
