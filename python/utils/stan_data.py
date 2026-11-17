"""Helpers for fitting model_switchable.stan from genertate_synth_AT_trial_data.ipynb"""
import numpy as np
import pandas as pd


def to_stan_prior(dist):
    """scipy beta(a, b) -> [a, b], scipy lognorm(s, scale) -> [log(scale), s] as in Stan's lognormal(mu, sigma)"""
    if dist.dist.name == 'beta':
        return list(dist.args)
    if dist.dist.name == 'lognorm':
        return [float(np.log(dist.kwds['scale'])), dist.kwds['s']]
    raise ValueError(f"No Stan conversion for {dist.dist.name}")


def prior_init(priorDic, rng):
    """Inits for one chain, drawn from the scipy priors. Stan names the fitted parameters <name>_free, except sigma"""
    draw = {k: d.rvs(random_state=rng) for k, d in priorDic.items()}
    return {**{f'{k}_free': [v] for k, v in draw.items() if k != 'sigma'}, 'sigma': draw['sigma']}


def make_stan_data(patients_df, patient_id, priors):
    """Stan data for model_switchable.stan from one patient's rows (columns PatientId, Time, DrugConcentration, Observation)"""
    visits = patients_df[patients_df['PatientId'] == patient_id]
    obs = visits[visits['Observation'].notna()] 
    visits['dose_shifted'] = visits['DrugConcentration'].shift()
    visits = visits[visits['Observation'].notna() | (visits['DrugConcentration'] != visits['dose_shifted'])] #keep rows with obs, and rows in real data with only a treatment change but no obs
    visits = visits[visits['Time'] <= obs['Time'].max()] #drop rows after last obs if no obs
    visits['segment'] = (visits['DrugConcentration']!= visits['dose_shifted']).cumsum()

    segments = visits.groupby('segment', as_index=False).agg(segment_start_time=('Time', 'first'),
                                                        Treatment=('DrugConcentration', 'first'))
    segments['segment_end_time'] = segments['segment_start_time'].shift(-1, fill_value=visits['Time'].max()) # give last end time final time in data
    segments = segments[segments['segment_end_time'] > segments['segment_start_time']] #treatment change at last visit has no follow up, so drop it
    output_times = pd.concat([visits['Time'], segments['segment_end_time']]).drop_duplicates().sort_values().tolist() #account for rows in real data that only treatment change but no ons
    segments['segment_to_idx'] = [output_times.index(t) + 1 for t in segments['segment_end_time']] # get idx of row n data where segment ends. account for python being 0 indexed
    segments['segment_from_idx'] = segments['segment_to_idx'].shift(1, fill_value=0) + 1 #same as abv

    return {
        #segments for ODE
        'n_out':                  len(output_times),
        'output_times':           output_times,
        'n_segments':             len(segments),
        'segment_from_idx':       segments['segment_from_idx'].tolist(),
        'segment_to_idx':         segments['segment_to_idx'].tolist(),
        'starting_segment_times': segments['segment_start_time'].astype(int).tolist(),
        'end_segment_times':      segments['segment_end_time'].astype(int).tolist(),
        'treatment_indicator':    segments['Treatment'].astype(int).tolist(),
        #observations
        'n_obs':                  len(obs),
        'obs_idx':                [output_times.index(t) + 1 for t in obs['Time']],
        'PSA_norm':               obs['Observation'].tolist(),
        #fix param flags
        'fit_cost': 1, 'fit_turnover': 1, 'fit_n0': 1, 'fit_rFrac': 1, 'fit_rS': 1, 'fit_K': 0, 'fit_dD': 1,
        'cost_fixed_val': 0, 'turnover_fixed_val': 0, 'n0_fixed_val': 0, 'rFrac_fixed_val': 0,
        'rS_fixed_val': 0, 'K_fixed_val': 1, 'dD_fixed_val': 0,
        **priors,
        'K_prior':                [1, 1],
        #1 samples the prior only (the data are ignored, but the ODE is still solved for every draw)
        'prior_only':             0,
        'run_diagnostics':        1,
    }

