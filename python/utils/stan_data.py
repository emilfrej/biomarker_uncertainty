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


def u_init(priorDic, rng):
    """prior_init for model_switchable_u.stan, redrawn until u is within its bounds: above 1 and below 1 / n0"""
    while True:
        init = prior_init(priorDic, rng)
        if 1 < init['u_free'][0] < 1 / init['n0_free'][0]:
            return init


def make_stan_data(patients_df, patient_id, priors):
    """Stan data for model_switchable.stan, or model_switchable_u.stan when priors has u_prior in place of turnover_prior,
    from one patient's rows (columns PatientId, Time, DrugConcentration, Observation).
    Every parameter with a <name>_prior in priors is fitted; K is fixed at 1"""
    fitted = [k.removesuffix('_prior') for k in priors if k != 'sigma_prior']  # sigma is always fitted, it has no flag
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
        #fix param flags: fit the parameters with a prior (turnover or u), K fixed at 1
        **{f'fit_{p}': 1 for p in fitted}, **{f'{p}_fixed_val': 0 for p in fitted},
        'fit_K': 0, 'K_fixed_val': 1,
        **priors,
        'K_prior':                [1, 1],
        #1 samples the prior only
        'prior_only':             0,
        'run_diagnostics':        1,
    }

