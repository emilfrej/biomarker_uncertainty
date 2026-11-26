"""Helpers for fitting model_switchable.stan from genertate_synth_AT_trial_data.ipynb"""
import numpy as np
import pandas as pd


def to_stan_prior(dist):
    """scipy beta(a, b) -> [a, b], scipy lognorm(s, scale) -> [log(scale), s] as in Stan's lognormal(mu, sigma),
    scipy norm(loc, scale) -> [loc, scale] as in Stan's normal(mu, sigma)"""
    if dist.dist.name == 'beta':
        return list(dist.args)
    if dist.dist.name == 'lognorm':
        return [float(np.log(dist.kwds['scale'])), dist.kwds['s']]
    if dist.dist.name == 'norm':
        return [dist.kwds['loc'], dist.kwds['scale']]
    raise ValueError(f"No Stan conversion for {dist.dist.name}")


def prior_init(priorDic, rng):
    """Inits for one chain, drawn from the scipy priors"""
    return {k: d.rvs(random_state=rng) for k, d in priorDic.items()}


def u_init(priorDic, rng):
    """prior_init for model_switchable_u.stan, redrawn until u is within its bounds: above 1 and below 1 / n0"""
    while True:
        init = prior_init(priorDic, rng)
        if 1 < init['u'] < 1 / init['n0']:
            return init


def rstar_init(priorDic, rng):
    """u_init's draw with rstar = u * n0 in place of u, for model_switchable_u_rstar.stan"""
    i = u_init(priorDic, rng)
    return {**{k: v for k, v in i.items() if k != 'u'}, 'rstar': i['u'] * i['n0']}


def turnover_init(priorDic, rng):
    """u_init's draw with turnover = (1 - cost)(1 - u * n0) in place of u, for model_switchable_u_turnover.stan"""
    i = u_init(priorDic, rng)
    return {**{k: v for k, v in i.items() if k != 'u'}, 'turnover': (1 - i['cost']) * (1 - i['u'] * i['n0'])}


def raw_init(priorDic, rng):
    """u_init's draw converted to model_switchable_raw_jac.stan's raw ODE parameters (rS, rR, d, S0, R0, dD, sigma)"""
    i = u_init(priorDic, rng)
    cost, u, n0, rFrac, rS = (i[k] for k in ['cost', 'u', 'n0', 'rFrac', 'rS'])
    return {'rS': rS, 'rR': (1 - cost) * rS, 'd': (1 - cost) * (1 - u * n0) * rS, 'S0': (1 - rFrac) * n0, 'R0': rFrac * n0,
            'dD': i['dD'], 'sigma': i['sigma']}


def rates_init(priorDic, rng):
    """u_init's draw converted to model_switchable_rates.stan's parameters: the net growth rates g_off, g_on, g_R, a = rS * n0, n0, rFrac, sigma"""
    i = u_init(priorDic, rng)
    cost, u, n0, rFrac, rS, dD = (i[k] for k in ['cost', 'u', 'n0', 'rFrac', 'rS', 'dD'])
    turnover = (1 - cost) * (1 - u * n0)
    return {'n0': n0, 'rFrac': rFrac, 'g_off': rS * (1 - turnover), 'g_on': rS * (1 - dD - turnover), 'g_R': rS * (1 - cost - turnover),
            'a': rS * n0, 'sigma': i['sigma']}


def make_stan_data(patients_df, patient_id, priors):
    """Stan data for model_switchable_u.stan (also used by the raw_jac and rates models, which ignore the entries they do not declare),
    from one patient's rows (columns PatientId, Time, DrugConcentration, Observation).
    K is fixed at 1"""
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
        'K_fixed_val':            1,
        **priors,
        #1 samples the prior only
        'prior_only':             0,
        'run_diagnostics':        1,
    }

