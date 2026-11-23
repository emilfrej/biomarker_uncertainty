"""Helpers for generating and plotting the synthetic AT patients in genertate_synth_AT_trial_data.ipynb"""
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from at_modelling.LotkaVolterraModel import LotkaVolterraModel


def draw_params(priorDic : dict):
    """Contstruct a dictionary of sampled values from prior dict"""
    return {paramName: dist.rvs() for paramName, dist in priorDic.items()}


def complete_param_dict(cost, u, n0, rFrac, rS, dD, sigma, K=1):
    """Pass unpacked dict to generate param dict in
    form required by model class. turnover is computed from u = (1 - turnover / (1 - cost)) / n0"""
    turnover = (1 - cost) * (1 - u * n0)
    return {"Cost": cost, "Turnover": turnover,
                "n0": n0, "R0": rFrac * n0, "S0": (n0 - rFrac*n0),
                "rS": rS, "rR": rS * (1-cost),
                "dS": rS * turnover, "dR": rS * turnover,
                'K': K, 'dD': dD, 'sigma' : sigma}


#steady state check
def progress_on_ct(paramDic):
    """Return true if the params would even progress on continous therapy using the derivation from Kit Gallagher'
    s DRL paper."""
    progressCTflag = paramDic['Turnover'] < (1 - 1.2 * (paramDic['S0']+paramDic['R0'])/paramDic['K']) * (1 - paramDic['Cost'])
    return(progressCTflag)


def add_cycle_col(Df, treatment_col):
    '''Function to add cycle identifier'''
    Df['Cycle'] = Df[treatment_col].ne(Df[treatment_col].shift()).cumsum()
    return Df


def generate_u_patients(priorDic, n_patients, out_dir, seed=1, t_end=5000, check_progression=True):
    """Synthetic patients for model_switchable_u.stan: draw from priorDic (u in place of turnover), simulate AT and keep
    draws that progress on CT and on AT. Writes synth_patients.csv and true_params.csv to out_dir, which is deleted first.
    t_end: ensure progression occurs in this period"""
    np.random.seed(seed)  # draw_params and the simulator's observation noise use the global numpy RNG
    shutil.rmtree(out_dir, ignore_errors=True) #delete if exists
    out_dir.mkdir(parents=True)

    patients, true_params = [], []
    for i in range(n_patients):
        #run loop until viable param combination is found
        while True:
            priorDrawDic = draw_params(priorDic)
            priorDrawDic['K'] = 1 #hard code carrying cap
            #redraw outside model_switchable_u.stan's bounds: u truncated at 1 (T[1, ]) and u * n0 < K
            if priorDrawDic['u'] < 1 or priorDrawDic['u'] * priorDrawDic['n0'] >= priorDrawDic['K']:
                continue # u
            paramDic = complete_param_dict(**priorDrawDic)
            priorDrawDic['turnover'] = paramDic['Turnover']  #saved in true_params.csv
            #if not progress_on_ct(paramDic=paramDic):
            #    print("Analytically doesnt progress CT")
            #    continue
            model = LotkaVolterraModel(method='RK45', dt=1)
            model.set_params(**{k: v for k, v in paramDic.items() if k not in ("Cost", "Turnover", "n0")})
            model.simulate_at(at_threshold=0.5, interval_length=28, t_end=t_end)
            if not np.isnan(model.ttp) or not check_progression:
                print(f"Check passed or not checking progression on AT! Saving Patient {i} out of {n_patients}")
                break
            print("Check failed")
        #break out while loop and store found param combination. Observation is NaN except at the check-ups
        patients.append(add_cycle_col(model.results_df(), 'DrugConcentration').assign(PatientId=i))
        true_params.append({**priorDrawDic, "PatientId": i})
    synth_df, true_params_df = pd.concat(patients, ignore_index=True), pd.DataFrame(true_params)
    synth_df.to_csv(out_dir / 'synth_patients.csv', index=False)
    true_params_df.to_csv(out_dir / 'true_params.csv', index=False)
    return synth_df, true_params_df


def plot_n_rand_patients(synth_df: pd.DataFrame, n: int = 6, seed: int | None = None, ncols: int = 3) -> plt.Figure:
    """Plot n random patients: true size / initial size (line), noisy observations (points),
    treatment (shaded) and the progression threshold (dashed)"""
    ids = synth_df['PatientId'].drop_duplicates().sample(n, random_state=seed).sort_values()
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), sharey=True, squeeze=False)
    for ax, pid in zip(axes.flat, ids):
        df = synth_df[synth_df['PatientId'] == pid].sort_values('Time')
        # DrugConcentration is the dose from each time on, so the shading steps after each row
        ax.fill_between(df['Time'], 0, 1, where=df['DrugConcentration'] > 0, step='post',
                        transform=ax.get_xaxis_transform(), color='grey', alpha=0.2, lw=0)
        ax.plot(df['Time'], df['TumourSize'] / df['TumourSize'].iloc[0], color='black', lw=1)
        obs = df.dropna(subset=['Observation'])
        ax.plot(obs['Time'], obs['Observation'], 'o', ms=3, color='tab:blue')
        ax.axhline(1.2, ls='--', lw=0.8, color='tab:red')
        ax.set_title(f'Patient {pid}')
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel('Time (days)')
    fig.supylabel('Size / initial size')
    fig.tight_layout()
    return fig
