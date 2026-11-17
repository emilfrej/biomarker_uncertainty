"""Helpers for generating and plotting the synthetic AT patients in genertate_synth_AT_trial_data.ipynb"""
import pandas as pd
import matplotlib.pyplot as plt


def draw_params(priorDic : dict):
    """Contstruct a dictionary of sampled values from prior dict"""
    return {paramName: dist.rvs() for paramName, dist in priorDic.items()}


def complete_param_dict(cost, turnover, n0, rFrac, rS, K, dD, sigma):
    """Pass unpacked dict to generate param dict in
    form required by model class."""
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
