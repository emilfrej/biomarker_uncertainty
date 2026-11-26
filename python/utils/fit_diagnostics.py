"""ArviZ diagnostics and pairs plots for one fit, on Stan's unconstrained scale, from the diagnostic CSVs fit_patient saves
(save_latent_dynamics=True). diagnose_fits runs them on every fit in a fit_routine.fit_all output folder"""
import json
import re
import shutil
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def unconstrained_datatree(csv_dir: Path, n_chains: int, n_post: int):
    """DataTree with the post-warmup draws of the sampled parameters on the unconstrained scale (posterior) and the sampler
    statistics (sample_stats). Each diagnostic CSV holds warmup then post-warmup rows (save_warmup=True), so the last n_post are kept"""
    files = sorted(csv_dir.glob('*-diagnostic_*.csv'))
    assert len(files) == n_chains, f'{csv_dir} has {len(files)} diagnostic CSVs, expected {n_chains}'  #more means runs are mixed
    chains = [pd.read_csv(f, comment='#').iloc[-n_post:] for f in files]
    max_depth = int(re.search(r'max_depth = (\d+)', files[0].read_text()).group(1))
    cols = list(chains[0].columns)
    params = cols[cols.index('energy__') + 1:]  #the parameters, then their momenta p_* and gradients g_*
    params = params[:len(params) // 3]
    stack = lambda c: np.stack([d[c].values for d in chains])  #(chain, draw)
    return az.from_dict({
        'posterior': {k: stack(k) for k in params},
        'sample_stats': {'diverging': stack('divergent__').astype(bool), 'tree_depth': stack('treedepth__'),
                         'reached_max_treedepth': stack('treedepth__') >= max_depth, 'energy': stack('energy__'),
                         'n_steps': stack('n_leapfrog__'), 'step_size': stack('stepsize__'), 'acceptance_rate': stack('accept_stat__'),
                         'lp': stack('lp__')}})


def diagnose_fit(dt) -> tuple[dict, pd.DataFrame]:
    """az.diagnose (divergences, max tree depth, E-BFMI, ESS, R-hat checks) summarised in one row, and az.summary's per-parameter
    diagnostics (ESS bulk / tail, R-hat, MCSE). R-hat and ESS are rank based, so they match the constrained scale's for parameters
    whose transform does not depend on other parameters"""
    has_errors, d = az.diagnose(dt, show_diagnostics=False, return_diagnostics=True)
    per_param = az.summary(dt, kind='diagnostics', round_to='none')
    row = {'has_errors': has_errors, 'divergences': d['divergent']['n_divergent'],
           'max_treedepth_hits': d.get('treedepth', {}).get('n_max'), 'min_bfmi': float(np.min(d['bfmi']['bfmi_values'])),
           'max_rhat': per_param['r_hat'].max(), 'min_ess_bulk': per_param['ess_bulk'].min(), 'min_ess_tail': per_param['ess_tail'].min(),
           'rhat_flagged': ' '.join(d['rhat']['bad_params'])}
    return row, per_param


def pairs_plot(dt, title: str = '', var_names: list | None = None, scale: str = 'unconstrained'):
    """az.plot_pair of dt's posterior (var_names, default all): lower triangle, divergent draws red. dt is unconstrained_datatree,
    or the fit's .nc for the constrained scale. Returns the PlotMatrix; save with .savefig"""
    with az.rc_context({'plot.max_subplots': 100}):  #7 parameters are 49 panels, the default limit is 40
        pc = az.plot_pair(dt, var_names=var_names, marginal_kind='hist', triangle='lower', figure_kwargs={'figsize': (13, 13)},
                          visuals={'divergence': {'color': 'red', 'size': 6}, 'scatter': {'alpha': 0.15, 'size': 1}})
    n_div = int(dt.sample_stats['diverging'].sum())
    pc.viz['figure'].item().suptitle(f'{title}: {n_div} divergent draws (red), post-warmup, {scale}')
    return pc


def diagnose_fits(fits_dir: Path, diagnose: bool = True, pairs: bool = True) -> pd.DataFrame | None:
    """Run on every fit in fits_dir/run_info.json (fit_routine.fit_all's output). diagnose: diagnose_fit summarised in a row per fit
    in fits_dir/diagnostics.csv (failed fits get their status only), per-parameter diagnostics in fits_dir/diagnostics/patient_<id>.csv.
    pairs: pairs_plots of the sampled parameters per fit in fits_dir/pairs/, patient_<id>_unconstrained.png (from the diagnostic CSVs)
    and patient_<id>_constrained.png (from the .nc). Each run deletes diagnostics/ and pairs/ first and redraws everything,
    so they always match the current fits and code. Returns the diagnostics table"""
    if diagnose: shutil.rmtree(fits_dir / 'diagnostics', ignore_errors=True)
    if pairs: shutil.rmtree(fits_dir / 'pairs', ignore_errors=True)
    rows = []
    for f in json.loads((fits_dir / 'run_info.json').read_text())['fits']:
        p = f['patient']
        row = {'patient': p, 'status': f['status']}
        if f['status'] == 'ok' and (diagnose or pairs):
            idata = az.from_netcdf(fits_dir / f'patient_{p}.nc')
            dt = unconstrained_datatree(fits_dir / f'patient_{p}_csv', idata.posterior.sizes['chain'], idata.posterior.sizes['draw'])
            if diagnose:
                d, per_param = diagnose_fit(dt)
                row |= d
                (fits_dir / 'diagnostics').mkdir(exist_ok=True)
                per_param.to_csv(fits_dir / 'diagnostics' / f'patient_{p}.csv')
            if pairs:
                (fits_dir / 'pairs').mkdir(exist_ok=True)
                for scale, data in [('unconstrained', dt), ('constrained', idata)]:
                    pc = pairs_plot(data, title=f'{fits_dir.name}, patient {p}', var_names=list(dt.posterior.data_vars), scale=scale)
                    pc.savefig(fits_dir / 'pairs' / f'patient_{p}_{scale}.png', dpi=100)
                    plt.close(pc.viz['figure'].item())
        rows.append(row)
    if diagnose:
        df = pd.DataFrame(rows)
        df.to_csv(fits_dir / 'diagnostics.csv', index=False)
        return df
