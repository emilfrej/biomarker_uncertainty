// hierarchical version of model_switchable.stan for a cohort of patients. K is fixed at 1
// each patient's parameters on the unconstrained scale (logit cost, logit turnover, logit n0, logit rFrac, log rS, log dD)
// are multivariate normal around a population mean. Parameters can be fixed for every patient (free_idx, fixed_unc)
// non-centred: theta = mu + diag(tau) * L_Omega * z, with z standard normal
// the data are PSA / first PSA, compared with (S + R) / n0. Noise is normal or lognormal (lognormal_noise)
functions {
  vector dpop_dt(real t, vector y, real rS, real dS, real dD, real rR, real dR, int D) {
    vector[2] dy_dt;
    real N = y[1] + y[2];
    dy_dt[1] = rS * y[1] * (1 - N) * (1 - dD * D) - dS * y[1];
    dy_dt[2] = rR * y[2] * (1 - N) - dR * y[2];
    return dy_dt;
  }
}

data {
  int<lower=1> n_patients;

  //constant dose segments of all patients, one after the other. Output indices are within the patient's output times
  int<lower=1> n_segments;
  array[n_segments] real seg_start_time;
  array[n_segments] int<lower=0, upper=1> seg_treatment;
  array[n_segments] int<lower=1> seg_from_idx;
  array[n_segments] int<lower=1> seg_to_idx;
  array[n_patients] int<lower=1> pat_seg_first;
  array[n_patients] int<lower=1> pat_seg_last;

  //times the ODE is solved at, all patients one after the other
  int<lower=1> n_out;
  array[n_out] real output_times;
  array[n_patients] int<lower=1> pat_out_first;
  array[n_patients] int<lower=1> pat_out_last;

  //observations, all patients one after the other. obs_idx is within the patient's output times
  int<lower=1> n_obs;
  vector<lower=0>[n_obs] psa;
  array[n_obs] int<lower=1> obs_idx;
  array[n_patients] int<lower=1> pat_obs_first;
  array[n_patients] int<lower=1> pat_obs_last;

  int<lower=0, upper=1> lognormal_noise;

  //which of the 6 parameters vary between patients. The rest are fixed at fixed_unc (unconstrained scale)
  int<lower=1, upper=6> n_free;
  array[n_free] int<lower=1, upper=6> free_idx;
  vector[6] fixed_unc;

  //priors: mu ~ normal(mu_loc, mu_scale), tau ~ half-normal(0, tau_scale), L_Omega ~ lkj(lkj_eta), sigma ~ half-normal(0, sigma_scale)
  vector[n_free] mu_loc;
  vector<lower=0>[n_free] mu_scale;
  vector<lower=0>[n_free] tau_scale;
  real<lower=0> lkj_eta;
  real<lower=0> sigma_scale;
}

parameters {
  vector[n_free] mu;
  vector<lower=0>[n_free] tau;
  cholesky_factor_corr[n_free] L_Omega;
  matrix[n_free, n_patients] z;
  real<lower=0> sigma;
}

transformed parameters {
  matrix[n_free, n_patients] theta_free = rep_matrix(mu, n_patients) + diag_pre_multiply(tau, L_Omega) * z;
  matrix[6, n_patients] pars; //natural scale: cost, turnover, n0, rFrac, rS, dD
  vector[n_obs] mu_psa;

  for (p in 1:n_patients) {
    vector[6] u = fixed_unc;
    u[free_idx] = theta_free[:, p];
    real cost     = inv_logit(u[1]);
    real turnover = inv_logit(u[2]);
    real n0       = inv_logit(u[3]);
    real rFrac    = inv_logit(u[4]);
    real rS       = exp(u[5]);
    real dD       = exp(u[6]);
    pars[:, p] = [cost, turnover, n0, rFrac, rS, dD]';

    real rR = (1 - cost) * rS;
    real dS = turnover * rS;
    real dR = turnover * rS;

    //solve segment by segment. The first starts just before day 0, so the first visit is an output time
    int out_offset = pat_out_first[p] - 1;
    array[pat_out_last[p] - out_offset] vector[2] states;
    for (s in pat_seg_first[p]:pat_seg_last[p]) {
      real t_start;
      vector[2] state_at_start;
      if (s == pat_seg_first[p]) {
        t_start = -1e-6;
        state_at_start = [(1 - rFrac) * n0, rFrac * n0]';
      } else {
        t_start = seg_start_time[s];
        state_at_start = states[seg_from_idx[s] - 1];
      }
      states[seg_from_idx[s]:seg_to_idx[s]] = ode_rk45(dpop_dt, state_at_start, t_start,
                                                       output_times[(out_offset + seg_from_idx[s]):(out_offset + seg_to_idx[s])],
                                                       rS, dS, dD, rR, dR, seg_treatment[s]);
    }
    for (o in pat_obs_first[p]:pat_obs_last[p]) {
      vector[2] state = states[obs_idx[o]];
      //the solver can return tiny negative values. The floor keeps log(mu_psa) finite for lognormal noise
      mu_psa[o] = fmax(fmax(state[1], 0) + fmax(state[2], 0), 1e-12) / n0;
    }
  }
}

model {
  mu ~ normal(mu_loc, mu_scale);
  tau ~ normal(0, tau_scale);
  L_Omega ~ lkj_corr_cholesky(lkj_eta);
  to_vector(z) ~ std_normal();
  sigma ~ normal(0, sigma_scale);

  if (lognormal_noise) psa ~ lognormal(log(mu_psa), sigma);
  else psa ~ normal(mu_psa, sigma);
}

generated quantities {
  matrix[n_free, n_free] Omega = multiply_lower_tri_self_transpose(L_Omega);
  //on the psa scale for both noise models (lognormal_lpdf includes the Jacobian), so the two can be compared with LOO
  vector[n_obs] log_lik;
  for (i in 1:n_obs) {
    log_lik[i] = lognormal_noise ? lognormal_lpdf(psa[i] | log(mu_psa[i]), sigma) : normal_lpdf(psa[i] | mu_psa[i], sigma);
  }
}
