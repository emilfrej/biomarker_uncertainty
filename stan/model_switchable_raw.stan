//model_switchable_u.stan sampled in the raw ODE parameters instead of cost, turnover (u) and n0, rFrac:
//rS, rR, d (death rate of both populations, dS = dR as in the switchable models), dD, and the initial sizes S0, R0. K = 1.
//The support is the switchable_u model's: rR < rS (0 < cost < 1), S0 + R0 < K, and 0 < d < rR * (1 - n0), which is
//u > 1 (plateau on continuous therapy above the starting size) and u * n0 < 1 (turnover > 0).
//Priors are independent lognormals on the raw parameters, restricted to that support by the declared bounds.
//Multiplicative (lognormal) observation noise, as in model_switchable_u.stan.
functions {
  vector dpop_dt(real t, vector state, array[] real theta, int D){
    real S  = state[1];
    real R  = state[2];
    real rS = theta[1];
    real dS = theta[2];
    real dD = theta[3];
    real rR = theta[4];
    real dR = theta[5];
    real K  = theta[6];

    vector[2] dy_dt;
    dy_dt[1] = rS * S * (1 - (S + R) / K) * (1 - dD * D) - dS * S;
    dy_dt[2] = rR * R * (1 - (S + R) / K) - dR * R;
    return dy_dt;
  }
}

data {
  //treatment segments, as in model_switchable_u.stan (from utils.stan_data.make_stan_data)
  int<lower=0> n_segments;
  array[n_segments] int<lower=0> starting_segment_times;
  array[n_segments] int<lower=0> end_segment_times;
  array[n_segments] int<lower=0> treatment_indicator;
  int<lower=1> n_out;
  array[n_out] real output_times;
  array[n_segments] int<lower=1, upper=n_out> segment_from_idx;
  array[n_segments] int<lower=1, upper=n_out> segment_to_idx;

  int<lower=0> n_obs;
  array[n_obs] real PSA_norm;
  array[n_obs] int<lower=1, upper=n_out> obs_idx;

  //lognormal location and scale for each raw parameter
  vector[2] rS_prior;
  vector[2] rR_prior;
  vector[2] d_prior;
  vector[2] dD_prior;
  vector[2] S0_prior;
  vector[2] R0_prior;
  vector[2] sigma_prior;

  int<lower=0, upper=1> prior_only;
}

parameters {
  real<lower=0> rS;
  real<lower=0, upper=rS> rR;                      //cost = 1 - rR / rS in (0, 1)
  real<lower=0, upper=1> S0;
  real<lower=0, upper=1 - S0> R0;                  //n0 = S0 + R0 < K = 1
  real<lower=0, upper=rR * (1 - S0 - R0)> d;       //u > 1 and turnover > 0
  real<lower=0> dD;
  real<lower=0> sigma;
}

transformed parameters {
  real n0 = S0 + R0;
  //the switchable_u model's parameters, for comparison with its fits
  real cost = 1 - rR / rS;
  real turnover = d / rS;
  real rFrac = R0 / n0;
  real u = (1 - d / rR) / n0;

  vector[2] y0 = [S0, R0]';
  array[6] real theta = {rS, d, dD, rR, d, 1.0};

  array[n_out] vector[2] ode_states;
  for (i in 1:n_segments) {
    int first_out_time_idx = segment_from_idx[i];
    int last_out_time_idx = segment_to_idx[i];
    real t_start;
    vector[2] state_at_start;
    if (i == 1) {
      t_start = -1e-6;
      state_at_start = y0;
    } else {
      t_start = starting_segment_times[i];
      state_at_start = ode_states[first_out_time_idx - 1];
    }
    ode_states[first_out_time_idx:last_out_time_idx] = ode_rk45(dpop_dt, state_at_start, t_start, output_times[first_out_time_idx:last_out_time_idx], theta, treatment_indicator[i]);
  }

  array[n_obs] vector[2] likelihood_states = ode_states[obs_idx];
  vector[n_obs] mu;
  for (i in 1:n_obs){
    mu[i] = fmax(likelihood_states[i][1] / n0, 0) + fmax(likelihood_states[i][2] / n0, 0);
  }
}

model {
  rS ~ lognormal(rS_prior[1], rS_prior[2]);
  rR ~ lognormal(rR_prior[1], rR_prior[2]);
  d ~ lognormal(d_prior[1], d_prior[2]);
  dD ~ lognormal(dD_prior[1], dD_prior[2]);
  S0 ~ lognormal(S0_prior[1], S0_prior[2]);
  R0 ~ lognormal(R0_prior[1], R0_prior[2]);
  sigma ~ lognormal(sigma_prior[1], sigma_prior[2]);

  if (!prior_only) PSA_norm ~ lognormal(log(mu), sigma);
}

generated quantities {
  array[n_obs] real PSA_pred = lognormal_rng(log(mu), sigma);
  vector[n_obs] log_lik;
  for (i in 1:n_obs){
    log_lik[i] = lognormal_lpdf(PSA_norm[i] | log(mu[i]), sigma);
  }
}
