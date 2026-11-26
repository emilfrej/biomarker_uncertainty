//model_switchable_u.stan, same model and priors, sampled in the net growth rates the data inform, far from K:
//g_off = rS (1 - turnover) (sensitive, off treatment), g_on = rS (1 - dD - turnover) (sensitive, on treatment),
//g_R = rS (1 - cost - turnover) (resistant), plus a = rS * n0 (sets the slowdown near K per unit of relative size), n0, rFrac.
//Back to the ODE: rS = a / n0, d = rS - g_off (dS = dR = d), rR = d + g_R, dD = (g_off - g_on) / rS. K = 1.
//The support is the switchable_u model's: 0 < g_R < g_off (0 < cost < 1), g_on < g_off (dD > 0),
//g_off n0 < a (turnover > 0) and a < g_off n0 + g_R (1 - n0) (u > 1).
//Priors are the switchable_u model's priors on cost, u, n0, rFrac, rS, dD, sigma, evaluated at the transformed values,
//plus the Jacobian of (n0, rFrac, a, g_off, g_R, g_on) -> (rS, cost, n0, rFrac, u, dD), so the posterior is the switchable_u model's
//and only the sampler's coordinates change. Takes the same data as model_switchable_u.stan (utils.stan_data.make_stan_data).
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

  //the switchable_u model's priors
  vector[2] cost_prior;   //beta shapes
  vector[2] u_prior;      //lognormal location and scale, truncated below at 1
  vector[2] n0_prior;     //beta shapes
  vector[2] rFrac_prior;  //beta shapes
  vector[2] rS_prior;     //lognormal
  vector[2] dD_prior;     //lognormal
  vector[2] sigma_prior;  //lognormal

  int<lower=0, upper=1> prior_only;
}

parameters {
  real<lower=0, upper=1> n0;
  real<lower=0, upper=1> rFrac;
  real<lower=0> g_off;
  real<lower=0, upper=g_off> g_R;                                  //0 < cost < 1
  real<lower=g_off * n0, upper=g_off * n0 + g_R * (1 - n0)> a;     //a = rS * n0; turnover > 0 and u > 1
  real<upper=g_off> g_on;                                          //dD > 0
  real<lower=0> sigma;
}

transformed parameters {
  //raw ODE parameters
  real rS = a / n0;
  real d = rS - g_off;
  real rR = d + g_R;
  real dD = (g_off - g_on) / rS;
  real S0 = (1 - rFrac) * n0;
  real R0 = rFrac * n0;
  //the switchable_u model's parameters, for comparison with its fits
  real cost = 1 - rR / rS;
  real turnover = d / rS;
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
  //priors on the switchable parameters; the truncation of u at 1 is the bound on d, its normalising term is a constant
  target += beta_lpdf(cost | cost_prior[1], cost_prior[2]);
  target += lognormal_lpdf(u | u_prior[1], u_prior[2]);
  target += beta_lpdf(n0 | n0_prior[1], n0_prior[2]);
  target += beta_lpdf(rFrac | rFrac_prior[1], rFrac_prior[2]);
  target += lognormal_lpdf(rS | rS_prior[1], rS_prior[2]);
  target += lognormal_lpdf(dD | dD_prior[1], dD_prior[2]);
  sigma ~ lognormal(sigma_prior[1], sigma_prior[2]);
  //log |det d(rS, cost, n0, rFrac, u, dD) / d(n0, rFrac, a, g_off, g_R, g_on)|: the rates -> raw ODE map has determinant 1 / rS
  //(n0 * 1/n0 * 1 * 1 * 1/rS), the raw -> switchable map 1 / (rS * rR * n0^2) (model_switchable_raw_jac.stan)
  target += -2 * log(rS) - log(rR) - 2 * log(n0);

  if (!prior_only) PSA_norm ~ lognormal(log(mu), sigma);
}

generated quantities {
  array[n_obs] real PSA_pred = lognormal_rng(log(mu), sigma);
  vector[n_obs] log_lik;
  for (i in 1:n_obs){
    log_lik[i] = lognormal_lpdf(PSA_norm[i] | log(mu[i]), sigma);
  }
}
