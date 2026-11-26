functions {
  vector dpop_dt(real t, vector state, array[] real theta, data real K, int D){
    
    // setup params for the ode
    real S  = exp(state[1]);  //state is log(S), log(R)
    real R  = exp(state[2]);
    real rS = theta[1];
    real d  = theta[2]; 
    real dD = theta[3];
    real rR = theta[4];
    
    vector[2] dy_dt;
    dy_dt[1] = rS * (1 - (S + R) / K) * (1 - dD * D) - d;  //d log(S) / dt = (dS / dt) / S
    dy_dt[2] = rR * (1 - (S + R) / K) - d;
    return dy_dt;
  }
}

data {
  //treatment segment info. 
  int<lower=0> n_segments;
  array[n_segments] int<lower=0> starting_segment_times;
  array[n_segments] int<lower=0> end_segment_times;
  array[n_segments] int<lower=0> treatment_indicator;
  int<lower=1> n_out; // number of outputs is not the same as number of obs since some visits dont have PSA measurements
  array[n_out] real output_times;
  array[n_segments] int<lower=1, upper=n_out> segment_from_idx;
  array[n_segments] int<lower=1, upper=n_out> segment_to_idx;
  
  // actual data
  int<lower=0> n_obs; 
  array[n_obs] real PSA_norm; 
  array[n_obs] int<lower=1, upper=n_out> obs_idx; //which ode state each PSA reading is compared with

  //params for priors
  vector[2] cost_prior; // beta shape parameters
  vector[2] u_prior; // lognormal location and scale
  vector[2] n0_prior; // 
  vector[2] rFrac_prior; // beta shape parameters
  vector[2] sigma_prior; 
  vector[2] rS_prior;
  vector[2] dD_prior; 
  
  real<lower=0> K_fixed_val;  //carrying capacity, passed to the ODE as data, so it gets no sensitivities
  
  //prior only flag
  int<lower=0, upper=1> prior_only;
  
  
  //1 for log like and joint prior evalutation
  int<lower=0, upper=1> run_diagnostics;

}

parameters {
  real<lower=0, upper=1> cost;
  real<lower=0, upper=1> rFrac;
  real<lower=0> rS;
  real<lower=0, upper=1> rstar;  //u * n0
  real<lower=0, upper=rstar> n0;  //n0 < rstar is u > 1
  real<lower=0> dD;

  real<lower=0> sigma;
}

transformed parameters{
  //u = (1 - turnover / (1 - cost)) / n0: the plateau on continuous therapy relative to the starting size. u > 1.2 means progression
  real u = rstar / n0;
  real turnover = (1 - cost) * (1 - rstar);  //rstar = 1 - turnover / (1 - cost), the resistant-only plateau relative to K; no n0
  //baseline level on the data scale. When not fitted it equals n0, so the model is compared with the data unscaled


  real rR = (1 - cost) * rS;
  real dS = turnover * rS;
  real dR = turnover * rS;
  
  //make prior over starting state prior
  vector<lower=0>[2]  y0;
  y0[1] = (1 - rFrac) * n0;   // S0
  y0[2] = rFrac * n0; // R0
  
  //setup theta array for ODE
  array[4] real theta;
  theta[1] = rS;
  theta[2] = dS;
  theta[3] = dD;
  theta[4] = rR;
  
  //array to keep track of ODE states. This is needed since there are some rows with treatment switches but no PSA readings. 
  array[n_out] vector[2] ode_states;
  
  // go over the segments and calc ODE states for each visit. Even NA PSA measurements.
  for (i in 1:n_segments) {
    // get index of measurement times for current segment
    int first_out_time_idx = segment_from_idx[i];
    int last_out_time_idx = segment_to_idx[i];
    
    // get when time and state from segment starts from idx
    real t_start;
    vector[2] state_at_start;
    if (i == 1) {
      t_start = -1e-6;                          
      state_at_start = log(y0);
    } else {
      t_start = starting_segment_times[i];
      state_at_start = ode_states[first_out_time_idx - 1];    
    }
    
    int treatment = treatment_indicator[i];
    
    //solve and store ODE states in right indices
    ode_states[first_out_time_idx:last_out_time_idx] = ode_rk45(dpop_dt, state_at_start, t_start, output_times[first_out_time_idx:last_out_time_idx], theta, K_fixed_val, treatment);
  }
  
  // make new vector the size of PSA obs and get the corresponding ODE solutions
  array[n_obs] vector[2] likelihood_states = ode_states[obs_idx];
  
  //some times ODE returns very small but neg values. force these to zero. Claude sugggested reparametrizing to working in log(cells) instead
  vector[n_obs] mu;
  for (i in 1:n_obs){
    mu[i] = (exp(likelihood_states[i][1]) + exp(likelihood_states[i][2])) / n0;
  }
}

model {
  cost ~ beta(cost_prior[1], cost_prior[2]);
  //u's prior plus the Jacobian of u = rstar / n0 (n0 fixed): |du / drstar| = 1 / n0. The truncation at 1 is the bound n0 < rstar,
  //its normalising term lognormal_lccdf(1 | ...) is a constant and left out
  target += lognormal_lpdf(u | u_prior[1], u_prior[2]) - log(n0);
  n0 ~ beta(n0_prior[1], n0_prior[2]);
  rFrac ~ beta(rFrac_prior[1], rFrac_prior[2]);
  rS ~ lognormal(rS_prior[1], rS_prior[2]);
  dD ~ lognormal(dD_prior[1], dD_prior[2]);

  sigma ~ lognormal(sigma_prior[1], sigma_prior[2]);
  
  //likelihood if prior only is not passed. Multiplicative noise: log(PSA_norm) ~ normal(log(mu), sigma), so sigma is roughly
  //a relative error and PSA_norm must be positive. Matches LotkaVolterraModel.emit_observation in at-modelling
  if (!prior_only) PSA_norm ~ lognormal(log(mu), sigma);

}

generated quantities{
  //posterior predictive PSA: a new measurement at each observation time
  array[n_obs] real PSA_pred = lognormal_rng(log(mu), sigma);
  real lprior;
  vector[n_obs] log_lik;
  //if flag is on, get log liklihood and prior estimate of making draw
  if (run_diagnostics == 1){
    lprior = lognormal_lpdf(sigma|sigma_prior[1], sigma_prior[2]);
    lprior += beta_lpdf(cost|cost_prior[1], cost_prior[2]);
    lprior += lognormal_lpdf(u | u_prior[1], u_prior[2]) - lognormal_lccdf(1 | u_prior[1], u_prior[2]);  //truncated at 1
    lprior += beta_lpdf(n0|n0_prior[1], n0_prior[2]);
    lprior += beta_lpdf(rFrac|rFrac_prior[1], rFrac_prior[2]);
    lprior += lognormal_lpdf(rS | rS_prior[1], rS_prior[2]);
    lprior += lognormal_lpdf(dD | dD_prior[1], dD_prior[2]);
    // for each observation compute the log likelihood of making that observation, used for model comp and sens analysis
    for (i in 1:n_obs){
      log_lik[i] = lognormal_lpdf(PSA_norm[i] | log(mu[i]), sigma);
    }
  }
}

