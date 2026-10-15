// same model as model.stan, but with cost, rS and R0 fixed to literature values (Table S2 of the biomarker paper)
// and the initial tumour density N0 set to the first PSA measurement instead of being fitted.
// only turnover, K and sigma are estimated
functions {
  vector dpop_dt(real t, vector state, array[] real theta, int D){

    // setup params for the ode
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
  //segments
  int<lower=0> n_segments;
  array[n_segments] int<lower=0> starting_segment_times;
  array[n_segments] int<lower=0> end_segment_times;
  array[n_segments] int<lower=0> treatment_indicator;
  int<lower=1> n_out;
  array[n_out] real output_times;
  array[n_segments] int<lower=1, upper=n_out> segment_from_idx;
  array[n_segments] int<lower=1, upper=n_out> segment_to_idx;


  int<lower=0> n_obs; // number of obs
  array[n_obs] real PSA_norm; //
  array[n_obs] int<lower=1, upper=n_out> obs_idx; //which ode state each PSA reading is compared with

  real <lower=0> dD;

  //values fixed from the literature
  real<lower=0, upper=1> cost_fixed; // 0.1
  real<lower=0> rS_fixed;            // 0.027 per day
  real<lower=0> R0_fixed;            // 0.001, initial resistant density

  //priors
    vector[2] turnover_prior; // beta(a, b)
    vector[2] sigma_prior; //lognormal(mu, sigma)
    vector[2] K_prior; //lognormal(mu, sigma)

  int<lower=0, upper=1> run_diagnostics; //1 for log like and joint prior evalutation
}

transformed data {
  //initial tumour density is the first PSA measurement (1 after normalising)
  real n0_fixed = PSA_norm[1];
}

parameters {
  real<lower=0, upper=1> turnover;
  real<lower=0> sigma;
  real <lower=n0_fixed> K;
}

transformed parameters{
  //fixed values, reported as draws so downstream code can read them like in model.stan
  real cost  = cost_fixed;
  real rS    = rS_fixed;
  real n0    = n0_fixed;
  real rFrac = R0_fixed / n0_fixed;

  real rR = (1 - cost) * rS;
  real dS = turnover * rS;
  real dR = turnover * rS;

  //starting state from the first PSA measurement and the fixed resistant density
  vector<lower=0>[2]  y0;
    y0[1] = n0 - R0_fixed;   // S0
    y0[2] = R0_fixed;        // R0

  //setup theta array for ODE
  array[6] real theta;
  theta[1] = rS;
  theta[2] = dS;
  theta[3] = dD;
  theta[4] = rR;
  theta[5] = dR;
  theta[6] = K;

//array to keep track of ODE states. This is needed since there are some rows with treatment switches but no PSA readings.
  array[n_out] vector[2] ode_states;

  for (i in 1:n_segments) {
    // get index of measurement times for current segment
    int first_out_time_idx = segment_from_idx[i];
    int last_out_time_idx = segment_to_idx[i];

    // get when time and state from segment starts from idx
    real t_start;
    vector[2] state_at_start;
    if (i == 1) {
      t_start = -1e-6;
      state_at_start = y0;
    } else {
      t_start = starting_segment_times[i];
      state_at_start = ode_states[first_out_time_idx - 1];
    }

   int treatment = treatment_indicator[i];

    //solve and store ODE states in right indices
   ode_states[first_out_time_idx:last_out_time_idx] = ode_rk45(dpop_dt, state_at_start, t_start, output_times[first_out_time_idx:last_out_time_idx], theta, treatment);
  }

  // make new vector the size of PSA obs and get the corresponding ODE solutions
  array[n_obs] vector[2] likelihood_states = ode_states[obs_idx];

  //some times ODE returns very small but neg values. force these to zero
  vector[n_obs] mu;
  for (i in 1:n_obs){
    mu[i] = fmax(likelihood_states[i][1], 0) + fmax(likelihood_states[i][2], 0);
  }
}

model {
  //priors
  turnover ~ beta(turnover_prior[1], turnover_prior[2]);
  sigma    ~ lognormal(sigma_prior[1], sigma_prior[2]);
  K  ~ lognormal(K_prior[1], K_prior[2]);

  //likelihood
  PSA_norm ~ normal(mu, sigma);
}

generated quantities{
  real lprior;
  vector[n_obs] log_lik;
  //if flag is on, then calc the log probability of each draw (all param values) given the prior.
  //used for prior sensitivity analysis
  if (run_diagnostics == 1){
    lprior = beta_lpdf(turnover|turnover_prior[1], turnover_prior[2]) +
              lognormal_lpdf(sigma|sigma_prior[1], sigma_prior[2])+
              lognormal_lpdf(K | K_prior[1], K_prior[2]);
    // for each observation compute the log likelihood of making that observation, used for model comp and sens analysis
    for (i in 1:n_obs){
      log_lik[i] = normal_lpdf(PSA_norm[i]| mu[i], sigma);
    }
  }
}
