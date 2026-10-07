functions {
  int look_up_treatment(real t, array[] int treatment){

    // find out what integer timepoint n_days corresponds to
    int day = 1; //will since treatment[1] gives the first observation
    while (day <= t){
      day += 1;
    }
    
    // check if treatment is on
    int treatment_indicator = treatment[day];
    
    return treatment_indicator;
  } 
  
  vector dpop_dt(real t, vector state, array[] real theta, array[] int D){
                   
    // setup params for the ode
    real S  = state[1];
    real R  = state[2];
    real rS = theta[1];
    real dS = theta[2];
    real dD = theta[3];
    real rR = theta[4];
    real dR = theta[5];
    real K  = theta[6];

    //look up treatment at time t
    int D_t = look_up_treatment(t, D);
    
    vector[2] dy_dt;
    dy_dt[1] = rS * S * (1 - (S + R) / K) * (1 - dD * D_t) - dS * S;
    dy_dt[2] = rR * R * (1 - (S + R) / K) - dR * R;
    return dy_dt;
  }
}

data {
  int<lower=0> n_obs; // number of obs
  array[n_obs] real <lower=0> measurement_times; //#day of observation, must be > 0
  array[n_obs] real PSA_norm; //
  int<lower=1> n_days; // number of days in treatment indicator
  array[n_days] int <lower=0,upper=1> treatment_indicator; //one value per day 0
  
  real <lower=0> dD;
  
  //priors
    vector[2] cost_prior; // beta(a, b)
    vector[2] turnover_prior; // beta(a, b)
    vector[2] n0_prior; // beta(a, b)
    vector[2] rFrac_prior; // beta(a, b)
    vector[2] sigma_prior; //lognormal(mu, sigma)
    vector[2] K_prior; //lognormal(mu, sigma)
    vector[2] rS_prior; // lognormal(mu, sigma)
  
  int<lower=0, upper=1> run_diagnostics; //1 for log like and joint prior evalutation
}

parameters {
  real<lower=0, upper=1> cost;
  real<lower=0, upper=1> turnover;
  real<lower=0, upper=1> n0;    
  real<lower=0, upper=1> rFrac;
  real<lower=0> sigma;
  real <lower=0> rS;
  real <lower=n0> K;
}

transformed parameters{
  real rR = (1 - cost) * rS;
  real dS = turnover * rS;
  real dR = turnover * rS;
  
  //make prior over starting state prior
  vector[2] starting_state;
    starting_state[1] = (1 - rFrac) * n0;   // S0
    starting_state[2] = rFrac * n0; // R0
  
  //setup theta array for ODE
  array[6] real theta;
  theta[1] = rS;
  theta[2] = dS;
  theta[3] = dD;
  theta[4] = rR;
  theta[5] = dR;
  theta[6] = K;
  
  array[n_obs] vector[2] cell_populations = ode_rk45(dpop_dt, starting_state, 0, measurement_times, theta, treatment_indicator);
  
  // get sum of cell pops at each time and normalize by initial cell amount
  vector[n_obs] mu;
  for (i in 1:n_obs){
    mu[i] = sum(cell_populations[i]);
  }
}



model {
  //priors
  cost     ~ beta(cost_prior[1], cost_prior[2]);
  turnover ~ beta(turnover_prior[1], turnover_prior[2]);
  n0       ~ beta(n0_prior[1], n0_prior[2]);  
  rFrac    ~ beta(rFrac_prior[1], rFrac_prior[2]);
  sigma    ~ lognormal(sigma_prior[1], sigma_prior[2]);
  PSA_norm ~ normal(mu, sigma);
  K  ~ lognormal(K_prior[1], K_prior[2]);
  rS ~ lognormal(rS_prior[1], rS_prior[2]);
}

generated quantities{
  real lprior;
  vector[n_obs] log_lik;
  //if flag is on, then calc the log probability of each draw (all param values) given the prior. 
  //used for prior sensitivity analysis
  if (run_diagnostics == 1){
    lprior = beta_lpdf(cost|cost_prior[1], cost_prior[2]) +
              beta_lpdf(turnover|turnover_prior[1], turnover_prior[2]) +
              beta_lpdf(n0|n0_prior[1], n0_prior[2]) +
              beta_lpdf(rFrac|rFrac_prior[1], rFrac_prior[2]) +
              lognormal_lpdf(sigma|sigma_prior[1], sigma_prior[2])+
              lognormal_lpdf(K | K_prior[1], K_prior[2])+
              lognormal_lpdf(rS | rS_prior[1], rS_prior[2]);
    // for each observation compute the log likelihood of making that observation, used for model comp and sens analysis
    for (i in 1:n_obs){
      log_lik[i] = normal_lpdf(PSA_norm[i]| mu[i], sigma);
    }
  }
}

