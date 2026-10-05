// copy of prototype_model.stan with a switch-time treatment lookup (faster than the per-day loop). used by sbc_single_patient.Rmd
functions {
  int look_up_treatment(real t, array[] real switch_times, array[] int treatment){

    // start with the first treatment value, then move to each later switch that t has passed
    int treatment_indicator = treatment[1];
    for (i in 1:size(switch_times)){
      if (t >= switch_times[i]) treatment_indicator = treatment[i];
    }

    return treatment_indicator;
  } 
  
  vector dpop_dt(real t, vector state, array[] real theta, array[] real switch_times, array[] int D){
                   
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
    int D_t = look_up_treatment(t, switch_times, D);
    

    vector[2] dy_dt;
    dy_dt[1] = rS * S * (1 - (S + R) / K) * (1 - dD * D_t) - dS * S;
    dy_dt[2] = rR * R * (1 - (S + R) / K) - dR * R;
    return dy_dt;
  }
}

data {
  int<lower=0> N; // number of obs
  array[N] real <lower=0> measurement_times; //#day of observation, must be > 0
  array[N] real PSA_norm; //
  int<lower=1> n_switches; // number of treatment segments
  array[n_switches] real <lower=0> switch_times; //day each segment starts
  array[n_switches] int <lower=0,upper=1> treatment_indicator; //treatment in each segment
  
  //fixed params
  real <lower=0, upper=1> rS;
  real <lower=0, upper=1> K;
  real <lower=0> dD;
}

parameters {
  real<lower=0, upper=1> cost;
  real<lower=0, upper=1> turnover;
  real<lower=0, upper=1> n0;    
  real<lower=0, upper=1> rFrac;
  real<lower=0> sigma;
}

transformed parameters{
  real rR = (1 - cost) * rS;
  real dS = turnover * rS;
  real dR = turnover * rS;
  
  //make prior over starting state prior
  vector[2] starting_state;
    starting_state[1] = (1 - rFrac) * n0 * K;   // S0
    starting_state[2] = rFrac * n0 * K; // R0
  
  //setup theta array for ODE
  array[6] real theta;
  theta[1] = rS;
  theta[2] = dS;
  theta[3] = dD;
  theta[4] = rR;
  theta[5] = dR;
  theta[6] = K;
  
  array[N] vector[2] cell_populations = ode_rk45(dpop_dt, starting_state, 0, measurement_times, theta, switch_times, treatment_indicator);
  
  // get sum of cell pops at each time and normalize by initial cell amount
  vector[N] mu;
  for (n in 1:N) mu[n] = sum(cell_populations[n]) / (n0 * K);
}

model {
  //priors
  cost     ~ beta(1, 1);
  turnover ~ beta(2, 5);
  n0       ~ beta(1, 1);  
  rFrac    ~ beta(1, 30);
  sigma    ~ lognormal(-2, 1);
  
  PSA_norm ~ normal(mu, sigma);
}

