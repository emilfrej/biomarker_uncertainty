//Hierarchical version of model_switchable_u.stan for a cohort of patients, with correlated patient parameters.
//Patient j's parameters on the unconstrained scale, eta_j = (logit cost, logit n0, logit((u - 1) / (1 / n0 - 1)), logit rFrac, log rS, log dD),
//are multivariate normal: eta_j = mu_pop + diag(tau) * L * z_j (non-centred), L the Cholesky factor of their correlation matrix.
//u's transform keeps 1 < u < 1 / n0 as in model_switchable_u.stan. K = 1. Lognormal (multiplicative) noise with a per-patient sigma,
//log sigma_j ~ normal(mu_lsig, tau_lsig). Patients are stacked: patient j's segments, output times and observations are the
//slices seg_start[j]:seg_end[j], out_start[j]:out_end[j] and obs_start[j]:obs_end[j]; the indices inside a slice
//(segment_from_idx, segment_to_idx, obs_idx) are local to the patient, as make_stan_data gives them.
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

  //one patient's (cost, n0, u, rFrac, rS, dD) from its unconstrained eta
  array[] real constrain(vector eta) {
    real n0 = inv_logit(eta[2]);
    return {inv_logit(eta[1]), n0, 1 + (1 / n0 - 1) * inv_logit(eta[3]), inv_logit(eta[4]), exp(eta[5]), exp(eta[6])};
  }

  //one patient's mean PSA_norm at its observations; the same ODE and segment loop as model_switchable_u.stan
  vector patient_mu(array[] real p, array[] int seg_start_t, array[] int treat, array[] int seg_from, array[] int seg_to,
                    array[] real out_t, array[] int obs_idx) {
    real cost = p[1];
    real n0 = p[2];
    real u = p[3];
    real rFrac = p[4];
    real rS = p[5];
    real dD = p[6];
    real turnover = (1 - cost) * (1 - u * n0);
    array[6] real theta = {rS, turnover * rS, dD, (1 - cost) * rS, turnover * rS, 1.0};

    array[size(out_t)] vector[2] states;
    for (i in 1:size(treat)) {
      real t_start;
      vector[2] state_at_start;
      if (i == 1) {
        t_start = -1e-6;
        state_at_start = [(1 - rFrac) * n0, rFrac * n0]';
      } else {
        t_start = seg_start_t[i];
        state_at_start = states[seg_from[i] - 1];
      }
      states[seg_from[i]:seg_to[i]] = ode_rk45(dpop_dt, state_at_start, t_start, out_t[seg_from[i]:seg_to[i]], theta, treat[i]);
    }

    vector[size(obs_idx)] mu;
    for (k in 1:size(obs_idx)) mu[k] = fmax(states[obs_idx[k]][1] / n0, 0) + fmax(states[obs_idx[k]][2] / n0, 0);
    return mu;
  }

  //log likelihood of patients start:end, for reduce_sum
  real partial_loglik(array[] int pid_slice, int start, int end, array[,] real p, vector sigma,
                      array[] int seg_start, array[] int seg_end, array[] int out_start, array[] int out_end,
                      array[] int obs_start, array[] int obs_end, array[] int starting_segment_times, array[] int treatment_indicator,
                      array[] int segment_from_idx, array[] int segment_to_idx, array[] real output_times, array[] int obs_idx,
                      array[] real PSA_norm) {
    real lp = 0;
    for (j in start:end) {
      vector[obs_end[j] - obs_start[j] + 1] mu = patient_mu(p[j], starting_segment_times[seg_start[j]:seg_end[j]],
          treatment_indicator[seg_start[j]:seg_end[j]], segment_from_idx[seg_start[j]:seg_end[j]], segment_to_idx[seg_start[j]:seg_end[j]],
          output_times[out_start[j]:out_end[j]], obs_idx[obs_start[j]:obs_end[j]]);
      lp += lognormal_lpdf(PSA_norm[obs_start[j]:obs_end[j]] | log(mu), sigma[j]);
    }
    return lp;
  }
}

data {
  int<lower=1> J;  //patients
  int<lower=1> n_segments;
  int<lower=1> n_out;
  int<lower=1> n_obs;
  array[J] int<lower=1, upper=n_segments> seg_start;
  array[J] int<lower=1, upper=n_segments> seg_end;
  array[J] int<lower=1, upper=n_out> out_start;
  array[J] int<lower=1, upper=n_out> out_end;
  array[J] int<lower=1, upper=n_obs> obs_start;
  array[J] int<lower=1, upper=n_obs> obs_end;

  //stacked per-patient data, indices local to the patient
  array[n_segments] int<lower=0> starting_segment_times;
  array[n_segments] int<lower=0> treatment_indicator;
  array[n_segments] int<lower=1> segment_from_idx;
  array[n_segments] int<lower=1> segment_to_idx;
  array[n_out] real output_times;
  array[n_obs] int<lower=1> obs_idx;
  array[n_obs] real<lower=0> PSA_norm;

  //hyperpriors: mu_pop ~ normal(mu_pop_loc, mu_pop_scale), tau ~ half-normal(0, tau_scale), correlation ~ LKJ(lkj_eta)
  vector[6] mu_pop_loc;
  vector<lower=0>[6] mu_pop_scale;
  vector<lower=0>[6] tau_scale;
  real<lower=0> lkj_eta;
  vector[2] mu_lsig_prior;  //normal location, scale of the population mean of log sigma
  real<lower=0> tau_lsig_scale;  //half-normal scale of the population sd of log sigma

  int<lower=0, upper=1> prior_only;
  int<lower=1> grainsize;  //reduce_sum grainsize, 1 lets the scheduler choose
}

transformed data {
  array[J] int pid;
  for (j in 1:J) pid[j] = j;
}

parameters {
  vector[6] mu_pop;
  vector<lower=0>[6] tau;
  cholesky_factor_corr[6] L;
  matrix[6, J] z;
  real mu_lsig;
  real<lower=0> tau_lsig;
  vector[J] z_lsig;
}

transformed parameters {
  matrix[6, J] eta = rep_matrix(mu_pop, J) + diag_pre_multiply(tau, L) * z;
  vector[J] sigma = exp(mu_lsig + tau_lsig * z_lsig);
  array[J, 6] real p;  //patient j's (cost, n0, u, rFrac, rS, dD)
  for (j in 1:J) p[j] = constrain(eta[, j]);
}

model {
  mu_pop ~ normal(mu_pop_loc, mu_pop_scale);
  tau ~ normal(0, tau_scale);
  L ~ lkj_corr_cholesky(lkj_eta);
  to_vector(z) ~ std_normal();
  mu_lsig ~ normal(mu_lsig_prior[1], mu_lsig_prior[2]);
  tau_lsig ~ normal(0, tau_lsig_scale);
  z_lsig ~ std_normal();

  if (!prior_only) target += reduce_sum(partial_loglik, pid, grainsize, p, sigma, seg_start, seg_end, out_start, out_end,
                                        obs_start, obs_end, starting_segment_times, treatment_indicator, segment_from_idx,
                                        segment_to_idx, output_times, obs_idx, PSA_norm);
}

generated quantities {
  corr_matrix[6] Omega = multiply_lower_tri_self_transpose(L);
  vector[J] cost;
  vector[J] n0;
  vector[J] u;
  vector[J] rFrac;
  vector[J] rS;
  vector[J] dD;
  vector[n_obs] mu;
  array[n_obs] real PSA_pred;  //posterior predictive: a new measurement at each observation
  vector[n_obs] log_lik;
  for (j in 1:J) {
    cost[j] = p[j, 1]; n0[j] = p[j, 2]; u[j] = p[j, 3]; rFrac[j] = p[j, 4]; rS[j] = p[j, 5]; dD[j] = p[j, 6];
    mu[obs_start[j]:obs_end[j]] = patient_mu(p[j], starting_segment_times[seg_start[j]:seg_end[j]], treatment_indicator[seg_start[j]:seg_end[j]],
        segment_from_idx[seg_start[j]:seg_end[j]], segment_to_idx[seg_start[j]:seg_end[j]], output_times[out_start[j]:out_end[j]],
        obs_idx[obs_start[j]:obs_end[j]]);
    for (i in obs_start[j]:obs_end[j]) {
      PSA_pred[i] = lognormal_rng(log(mu[i]), sigma[j]);
      log_lik[i] = lognormal_lpdf(PSA_norm[i] | log(mu[i]), sigma[j]);
    }
  }
}
