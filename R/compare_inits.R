#compares three ways of starting the chains for stan/model_switchable.stan on a few synthetic patients:
#Stan's default inits (uniform(-2, 2) on the unconstrained scale), draws from the prior, and draws from a pathfinder fit
#run from the project root: Rscript R/compare_inits.R
suppressMessages({
  source(here::here("R/functions.R"))
  source(here::here("R/synth_functions.R"))
  pacman::p_load(cmdstanr)
})

patients <- c(4, 8)
window   <- "two_cycles"

#priorDic from the notebook, as in R/fit_synth_patients.Rmd
priors <- list(
  cost_prior     = c(1, 1),
  turnover_prior = c(1.2, 2.5),
  n0_prior       = c(2, 2),
  rFrac_prior    = c(1, 20),
  K_prior        = c(log(2), 0.8),
  rS_prior       = c(log(0.012), 0.6),
  dD_prior       = c(log(2.5), 0.6),
  sigma_prior    = c(-5, 1)
)
fixed_vals <- list(K = 1)
fit_params <- c("cost", "turnover", "n0", "rFrac", "rS", "dD", "sigma")

#one draw from the prior per chain, as in R/fit_synth_patients.Rmd
draw_prior_init <- function() {
  list(cost_free     = array(rbeta(1, priors$cost_prior[1], priors$cost_prior[2])),
       turnover_free = array(rbeta(1, priors$turnover_prior[1], priors$turnover_prior[2])),
       n0_free       = array(rbeta(1, priors$n0_prior[1], priors$n0_prior[2])),
       rFrac_free    = array(max(rbeta(1, priors$rFrac_prior[1], priors$rFrac_prior[2]), 1e-8)),
       rS_free       = array(rlnorm(1, priors$rS_prior[1], priors$rS_prior[2])),
       dD_free       = array(rlnorm(1, priors$dD_prior[1], priors$dD_prior[2])),
       sigma         = rlnorm(1, priors$sigma_prior[1], priors$sigma_prior[2]))
}

stan_model <- cmdstan_model(here::here("stan/model_switchable.stan"))

#fit one patient with one kind of inits. The seconds include the pathfinder run
fit_with_inits <- function(patient_id, inits) {
  stan_data <- make_switchable_stan_data(patient_id, synth_windows[[window]], priors, fixed_vals)
  set.seed(patient_id)
  start <- Sys.time()
  tryCatch({
    init <- switch(inits,
                   default    = NULL,
                   prior      = map(1:4, \(i) draw_prior_init()),
                   pathfinder = stan_model$pathfinder(data = stan_data, seed = patient_id, num_paths = 4,
                                                      refresh = 0, show_messages = FALSE))
    fit <- stan_model$sample(data = stan_data, init = init, chains = 4, parallel_chains = 4, adapt_delta = 0.95,
                             seed = patient_id, refresh = 0, show_messages = FALSE)
    sampler <- fit$diagnostic_summary(quiet = TRUE)
    summ    <- fit$summary(fit_params, "rhat", "ess_bulk")
    tibble(patient       = patient_id,
           inits         = inits,
           seconds       = as.numeric(difftime(Sys.time(), start, units = "secs")),
           slowest_chain = max(fit$time()$chains$total),
           fastest_chain = min(fit$time()$chains$total),
           divergences   = sum(sampler$num_divergent),
           max_rhat      = max(summ$rhat, na.rm = TRUE),
           min_ess_bulk  = min(summ$ess_bulk, na.rm = TRUE))
  }, error = \(e) tibble(patient = patient_id, inits = inits, error = conditionMessage(e)))
}

results <- expand_grid(patient = patients, inits = c("default", "prior", "pathfinder")) |>
  pmap_dfr(fit_with_inits)
print(results, width = Inf)
