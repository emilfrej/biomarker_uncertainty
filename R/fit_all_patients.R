pacman::p_load(cmdstanr, tidyverse)
source(here::here("R/functions.R"))
set.seed(1)

priors <- list(
  cost_prior     = c(5, 10),
  turnover_prior = c(2, 2),
  n0_prior       = c(2, 2),
  rFrac_prior    = c(2, 30),
  K_prior        = c(log(2), .4),
  rS_prior       = c(log(0.019), 0.025),
  sigma_prior    = c(-2, 1)
)

for (cycle in c(TRUE, FALSE)) {
  fits <- fit_patients(
    stan_file   = here::here("stan/free_k_free_rs_prototype_model.stan"),
    priors      = priors,
    first_cycle = cycle,
    overwrite   = F,
    chains = 4, parallel_chains = 4, adapt_delta = .8
  )
}