pacman::p_load(cmdstanr, tidyverse)
source(here::here("R/functions.R"))
set.seed(1)

for (cycle in c(TRUE, FALSE)) {
  fits <- fit_patients(
    patient_ids = c(12, 20, 25, 78, 85),
    stan_file   = here::here("stan/model.stan"),
    priors      = make_priors(),
    first_cycle = cycle,
    out_dir     = here::here("fits", paste0("segmented_", if (cycle) "first_cycle" else "full_history")), #keep separate from old model fits
    overwrite   = T,
    chains = 4, parallel_chains = 4, adapt_delta = .99
  )
}


