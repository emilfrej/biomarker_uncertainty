#checks that stan/model_switchable.stan reproduces the python simulator (at_modelling, via the notebook)
#every parameter is fixed at a synthetic patient's true values and Stan is run once with fixed_param, so mu = (S + R) / n0
#at each visit should equal the simulated size / initial size in synth_data/synth_patients.csv, up to solver tolerance.
#This tests the dose convention, the segments and the ODE together, over each patient's full history
#run from the project root: Rscript tests/test_stan_matches_simulator.R [patient ids], e.g. ... 0 1. Default is every patient
suppressMessages({
  source(here::here("R/functions.R"))
  source(here::here("R/synth_functions.R"))
  pacman::p_load(cmdstanr, testthat)
})
options(cmdstanr_warn_inits = FALSE) #the fixed parameters are zero-length arrays, so only sigma needs an init

#largest allowed difference on the normalised scale. Both solvers are RK45 with rtol 1e-6 (Stan's atol is 1e-6, the
#simulator's 1e-8), so differences should be far below this. The observation noise sigma is around 1e-3 to 1e-2
tolerance <- 1e-4

stan_model <- cmdstan_model(here::here("stan/model_switchable.stan"))
truth <- list.files(here::here("synth_data"), pattern = "^randSynthetic\\d+Patients\\.csv$", full.names = TRUE) |>
  read_csv(show_col_types = FALSE, name_repair = "unique_quiet")

#Stan's mu and the simulated size / initial size at every visit of one patient, with the parameters at their true values
compare_with_truth <- function(patient_id) {
  true_pars  <- truth |> filter(PatientId == patient_id)
  fixed_vals <- with(true_pars, list(cost = cost, turnover = turnover, n0 = n0, rFrac = rFrac, rS = rS, K = K, dD = dD))
  #no parameter is fitted, so the priors are never used. sigma is the only parameter left and is set to its true value
  priors <- setNames(rep(list(c(1, 1)), 8), paste0(c("cost", "turnover", "n0", "rFrac", "K", "rS", "dD", "sigma"), "_prior"))
  stan_data <- make_switchable_stan_data(patient_id, Inf, priors, fixed_vals)

  fit <- stan_model$sample(data = stan_data, fixed_param = TRUE, chains = 1, iter_sampling = 1,
                           init = list(list(sigma = true_pars$sigma)), refresh = 0, show_messages = FALSE)
  visits <- load_synth_patient(patient_id)
  tibble(patient   = patient_id,
         Time      = visits$Time,
         simulated = visits$TumourSize / visits$TumourSize[1],
         stan      = as.numeric(fit$draws("mu", format = "draws_matrix")),
         sigma     = true_pars$sigma)
}

patient_ids <- as.integer(commandArgs(trailingOnly = TRUE))
if (!length(patient_ids)) patient_ids <- list_synth_patients()
comparison <- map_dfr(patient_ids, compare_with_truth)

per_patient <- comparison |>
  group_by(patient) |>
  summarise(n_visits = n(), last_visit = max(Time), max_abs_diff = max(abs(stan - simulated)),
            max_diff_over_sigma = max(abs(stan - simulated) / sigma))
print(per_patient |> arrange(desc(max_abs_diff)), n = 5)
cat(sprintf("%d patients, %d visits. Largest difference %.2e (%.2e x sigma)\n", nrow(per_patient), nrow(comparison),
            max(per_patient$max_abs_diff), max(per_patient$max_diff_over_sigma)))

test_that("Stan's mu at the true parameters matches the simulated trajectory at every visit", {
  expect_equal(nrow(comparison), sum(per_patient$n_visits))
  expect_false(anyNA(comparison$stan))
  expect_lt(max(per_patient$max_abs_diff), tolerance)
})
