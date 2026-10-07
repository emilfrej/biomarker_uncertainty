pacman::p_load(cmdstanr, tidyverse)
source(here::here("R/functions.R"))

#make out dir
out_dir <- here::here("fits/stan_all_adapt_delta_95_patients")
dir.create(out_dir, recursive = TRUE, showWarnings = TRUE)

#list all files in bruchowsky data
data_paths <- list.files(here::here("dataTanaka/Bruchovsky_et_al"), pattern = "^patient\\d+\\.txt$", full.names=T)

mod <- cmdstan_model(here::here("stan/prototype_model.stan"))

fixed_vals <- list(rS = 0.027, K = 1, dD = 1.5)

priors <- list(
  cost_prior     = c(1, 1),
  turnover_prior = c(2, 5),
  n0_prior       = c(1, 1),
  rFrac_prior    = c(1, 30),
  sigma_prior    = c(-2, 1)
)

OVERWRITE <- F
target_ids <- c(12, 20, 25, 78, 85)

#fit all data and save.
for (path in data_paths){
  
  patient_id <- parse_number(basename(path))

  #make fit path
  fit_path <- make_fit_path(patient_id, dir = out_dir)
  
  #skip if fit exists and overwrite is not on.
  if (file.exists(fit_path) && !(OVERWRITE)) next
  
  #skip if patient ID is not in target id
  if (!is.null(target_ids) && !(patient_id %in% target_ids)) next
  
  #read in the data
  patient_data <- load_bruchowsky_patients(patient_id)[[as.character(patient_id)]]
  
  ## setup data
  days <- 0:max(patient_data$Time)
  treatment_indicator <- as.integer(patient_data$DrugConcentration[findInterval(days, patient_data$Time)])
  
  obs <- patient_data |> filter(!is.na(PSA), Time > 0)
  
  stan_data <- c(
    list(
      n_obs               = nrow(obs),
      measurement_times   = obs$Time,
      PSA_norm            = obs$PSA,
      n_days              = length(treatment_indicator),
      treatment_indicator = treatment_indicator,
      run_diagnostics     = 1
    ),
    fixed_vals,
    priors
  )
  
  ## try to run model
  #run path finder for values
  # pf <- tryCatch(
  #   mod$pathfinder(data = stan_data, num_paths = 4, seed = patient_id, refresh = 0),
  #   error = function(e) { message("patient ", patient_id, " pathfinder failed, using default inits"); NULL }
  # )
  
  fit <- tryCatch(
    mod$sample(data = stan_data, chains = 4, parallel_chains = 4, seed = patient_id, adapt_delta = .95),
    error = function(e) { message("patient ", patient_id, " failed: ", conditionMessage(e)); NULL }
  )
  
  # only save if it ran and every chain finished
  if (!is.null(fit) && all(fit$return_codes() == 0)) {
    fit$save_object(fit_path)
  }

}