#helpers for fitting the synthetic AT patients made in python/genertate_synth_AT_trial_data.ipynb
#source after functions.R, since these reuse make_segment_df, make_output_times, make_fixed_val_data and make_fit_path

#data windows that each patient is fitted on, as the number of cycles kept. Inf is the full history
synth_windows <- c(first_cycle = 1, two_cycles = 2, full_history = Inf)

#parameters that are compared against the true values. n0 is fixed at 1 in the fits, psa0 is the fitted baseline level
synth_params <- c("cost", "turnover", "n0", "rFrac", "K", "rS", "dD", "sigma", "psa0")

#priors for the synthetic fits. Uniform where priorDic is beta (as in make_priors), lognormals as in priorDic
#K is relative to the baseline size (n0 = 1). With K ~ lognormal(log(2), 0.8) and n0 ~ beta(2, 2) in priorDic,
#log(K / n0) has mean log(2) + 0.83 and sd sqrt(0.8^2 + 0.6^2)
make_synth_priors <- function(){
  priors <- make_priors()
  priors$K_prior     <- c(1.53, 1)
  priors$rS_prior    <- c(log(0.012), 0.6)
  priors$dD_prior    <- c(log(2.5), 0.6)
  priors$sigma_prior <- c(-5, 1)
  return(priors)
}

#true parameters of every synthetic patient, renamed to match the stan model
#the fits work relative to the baseline size, so the true n0 becomes psa0, K becomes K / n0 and n0 is 1
load_synth_truth <- function(dir = here::here("synth_data")){
  truth_file <- list.files(dir, pattern = "^randSynthetic\\d+Patients\\.csv$", full.names = TRUE)
  if (length(truth_file) != 1) stop("expected one randSynthetic*Patients.csv in ", dir)
  read_csv(truth_file, show_col_types = FALSE) |>
    select(patient = PatientId, cost, turnover, n0, rFrac, K, rS, dD = DD, sigma) |>
    mutate(AT_score = calc_delta_benefit(kappa = .2, dR = turnover * rS, K = K, rR = (1 - cost) * rS, R0 = rFrac * n0),
           psa0     = n0,
           K        = K / n0,
           n0       = 1)
}

#ids of the synthetic patients that have an observation file
list_synth_patients <- function(dir = here::here("synth_data")){
  list.files(dir, pattern = "^patient_\\d+_OBSdf\\.csv$") |>
    parse_number() |>
    sort()
}

#loads one synthetic patient in the same format as load_bruchovsky_patient, so make_segment_df works on it
#in the python output DrugConcentration is the dose given in the interval leading up to a visit, here it is the dose from the visit on
#a cycle is a treatment period and the treatment holiday after it. The visit where treatment restarts still belongs to the old cycle,
#since its PSA was measured at the end of the holiday
load_synth_patient <- function(patient_id, dir = here::here("synth_data")){
  read_csv(file.path(dir, sprintf("patient_%d_OBSdf.csv", patient_id)), show_col_types = FALSE, name_repair = "unique_quiet") |>
    select(Time, PSA = Observation, TumourSize, dose_before = DrugConcentration) |>
    arrange(Time) |>
    mutate(DrugConcentration = as.integer(lead(dose_before, default = last(dose_before))),
           CycleId           = 1 + cumsum(dose_before == 1 & lag(dose_before, default = 1) == 0),
           segment           = consecutive_id(DrugConcentration))
}

#build the Stan data list for one synthetic patient, keeping the first n_cycles cycles
#the data stays on its raw scale and is normalised in Stan: n0 is fixed at 1 and the fitted psa0 scales the model to the data
make_synth_stan_data <- function(patient_id, n_cycles = Inf, priors, fixed_vals = list(), run_diagnostics = 1, prior_only = 0,
                                 psa0_prior_sd = 0.5) {

  patient_data <- load_synth_patient(patient_id) |>
    filter(CycleId <= n_cycles)

  #segments for the ODE solver and observations for the likelihood, as in make_stan_data
  segments_df  <- make_segment_df(patient_data)
  output_times <- make_output_times(patient_data, segments_df)
  obs_df       <- patient_data |> filter(!is.na(PSA))
  if (obs_df$PSA[1] <= 0) stop("patient ", patient_id, ": first observation is not positive, so it cannot centre the psa0 prior")

  c(
    list(
      n_segments             = nrow(segments_df),
      starting_segment_times = segments_df$segment_start_time,
      end_segment_times      = segments_df$segment_end_time,
      treatment_indicator    = segments_df$Treatment,
      n_out                  = length(output_times),
      output_times           = output_times,
      segment_from_idx       = segments_df$segment_from_idx,
      segment_to_idx         = segments_df$segment_to_idx,
      n_obs                  = nrow(obs_df),
      obs_idx                = match(obs_df$Time, output_times),
      PSA_norm               = obs_df$PSA
    ),
    make_fixed_val_data(modifyList(list(n0 = 1), as.list(fixed_vals))),
    fit_psa0   = 1,
    list(psa0_prior = c(log(obs_df$PSA[1]), psa0_prior_sd)), #centred on the first observation. In a list so c() keeps it as one vector
    prior_only = prior_only,
    priors,
    run_diagnostics = run_diagnostics)
}

#fit a set of synthetic patients on one window and save one .rds (+ CSVs and settings) per patient. Saved fits are skipped
#each fit runs in its own R process that is killed after time_limit seconds. A killed fit leaves a _timeout.rds marker
#and is skipped on later runs, unless overwrite is TRUE
fit_synth_patients <- function(patient_ids,
                               n_cycles,
                               out_dir,
                               stan_file  = here::here("stan/model_switchable.stan"),
                               priors     = make_synth_priors(),
                               fixed_vals = list(),
                               overwrite  = FALSE,
                               time_limit = 120,
                               ...) {                       # extra args go to $sample(), e.g. chains, adapt_delta, seed
  mod <- cmdstan_model(stan_file) #compile once here, so the fit processes only load the model
  csv_dir <- file.path(out_dir, "csv")
  dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)

  message("out: ", out_dir, " | cycles: ", n_cycles, " | patients: ", length(patient_ids), " | time limit: ", time_limit, " s")

  for (patient_id in patient_ids) {
    fit_path     <- make_fit_path(patient_id, dir = out_dir)
    timeout_path <- sub("\\.rds$", "_timeout.rds", fit_path)
    if ((file.exists(fit_path) || file.exists(timeout_path)) && !overwrite) next

    stan_data <- make_synth_stan_data(patient_id, n_cycles, priors, fixed_vals)
    if (stan_data$n_obs < 5) { message("patient ", patient_id, ": fewer than 5 observations, skipping"); next }
    settings <- list(priors = priors, fixed_vals = fixed_vals, n_cycles = n_cycles, n_obs = stan_data$n_obs,
                     stan_file = stan_file, stan_code = mod$code(), time_limit = time_limit)

    start_time <- Sys.time()
    fit_process <- callr::r_bg(function(stan_file, stan_data, csv_dir, basename, fit_path, sample_args) {
        mod <- cmdstanr::cmdstan_model(stan_file)
        fit <- do.call(mod$sample, c(list(data = stan_data, output_dir = csv_dir, output_basename = basename), sample_args))
        if (all(fit$return_codes() == 0)) fit$save_object(fit_path)
        fit$return_codes()
      },
      args = list(stan_file = stan_file, stan_data = stan_data, csv_dir = csv_dir,
                  basename = sprintf("patient%03d", patient_id), fit_path = fit_path, sample_args = list(...)),
      supervise = TRUE)
    fit_process$wait(timeout = time_limit * 1000)
    if (fit_process$is_alive()) {
      fit_process$kill_tree() #also kills the cmdstan chains started by the fit process
      return_codes <- "timeout"
    } else {
      return_codes <- tryCatch(fit_process$get_result(),
                               error = function(e) { message("patient ", patient_id, " failed: ", conditionMessage(e)); NULL })
    }
    seconds <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    settings <- c(settings, date = Sys.time(), seconds = seconds)

    if (identical(return_codes, "timeout")) {
      saveRDS(settings, timeout_path)
      message("patient ", patient_id, ": stopped after ", time_limit, " s")
    } else if (!is.null(return_codes) && all(return_codes == 0)) {
      #save priors and settings next to the fit so we know what produced it
      saveRDS(settings, sub("\\.rds$", "_settings.rds", fit_path))
      message("patient ", patient_id, ": done in ", round(seconds), " s")
    } else if (!is.null(return_codes)) {
      message("patient ", patient_id, ": ", sum(return_codes != 0), " chain(s) failed, not saved")
    }
  }
}

#number of PSA observations a patient has in the first n_cycles cycles
count_synth_obs <- function(patient_id, n_cycles = Inf){
  load_synth_patient(patient_id) |>
    filter(CycleId <= n_cycles, !is.na(PSA)) |>
    nrow()
}

#posterior draws of the compared parameters and sampler diagnostics for every saved fit in a window, and the fits that timed out
#the result is cached next to the fits, and rebuilt when a fit or timeout marker is newer than the cache
read_synth_window <- function(window, dir = here::here("fits", paste0("synth_", window))){
  cache_path    <- file.path(dir, "summary_cache.rds")
  fit_files     <- list.files(dir, pattern = "^fit_patient\\d+\\.rds$", full.names = TRUE)
  timeout_files <- list.files(dir, pattern = "^fit_patient\\d+_timeout\\.rds$", full.names = TRUE)
  if (!length(fit_files)) return(NULL)
  if (file.exists(cache_path) && file.mtime(cache_path) > max(file.mtime(c(fit_files, timeout_files)))) return(readRDS(cache_path))

  per_fit <- map(fit_files, \(path) {
    fit <- readRDS(path)
    id  <- parse_number(basename(path))
    settings <- readRDS(sub("\\.rds$", "_settings.rds", path))
    sampler  <- fit$diagnostic_summary(quiet = TRUE)
    summ     <- fit$summary(synth_params, "rhat", "ess_bulk", "ess_tail")
    list(
      draws = fit$draws(synth_params, format = "draws_df") |> as_tibble() |>
        select(all_of(synth_params)) |> mutate(patient = id),
      diagnostics = tibble(patient     = id,
                           n_obs       = settings$n_obs,
                           seconds     = settings$seconds,
                           divergences = sum(sampler$num_divergent),
                           treedepth   = sum(sampler$num_max_treedepth),
                           min_ebfmi   = min(sampler$ebfmi),
                           max_rhat    = max(summ$rhat, na.rm = TRUE),
                           min_ess_bulk = min(summ$ess_bulk, na.rm = TRUE),
                           min_ess_tail = min(summ$ess_tail, na.rm = TRUE))
    )
  })
  out <- list(draws       = map_dfr(per_fit, "draws"),
              diagnostics = map_dfr(per_fit, "diagnostics"),
              timeouts    = map_dfr(timeout_files, \(path) tibble(patient = parse_number(basename(path)),
                                                                  n_obs   = readRDS(path)$n_obs)))
  saveRDS(out, cache_path)
  out
}
