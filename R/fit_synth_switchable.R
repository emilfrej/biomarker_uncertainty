#fits synthetic patients with stan/model_switchable.stan on the first cycle, two cycles and the full history
#K is fixed at 1 (as in the notebook), n0 is fitted on the raw observations, and the priors are priorDic from the notebook
#only patients with more than 2 cycles are fitted, so the three windows always hold different data
#run from the project root: Rscript R/fit_synth_switchable.R <n patients> <worker id> <n workers> <time limit s>
#e.g. Rscript R/fit_synth_switchable.R 100 0 2 600 and Rscript R/fit_synth_switchable.R 100 1 2 600 fit the first 100 such patients
#in two processes. Fits go in fits/synth_switchable/<window>
#an optional 5th argument, e.g. "2026-10-28 07:55", is a deadline: no fit is started that could run past it
suppressMessages({
  source(here::here("R/functions.R"))
  source(here::here("R/synth_functions.R"))
  pacman::p_load(cmdstanr)
})

args       <- commandArgs(trailingOnly = TRUE)
n_patients <- as.numeric(args[1])
worker     <- as.numeric(args[2])
n_workers  <- as.numeric(args[3])
time_limit <- as.numeric(args[4])
deadline   <- if (length(args) >= 5) as.POSIXct(args[5]) else as.POSIXct(Inf)

stan_file <- here::here("stan/model_switchable.stan")
out_root  <- here::here("fits/synth_switchable")

#priorDic in python/genertate_synth_AT_trial_data.ipynb. scipy lognorm(s, scale) is lognormal(log(scale), s)
#K_prior is not used since K is fixed
priors <- list(
  cost_prior     = c(1, 1),
  turnover_prior = c(1.2, 2.5),
  n0_prior       = c(2, 2),
  rFrac_prior    = c(0.3, 40),
  K_prior        = c(log(2), 0.8),
  rS_prior       = c(log(0.012), 0.6),
  dD_prior       = c(log(2.5), 0.6),
  sigma_prior    = c(-5, 1)
)
fixed_vals <- list(K = 1)

#same as make_synth_stan_data, but n0 is fitted and the model is compared with the raw observations directly (no psa0)
make_switchable_stan_data <- function(patient_id, n_cycles) {
  patient_data <- load_synth_patient(patient_id) |> filter(CycleId <= n_cycles)
  segments_df  <- make_segment_df(patient_data)
  output_times <- make_output_times(patient_data, segments_df)
  obs_df       <- patient_data |> filter(!is.na(PSA))
  c(list(n_segments             = nrow(segments_df),
         starting_segment_times = segments_df$segment_start_time,
         end_segment_times      = segments_df$segment_end_time,
         treatment_indicator    = segments_df$Treatment,
         n_out                  = length(output_times),
         output_times           = output_times,
         segment_from_idx       = segments_df$segment_from_idx,
         segment_to_idx         = segments_df$segment_to_idx,
         n_obs                  = nrow(obs_df),
         obs_idx                = match(obs_df$Time, output_times),
         PSA_norm               = obs_df$PSA),
    make_fixed_val_data(fixed_vals),
    fit_psa0   = 0,
    list(psa0_prior = c(0, 1)), #not used with fit_psa0 = 0
    prior_only = 0,
    priors,
    run_diagnostics = 1)
}

#one draw from the prior per chain. Stan's default inits (uniform(-2, 2) on the unconstrained scale) put e.g. rS up to 7,
#where the ODE gets stiff and chains got stuck in warmup. The julia port had the same problem (julia_port/PROJECT_LOG.md)
draw_prior_init <- function() {
  list(cost_free     = array(rbeta(1, priors$cost_prior[1], priors$cost_prior[2])),
       turnover_free = array(rbeta(1, priors$turnover_prior[1], priors$turnover_prior[2])),
       n0_free       = array(rbeta(1, priors$n0_prior[1], priors$n0_prior[2])),
       #rFrac draws from beta(0.3, 40) can underflow to 0, which is outside the support
       rFrac_free    = array(max(rbeta(1, priors$rFrac_prior[1], priors$rFrac_prior[2]), 1e-8)),
       rS_free       = array(rlnorm(1, priors$rS_prior[1], priors$rS_prior[2])),
       dD_free       = array(rlnorm(1, priors$dD_prior[1], priors$dD_prior[2])),
       sigma         = rlnorm(1, priors$sigma_prior[1], priors$sigma_prior[2]))
}

#the first n_patients patients with more than 2 cycles, and one job per patient and window
cycles   <- map_int(list_synth_patients(), \(id) max(load_synth_patient(id)$CycleId))
patients <- head(list_synth_patients()[cycles > 2], n_patients)
jobs <- map_dfr(patients, \(id) {
  tibble(patient = id, window = names(synth_windows), n_cycles = unname(synth_windows),
         n_obs = map_int(synth_windows, \(w) count_synth_obs(id, w)))
})
#spread the jobs over the workers, interleaved so each worker gets a mix of short and long histories
jobs <- jobs[seq_len(nrow(jobs)) %% n_workers == worker, ]
message("worker ", worker, ": ", nrow(jobs), " fits, time limit ", time_limit, " s")

mod <- cmdstan_model(stan_file)

#higher than the default .8: at .8 the first fit (patient 0, first cycle) had 135 divergences, at .95 none
adapt_delta <- .95

#one fit in a separate process that is killed after time_limit seconds. Returns the status and the number of divergences
run_fit <- function(stan_data, inits, csv_dir, basename, fit_path) {
  fit_process <- callr::r_bg(function(stan_file, stan_data, inits, csv_dir, basename, fit_path, adapt_delta) {
      mod <- cmdstanr::cmdstan_model(stan_file)
      fit <- mod$sample(data = stan_data, init = inits, output_dir = csv_dir, output_basename = basename,
                        chains = 4, parallel_chains = 4, seed = 1, refresh = 0, adapt_delta = adapt_delta,
                        show_messages = FALSE)
      if (all(fit$return_codes() == 0)) fit$save_object(fit_path)
      fit$return_codes()
    },
    args = list(stan_file = stan_file, stan_data = stan_data, inits = inits, csv_dir = csv_dir,
                basename = basename, fit_path = fit_path, adapt_delta = adapt_delta),
    supervise = TRUE)
  fit_process$wait(timeout = time_limit * 1000)
  csv_files <- list.files(csv_dir, pattern = paste0("^", basename, ".*\\.csv$"), full.names = TRUE)
  if (fit_process$is_alive()) {
    fit_process$kill_tree()
    #the R process sometimes hangs after all chains have finished, so keep the fit if all 4 csvs are complete
    salvaged <- length(csv_files) == 4 && all(map_lgl(csv_files, \(f) any(grepl("Elapsed Time", readLines(f)))))
    status   <- if (salvaged) "done" else "timeout"
    if (salvaged) as_cmdstan_fit(csv_files)$save_object(fit_path)
  } else {
    salvaged <- FALSE
    codes    <- tryCatch(fit_process$get_result(), error = \(e) NULL)
    status   <- if (!is.null(codes) && all(codes == 0)) "done" else "failed"
  }
  divergences <- if (status == "done") {
    sum(readRDS(fit_path)$diagnostic_summary("divergences", quiet = TRUE)$num_divergent)
  } else NA
  #the csvs are in the saved fit object, so drop them to keep disk use down
  unlink(csv_files)
  list(status = status, salvaged = salvaged, divergences = divergences)
}

run_start <- Sys.time()
for (j in seq_len(nrow(jobs))) {
  job       <- jobs[j, ]
  out_dir   <- file.path(out_root, job$window)
  csv_dir   <- file.path(out_dir, "csv")
  dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)
  fit_path  <- make_fit_path(job$patient, dir = out_dir)
  done_path <- sub("\\.rds$", "_settings.rds", fit_path)
  if (file.exists(done_path)) next
  if (Sys.time() + time_limit > deadline) {
    message("stopping: the next fit could run past the deadline ", format(deadline))
    break
  }

  stan_data  <- make_switchable_stan_data(job$patient, job$n_cycles)
  set.seed(job$patient)
  inits      <- replicate(4, draw_prior_init(), simplify = FALSE)
  start_time <- Sys.time()
  result     <- run_fit(stan_data, inits, csv_dir, sprintf("patient%03d", job$patient), fit_path)
  seconds <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  saveRDS(list(patient = job$patient, window = job$window, n_cycles = job$n_cycles, n_obs = job$n_obs,
               status = result$status, salvaged = result$salvaged, seconds = seconds, adapt_delta = adapt_delta,
               divergences = result$divergences, priors = priors, fixed_vals = fixed_vals, inits = inits,
               stan_code = mod$code(), date = Sys.time()),
          done_path)
  message(sprintf("patient %d %s (%d obs): %s%s in %.0f s, adapt_delta %.2f, %s divergences",
                  job$patient, job$window, job$n_obs, result$status, if (result$salvaged) " (salvaged)" else "",
                  seconds, adapt_delta, result$divergences))
}
run_end <- Sys.time()

#timing of the whole run plus one row per fit of this worker (also fits skipped because they were done by an earlier run)
settings_paths <- map_chr(seq_len(nrow(jobs)), \(j) {
  sub("\\.rds$", "_settings.rds", make_fit_path(jobs$patient[j], dir = file.path(out_root, jobs$window[j])))
})
fit_times <- map_dfr(settings_paths[file.exists(settings_paths)], \(path) {
  s <- readRDS(path)
  tibble(patient = s$patient, window = s$window, n_obs = s$n_obs, status = s$status, salvaged = s$salvaged,
         seconds = s$seconds, divergences = s$divergences)
})
write_csv(fit_times, file.path(out_root, sprintf("fit_times_worker%d.csv", worker)))
saveRDS(list(run_start = run_start, run_end = run_end, n_patients = n_patients, worker = worker, n_workers = n_workers,
             time_limit = time_limit, adapt_delta = adapt_delta, fit_times = fit_times),
        file.path(out_root, sprintf("run_summary_worker%d.rds", worker)))
message(sprintf("worker %d finished %d fits in %.1f h (summed fit time %.1f h)", worker, nrow(fit_times),
                as.numeric(difftime(run_end, run_start, units = "hours")), sum(fit_times$seconds) / 3600))
