#reads the fits from R/fit_synth_switchable.R and computes diagnostics, parameter recovery and Delta AT per window
#every patient has more than 2 cycles, so each window is its own fit
#run from the project root: Rscript R/analyse_synth_switchable.R. Results go in fits/synth_switchable/results
suppressMessages({
  source(here::here("R/functions.R"))
  source(here::here("R/synth_functions.R"))
  pacman::p_load(cmdstanr, posterior)
})
out_root    <- here::here("fits/synth_switchable")
results_dir <- file.path(out_root, "results")
dir.create(results_dir, showWarnings = FALSE)

#status of every fit (done, timeout or failed)
status <- map_dfr(list.files(out_root, pattern = "_settings\\.rds$", recursive = TRUE, full.names = TRUE), \(path) {
  s <- readRDS(path)
  tibble(patient = s$patient, fit_window = s$window, n_obs = s$n_obs, status = s$status, salvaged = s$salvaged,
         seconds = s$seconds, adapt_delta = s$adapt_delta)
})
patients <- sort(unique(status$patient))

#K is fixed at 1 in the fits and in the simulation
params <- c("cost", "turnover", "n0", "rFrac", "rS", "dD", "sigma", "K")
recovered <- setdiff(params, "K")

#true values. The notebook simulates with K = 1 but saves the drawn K, so it is overwritten here
truth <- read_csv(here::here("synth_data/randSynthetic1000Patients.csv"), show_col_types = FALSE, name_repair = "unique_quiet") |>
  select(patient = PatientId, cost, turnover, n0, rFrac, K, rS, dD = DD, sigma) |>
  filter(patient %in% patients) |>
  mutate(K        = 1,
         AT_score = suppressWarnings(calc_delta_benefit(kappa = .2, dR = turnover * rS, K = K, rR = (1 - cost) * rS, R0 = rFrac * n0)))

#one row per patient x window
windows <- status |>
  transmute(patient, window = fit_window, fit_window, n_obs) |>
  left_join(tibble(patient = patients, cycles = map_int(patients, \(id) max(load_synth_patient(id)$CycleId))), by = "patient")

#draws and diagnostics of every finished fit
fits <- status |> filter(status == "done") |>
  mutate(res = map2(patient, fit_window, \(id, w) {
    fit <- readRDS(make_fit_path(id, file.path(out_root, w)))
    sampler <- fit$diagnostic_summary(quiet = TRUE)
    summ    <- fit$summary(recovered, "rhat", "ess_bulk", "ess_tail")
    list(draws = fit$draws(params, format = "draws_df") |> as_tibble() |> select(all_of(params)),
         diag  = tibble(divergences  = sum(sampler$num_divergent),
                        treedepth    = sum(sampler$num_max_treedepth),
                        min_ebfmi    = min(sampler$ebfmi),
                        max_rhat     = max(summ$rhat, na.rm = TRUE),
                        min_ess_bulk = min(summ$ess_bulk, na.rm = TRUE),
                        min_ess_tail = min(summ$ess_tail, na.rm = TRUE)))
  }))

diag <- fits |> mutate(d = map(res, "diag")) |> select(patient, fit_window, seconds, d) |> unnest(d) |>
  mutate(good = divergences == 0 & max_rhat < 1.01 & min_ess_bulk > 400 & min_ess_tail > 400)
draws <- fits |> mutate(d = map(res, "draws")) |> select(patient, fit_window, d) |> unnest(d)

#posterior summaries per patient x window, with truth
post <- windows |>
  inner_join(draws, by = c("patient", "fit_window"), relationship = "many-to-many") |>
  pivot_longer(all_of(recovered), names_to = "param") |>
  group_by(window, patient, cycles, n_obs, param) |>
  summarise(median = median(value), lower = quantile(value, .05), upper = quantile(value, .95), .groups = "drop") |>
  left_join(truth |> select(patient, all_of(recovered)) |> pivot_longer(-patient, names_to = "param", values_to = "true"),
            by = c("patient", "param")) |>
  mutate(covered = between(true, lower, upper),
         log_err = log(median / true))

recovery <- post |>
  group_by(window, param) |>
  summarise(n = n(), coverage = mean(covered), median_abs_log_err = median(abs(log_err)),
            median_rel_width = median((upper - lower) / true), .groups = "drop")

#Delta AT per draw, conditioned on progression being possible, as in the meeting notes
delta <- windows |>
  inner_join(draws, by = c("patient", "fit_window"), relationship = "many-to-many") |>
  mutate(AT_score     = suppressWarnings(calc_delta_benefit(kappa = .2, dR = turnover * rS, K = K, rR = (1 - cost) * rS, R0 = rFrac * n0)),
         can_progress = progression_possible(cost, turnover, n0, K)) |>
  group_by(window, patient, cycles, n_obs) |>
  summarise(frac_kept = mean(is.finite(AT_score)),
            AT_median = median(AT_score[is.finite(AT_score)]),
            AT_lower  = quantile(AT_score[is.finite(AT_score)], .05),
            AT_upper  = quantile(AT_score[ is.finite(AT_score)], .95), .groups = "drop") |>
  left_join(select(truth, patient, AT_true = AT_score), by = "patient") |>
  filter(is.finite(AT_true)) |>
  mutate(covered = between(AT_true, AT_lower, AT_upper), abs_err = abs(AT_median - AT_true))

delta_summary <- delta |>
  group_by(window) |>
  summarise(n = n(), coverage = mean(covered), median_abs_err = median(abs_err),
            median_width = median(AT_upper - AT_lower), median_frac_kept = median(frac_kept),
            spearman = cor(AT_median, AT_true, method = "spearman"), .groups = "drop")

saveRDS(list(windows = windows, status = status, diag = diag, post = post, recovery = recovery,
             delta = delta, delta_summary = delta_summary, truth = truth),
        file.path(results_dir, "switchable_results.rds"))

options(width = 200, pillar.sigfig = 3)
print(status |> count(fit_window, status) |> pivot_wider(names_from = status, values_from = n, values_fill = 0))
print(diag |> group_by(fit_window) |> summarise(n = n(), good = sum(good), with_div = sum(divergences > 0),
                                                rhat_bad = sum(max_rhat >= 1.01), treedepth = sum(treedepth > 0),
                                                median_s = median(seconds), max_s = max(seconds)))
print(recovery |> select(window, param, coverage) |> pivot_wider(names_from = window, values_from = coverage))
print(recovery |> select(window, param, median_abs_log_err) |> pivot_wider(names_from = window, values_from = median_abs_log_err))
print(delta_summary)
print(status |> group_by(fit_window) |> summarise(fits = n(), salvaged = sum(salvaged), hours = sum(seconds) / 3600))
cat(sprintf("total fitting time: %.1f h over %d fits\n", sum(status$seconds) / 3600, nrow(status)))

