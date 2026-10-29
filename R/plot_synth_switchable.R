#plots diagnostics, parameter recovery and Delta AT from switchable_results.rds (made by R/analyse_synth_switchable.R)
#run from the project root: Rscript R/plot_synth_switchable.R. Plots go in fits/synth_switchable/results
suppressMessages(pacman::p_load(tidyverse, patchwork))
results_dir <- here::here("fits/synth_switchable/results")
res <- readRDS(file.path(results_dir, "switchable_results.rds"))

#reference palette (dataviz skill), slots 1 and 2, light mode
col_in   <- "#2a78d6"
col_out  <- "#eb6834"
surface  <- "#fcfcfb"
ink      <- "#0b0b0b"
ink_2    <- "#52514e"
grid_col <- "#e7e6e2"
window_labels <- c(first_cycle = "First cycle", two_cycles = "Two cycles", full_history = "Full history")

theme_set(theme_minimal(base_size = 11) +
  theme(plot.background = element_rect(fill = surface, colour = NA),
        panel.grid.major = element_line(colour = grid_col, linewidth = .3), panel.grid.minor = element_blank(),
        text = element_text(colour = ink), axis.text = element_text(colour = ink_2),
        strip.text = element_text(colour = ink, face = "bold"), legend.position = "top",
        plot.title = element_text(face = "bold"), plot.subtitle = element_text(colour = ink_2)))
win <- \(x) factor(window_labels[x], levels = window_labels)

#---- parameter recovery: true value against posterior median with 90% interval
log_params <- c()
coverage_lab <- res$recovery |>
  mutate(label = sprintf("%.0f%% covered", 100 * coverage))
rec <- res$post |>
  mutate(across(c(median, lower, upper, true), \(v) if_else(param %in% log_params, log10(v), v)),
         param = if_else(param %in% log_params, paste0("log10 ", param), param),
         param = factor(param, levels = c("cost", "turnover", "n0", "rFrac", "rS", "dD", "sigma")),
         window = win(window), covered = if_else(covered, "Truth in 90% interval", "Truth outside"))
cov_txt <- coverage_lab |>
  mutate(param = factor(if_else(param %in% log_params, paste0("log10 ", param), param), levels = levels(rec$param)),
         window = win(window))
p_rec <- ggplot(rec, aes(true, median)) +
  geom_abline(colour = ink_2, linewidth = .3, linetype = "dashed") +
  geom_linerange(aes(ymin = lower, ymax = upper), linewidth = .35, alpha = .6) +
  geom_point( size = 1.2) +
  facet_grid(param ~ window, scales = "free") +
  labs(x = "True value", y = "Posterior median (90% interval)",
       title = "Parameter recovery by data window",
       subtitle = "Dashed line: perfect recovery. Rows are shared scales across windows.")
ggsave(file.path(results_dir, "recovery.png"), p_rec, width = 9, height = 15, dpi = 150)

#one png per parameter, with free scales in each window. Full history is left out
for (p in levels(rec$param)) {
  p_param <- ggplot(filter(rec, param == p, window != "Full history"), aes(true, median)) +
    geom_abline(colour = ink_2, linewidth = .3, linetype = "dashed") +
    geom_linerange(aes(ymin = lower, ymax = upper), linewidth = .35, alpha = .6) +
    geom_point(size = 1.2) +
    facet_wrap(~ window, scales = "free") +
    labs(x = paste("True", p), y = "Posterior median (90% interval)")
  ggsave(file.path(results_dir, sprintf("recovery_%s.png", p)), p_param, width = 8, height = 4.5, dpi = 150)
}

#---- Delta AT
dat <- res$delta |> mutate(window = win(window), covered = if_else(covered, "Truth in 90% interval", "Truth outside"))
dat_txt <- res$delta_summary |>
  mutate(window = win(window),
         label = sprintf("%.0f%% covered\nmedian |error| %.2f\nSpearman %.2f", 100 * coverage, median_abs_err, spearman))
p_dat <- ggplot(dat, aes(AT_true, AT_median)) +
  geom_abline(colour = ink_2, linewidth = .3, linetype = "dashed") +
  geom_linerange(aes(ymin = AT_lower, ymax = AT_upper), linewidth = .35, alpha = .6) +
  facet_wrap(~ window) +
  labs(x = "True AT Score", y = "Estimated AT Score (90% interval)")
ggsave(file.path(results_dir, "delta_at.png"), p_dat, width = 11, height = 4.5, dpi = 150)

#---- diagnostics
status <- res$status |> mutate(window = win(fit_window))
diag   <- res$diag |> left_join(select(res$status, patient, fit_window, n_obs), by = c("patient", "fit_window")) |>
  mutate(window = win(fit_window), any_div = if_else(divergences > 0, "Divergences", "No divergences"))
p_status <- status |> count(window, status) |>
  ggplot(aes(window, n, fill = status)) +
  geom_col(position = position_dodge(width = .8), width = .7) +
  geom_text(aes(label = n), position = position_dodge(width = .8), vjust = -.4, size = 3, colour = ink) +
  scale_fill_manual(values = c(done = col_in, timeout = col_out, failed = ink_2), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, .12))) +
  labs(x = NULL, y = "Fits", title = "Fit status (5 min cap)")
p_time <- ggplot(status, aes(n_obs, seconds, colour = status)) +
  geom_hline(yintercept = 300, colour = ink_2, linewidth = .3, linetype = "dashed") +
  geom_point(size = 1.4) +
  facet_wrap(~ window, scales = "free_x") +
  scale_colour_manual(values = c(done = col_in, timeout = col_out, failed = ink_2), name = NULL) +
  labs(x = "Observations", y = "Seconds", title = "Fit time")
p_rhat <- ggplot(diag, aes(n_obs, max_rhat, colour = any_div)) +
  geom_hline(yintercept = 1.01, colour = ink_2, linewidth = .3, linetype = "dashed") +
  geom_point(size = 1.4) +
  facet_wrap(~ window, scales = "free_x") +
  scale_colour_manual(values = c("No divergences" = col_in, "Divergences" = col_out), name = NULL) +
  labs(x = "Observations", y = "Max R-hat", title = "Convergence of finished fits", subtitle = "Dashed line: R-hat 1.01")
p_ess <- ggplot(diag, aes(n_obs, min_ess_bulk, colour = any_div)) +
  geom_hline(yintercept = 400, colour = ink_2, linewidth = .3, linetype = "dashed") +
  geom_point(size = 1.4) +
  facet_wrap(~ window, scales = "free_x") +
  scale_y_log10() +
  scale_colour_manual(values = c("No divergences" = col_in, "Divergences" = col_out), name = NULL) +
  labs(x = "Observations", y = "Min bulk ESS (log)", subtitle = "Dashed line: ESS 400")
p_diag <- (p_status | p_time) / p_rhat / p_ess + plot_layout(heights = c(1, 1, 1)) &
  theme(plot.background = element_rect(fill = surface, colour = NA))
ggsave(file.path(results_dir, "diagnostics.png"), p_diag, width = 12, height = 11, dpi = 150)

divergence_plot <- ggplot(diag, aes(x=divergences)) +
  geom_density() +
  facet_wrap(~fit_window)

ggsave(file.path(results_dir, "divergence.png"), divergence_plot, width = 12, height = 11, dpi = 150)