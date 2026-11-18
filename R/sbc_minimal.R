# Minimal simulation-based calibration (SBC package) for stan/model_switchable_u.stan.
# Datasets come from the Python simulator (python/sbc_sim.py, through reticulate), so patients are simulated as in the
# parameter recovery runs, but with true values drawn from the fit's prior (no progression rejection).
# Run from biomarker_uncertainty/:   Rscript R/sbc_minimal.R            (N_SIMS=5 Rscript R/sbc_minimal.R for a quick test)
# Needs the SBC package: install.packages("SBC", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
library(SBC)
library(cmdstanr)
library(reticulate)
library(ggplot2)

use_python("/Users/80002536/Documents/moffitt/at-modelling/.venv/bin/python", required = TRUE)
sim <- import_from_path("sbc_sim", path = "python")

n_sims <- as.integer(Sys.getenv("N_SIMS", "20"))  # a real check needs a few hundred
out_dir <- "sbc_results"
dir.create(out_dir, showWarnings = FALSE)
set.seed(1)  # also sets the seed passed to each Python simulation below

# Stan array fields: a length-1 array comes back from Python as a scalar, which Stan rejects for array[] data
array_fields <- c("output_times", "segment_from_idx", "segment_to_idx", "starting_segment_times",
                  "end_segment_times", "treatment_indicator", "obs_idx", "PSA_norm")
generator <- SBC_generator_function(function() {
  out <- sim$simulate_one(seed = sample.int(.Machine$integer.max, 1))
  data <- out$stan_data
  for (nm in array_fields) data[[nm]] <- as.array(data[[nm]])
  list(variables = out$truth, generated = data)
})
datasets <- generate_datasets(generator, n_sims)

#compile into out_dir, so this doesn't overwrite stan/model_switchable_u, the executable cmdstanpy uses
model <- cmdstan_model("stan/model_switchable_u.stan", dir = out_dir)
backend <- SBC_backend_cmdstan_sample(model, chains = 4, iter_warmup = 1000, iter_sampling = 1000, adapt_delta = 0.95)
results <- compute_SBC(datasets, backend)

params <- c("cost", "u", "n0", "rFrac", "rS", "dD", "sigma")
stats <- results$stats[results$stats$variable %in% params, ]
write.csv(stats, file.path(out_dir, "sbc_minimal_stats.csv"), row.names = FALSE)
print(stats[, c("sim_id", "variable", "simulated_value", "rank", "z_score")])
print(results$backend_diagnostics)
ggsave(file.path(out_dir, "sbc_minimal_ecdf_diff.png"), plot_ecdf_diff(results, variables = params), width = 10, height = 6)
ggsave(file.path(out_dir, "sbc_minimal_ranks.png"), plot_rank_hist(results, variables = params), width = 10, height = 6)
