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
#Pathfinder inits as in fit_routine.fit_patient: Pathfinder started from one u_init draw, chains started from its draws
stan_data_py <- import_from_path("utils.stan_data", path = "python")
np <- import("numpy")
pathfinder_init <- function(generated) {
  init <- stan_data_py$u_init(sim$priorDic, np$random$default_rng(sample.int(.Machine$integer.max, 1)))
  free <- names(init) != "sigma"
  init[free] <- lapply(init[free], function(x) as.array(as.numeric(x)))  # Stan's array[1] parameters, see array_fields
  #one init per Pathfinder path (4 by default); cmdstanpy reuses the one dict for every path
  model$pathfinder(data = generated, init = rep(list(init), 4), seed = sample.int(.Machine$integer.max, 1),
                   refresh = 0, show_messages = FALSE)
}
backend <- SBC_backend_cmdstan_sample(model, chains = 4, iter_warmup = 1000, iter_sampling = 1000, adapt_delta = 0.95,
                                      init_factory = pathfinder_init)
results <- compute_SBC(datasets, backend)

params <- c("cost", "u", "n0", "rFrac", "rS", "dD", "sigma")
stats <- results$stats[results$stats$variable %in% params, ]
write.csv(stats, file.path(out_dir, "sbc_minimal_stats.csv"), row.names = FALSE)
print(stats[, c("sim_id", "variable", "simulated_value", "rank", "z_score")])
print(results$backend_diagnostics)
#the plots need one max_rank: a sim with a failed chain has fewer draws (lower max_rank), so leave those out of the plots
full <- stats[stats$max_rank == max(stats$max_rank), ]
cat(length(unique(full$sim_id)), "of", length(unique(stats$sim_id)), "sims in the plots\n")
ggsave(file.path(out_dir, "sbc_minimal_ecdf_diff.png"), plot_ecdf_diff(full, variables = params), width = 10, height = 6)
ggsave(file.path(out_dir, "sbc_minimal_ranks.png"), plot_rank_hist(full, variables = params), width = 10, height = 6)
