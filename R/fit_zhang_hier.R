#fits stan/model_hier.stan to all Zhang patients and prints the sampler diagnostics
#run from the project root: Rscript R/fit_zhang_hier.R <noise: normal|lognormal> <n_par: 4|6> [adapt_delta] [warmup] [sampling]
#with 4 parameters, rS and dD are fixed at Strobl's values (0.027 and 1.5) for every patient
#the fit is saved in fits/zhang_hier/<noise>_<n_par>par
suppressMessages({
  source(here::here("R/functions.R"))
  source(here::here("R/zhang_hier_functions.R"))
  pacman::p_load(cmdstanr)
})

args        <- commandArgs(trailingOnly = TRUE)
noise       <- args[1]
n_par       <- as.integer(args[2])
adapt_delta <- if (length(args) >= 3) as.numeric(args[3]) else 0.95
warmup      <- if (length(args) >= 4) as.integer(args[4]) else 1000
sampling    <- if (length(args) >= 5) as.integer(args[5]) else 1000

free  <- if (n_par == 4) c("cost", "turnover", "n0", "rFrac") else hier_par_names
fixed <- if (n_par == 4) c(rS = 0.027, dD = 1.5) else c()
stan_data <- make_hier_stan_data(list_zhang_patients(), free, fixed, lognormal_noise = noise == "lognormal")

out_dir <- here::here("fits/zhang_hier", sprintf("%s_%dpar", noise, n_par))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(1)
model <- cmdstan_model(here::here("stan/model_hier.stan"))
fit <- model$sample(data = stan_data, init = map(1:4, \(i) hier_init(stan_data)), chains = 4, parallel_chains = 4,
                    iter_warmup = warmup, iter_sampling = sampling, adapt_delta = adapt_delta, seed = 1,
                    refresh = 100, output_dir = out_dir)
fit$save_object(file.path(out_dir, "fit.rds"))

#diagnostics: population parameters, then the worst patient level parameter
sampler <- fit$diagnostic_summary(quiet = TRUE)
cat("\n", noise, n_par, "parameters, adapt_delta", adapt_delta, "\n")
cat("divergences per chain:", sampler$num_divergent, "\nmax treedepth hits per chain:", sampler$num_max_treedepth,
    "\nE-BFMI per chain:", round(sampler$ebfmi, 2), "\nseconds per chain:", round(fit$time()$chains$total), "\n")
population <- fit$summary(c("mu", "tau", "sigma"), "median", ~quantile(.x, c(0.05, 0.95)), "rhat", "ess_bulk", "ess_tail")
print(population, n = Inf)
patients <- fit$summary("pars", "rhat", "ess_bulk", "ess_tail")
cat("patient level pars: max rhat", round(max(patients$rhat, na.rm = TRUE), 3),
    " min ess_bulk", round(min(patients$ess_bulk, na.rm = TRUE)), " min ess_tail", round(min(patients$ess_tail, na.rm = TRUE)), "\n")
