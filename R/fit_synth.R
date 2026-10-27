#fits synthetic patients on every data window (first cycle, two cycles, full history) and saves the fits in fits/synth_<window>
#run from the project root, e.g. Rscript R/fit_synth.R 0 19 to fit patients 0 to 19. Without arguments every patient is fitted
#saved fits are skipped, so the script can be stopped and restarted
source(here::here("R/functions.R"))
source(here::here("R/synth_functions.R"))
pacman::p_load(cmdstanr)

args <- as.integer(commandArgs(trailingOnly = TRUE))
patient_ids <- list_synth_patients()
if (length(args) == 2) patient_ids <- patient_ids[between(patient_ids, args[1], args[2])]

for (window in names(synth_windows)) {
  fit_synth_patients(patient_ids, n_cycles = synth_windows[[window]],
                     out_dir = here::here("fits", paste0("synth_", window)), time_limit = 120,
                     chains = 4, parallel_chains = 4, seed = 1, refresh = 0, adapt_delta = .95,
                     show_messages = FALSE)
}
