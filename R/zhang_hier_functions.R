#helpers for fitting stan/model_hier.stan to the Zhang patients in data/
#source after functions.R, since these reuse make_segment_df and make_output_times

#the 6 patient level parameters, in the order model_hier.stan uses
hier_par_names <- c("cost", "turnover", "n0", "rFrac", "rS", "dD")

#priors on the unconstrained scale: logit for cost, turnover, n0 and rFrac, log for rS and dD
#mu_loc for rS and dD are the values Strobl fixed in the 4 parameter fits (rS = 0.027, dD = 1.5)
hier_priors <- list(mu_loc      = c(cost = 0, turnover = -1, n0 = -1, rFrac = -3, rS = log(0.027), dD = log(1.5)),
                    mu_scale    = c(1.5, 1.5, 1.5, 1.5, 0.75, 0.75),
                    tau_scale   = rep(1, 6),
                    lkj_eta     = 2,
                    sigma_scale = 1)

#ids of the Zhang patients
list_zhang_patients <- function(dir = here::here("data")){
  sort(parse_number(list.files(dir, pattern = "^patient\\d+\\.csv$")))
}

#one Zhang patient in the format make_segment_df takes. PSA is divided by the first PSA,
#Abi is the dose from each visit on (the same convention as the synthetic data)
load_zhang_patient <- function(patient_id, dir = here::here("data")){
  read_csv(file.path(dir, sprintf("patient%d.csv", patient_id)), show_col_types = FALSE) |>
    arrange(Days) |>
    transmute(Time              = Days,
              PSA               = PSA / first(PSA),
              DrugConcentration = as.integer(Abi),
              segment           = consecutive_id(DrugConcentration))
}

#unconstrained value of a natural scale parameter, as in model_hier.stan
to_unconstrained <- function(value, name) if (name %in% c("rS", "dD")) log(value) else qlogis(value)

#Stan data for model_hier.stan. free are the parameters that vary between patients, fixed the natural scale values of the rest
make_hier_stan_data <- function(patient_ids, free = hier_par_names, fixed = c(), lognormal_noise = TRUE, priors = hier_priors){
  stopifnot(setequal(c(free, names(fixed)), hier_par_names), !length(intersect(free, names(fixed))))
  free_idx  <- match(free, hier_par_names)
  fixed_unc <- setNames(rep(0, 6), hier_par_names)
  for (name in names(fixed)) fixed_unc[name] <- to_unconstrained(fixed[[name]], name)

  #segments, output times and observations of each patient, with output indices within the patient
  per_patient <- map(patient_ids, \(id) {
    patient_data <- load_zhang_patient(id)
    segments     <- make_segment_df(patient_data)
    output_times <- make_output_times(patient_data, segments)
    obs          <- patient_data |> filter(!is.na(PSA))
    list(segments = segments, output_times = output_times, psa = obs$PSA, obs_idx = match(obs$Time, output_times))
  })
  #first and last position of each patient in the concatenated arrays
  first_last <- function(lengths) list(first = cumsum(lengths) - lengths + 1, last = cumsum(lengths))
  segs <- first_last(map_int(per_patient, \(x) nrow(x$segments)))
  outs <- first_last(map_int(per_patient, \(x) length(x$output_times)))
  obs  <- first_last(map_int(per_patient, \(x) length(x$psa)))
  all_segments <- map_dfr(per_patient, "segments")

  list(n_patients      = length(patient_ids),
       n_segments      = nrow(all_segments),
       seg_start_time  = all_segments$segment_start_time,
       seg_treatment   = all_segments$Treatment,
       seg_from_idx    = all_segments$segment_from_idx,
       seg_to_idx      = all_segments$segment_to_idx,
       pat_seg_first   = segs$first,
       pat_seg_last    = segs$last,
       n_out           = sum(map_int(per_patient, \(x) length(x$output_times))),
       output_times    = unlist(map(per_patient, "output_times")),
       pat_out_first   = outs$first,
       pat_out_last    = outs$last,
       n_obs           = sum(map_int(per_patient, \(x) length(x$psa))),
       psa             = unlist(map(per_patient, "psa")),
       obs_idx         = unlist(map(per_patient, "obs_idx")),
       pat_obs_first   = obs$first,
       pat_obs_last    = obs$last,
       lognormal_noise = as.integer(lognormal_noise),
       n_free          = length(free),
       free_idx        = array(free_idx),
       fixed_unc       = unname(fixed_unc),
       mu_loc          = array(unname(priors$mu_loc[free_idx])),
       mu_scale        = array(priors$mu_scale[free_idx]),
       tau_scale       = array(priors$tau_scale[free_idx]),
       lkj_eta         = priors$lkj_eta,
       sigma_scale     = priors$sigma_scale)
}

#inits near the prior means, with every patient close to the population mean
hier_init <- function(stan_data){
  n_free <- stan_data$n_free
  list(mu      = array(stan_data$mu_loc + rnorm(n_free, 0, 0.3)),
       tau     = array(rep(0.5, n_free)),
       L_Omega = diag(n_free),
       z       = matrix(rnorm(n_free * stan_data$n_patients, 0, 0.3), n_free),
       sigma   = if (stan_data$lognormal_noise) 0.3 else 0.5)
}
