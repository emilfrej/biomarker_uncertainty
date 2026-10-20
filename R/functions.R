pacman::p_load(tidyverse, deSolve)

make_priors <- function(){
  priors <- list(
    cost_prior     = c(5, 10),
    turnover_prior = c(0, 1),
    n0_prior       = c(2, 2),
    rFrac_prior    = c(2, 30),
    K_prior        = c(2, 5), #uniform bounds
    rS_prior       = c(log(0.027), 0.05),
    dD_prior       = c(log(1.5), 0.5),
    sigma_prior    = c(-2, 1)
  )
  return(priors)
}

#get n best fits from Strobl fit
get_n_best_fits <- function(n){
  strobl_fit_results <- read_csv(here::here("AT_costOfResistance_LVModel/data/fits/4params/fitSummaryDf.csv"))

  top_n_fits <- strobl_fit_results |>
    arrange(desc(RSquared)) |>
    head(n) |>
    select(PatientId, RSquared, R0, S0, dR, dS, n0, rR, turnover, cost)
}

#params and starting state from strobl fit
get_params_and_state <- function(patient_id, strobl_fits){
  
  patient_row <- strobl_fits |> 
    filter(PatientId == patient_id)
  
  params <- c(rS = patient_row$rS, rR = patient_row$rR, d=patient_row$dR, K = patient_row$k, dD = patient_row$dD)
  starting_state = c(S=patient_row$S0, R=patient_row$R0)
  
  return(list(params=params, starting_state=starting_state))
}


#ODE system that describes tumour trajectory.
# Takes State = c(S, R) and Pars = c(rS, rR, K, dD, D, d)
lotka_voltera_ode <- function(Time, State, Pars){
  

  #construct a temporary list of state variables and state pars to evaluate nex lines
  with(as.list(c(State, Pars)), {

    dSdt <- rS*S * (1 - ((S+R) / K)) * (1 - dD*D) - d*S #derivative of sensitives
    dRdt <- rR*R * (1 - ((S+R) / K)) - d*R  #derivative of resistant
    return (list(c(dSdt, dRdt)))
  })
}

#define noise form for tumour size
noise_func <- function(N_vec, sigma) {
  rnorm(length(N_vec), mean=N_vec, sd = sigma)
}

#function for taking one step of forward patient trajectory and emitting a PSA signal
forward_sim_single_patient <- function(pars, starting_state, schedule, sigma){

  state = starting_state

  out_dfs <- list()

  # Loop over treatment schedule. Might need to vectorize **
  for (i in 1:nrow(schedule)) {

    #extract values needed for ode
    D = schedule$dose[i]

    #setup time grid for ode
    Time = seq(schedule$start[i], schedule$end[i], by = 1) #change ODE RES here

    pars = pars
    pars["D"] = D

    #calculate the ode for this segment
    ode_segment <- as.data.frame(ode(func = lotka_voltera_ode, y = state, parms = pars, times = Time))

    #add treatment col
    ode_segment$D <- rep(D, nrow(ode_segment))

    #reset state using last state
    state <- c(S = tail(ode_segment$S, 1), R = tail(ode_segment$R, 1))

    #store dfs
    if (i == nrow(schedule)){
      out_dfs[[i]] <- ode_segment
    } else{
      out_dfs[[i]] <- head(ode_segment, -1)
    }
  }

  # build dataframe
  out_df_combined <- bind_rows(out_dfs)

  # add noise around
  out_df_combined <- out_df_combined |>
    mutate(N = S + R,
           N_non_normalized = N / first(N), #divide by initial cell density
           PSA = noise_func(N_non_normalized, sigma))

  # return dataframe of SR
  return(out_df_combined)
}

#loads a df with patient level observations from trial for one patient
load_bruchovsky_patient <- function(patient_id){

    file_path <- here::here("dataTanaka/Bruchovsky_et_al", sprintf("patient%03d.txt", patient_id))

    #load in the df and clean
    read_csv(file_path, col_names = FALSE) |>
      rename(PatientID = 1, Date = 2, CPA = 3, LEU = 4, PSA = 5, Testosterone = 6, CycleId = 7, DrugConcentration = 8) |>
      mutate(Date = ymd_hms(Date),
             CPA  = parse_number(as.character(CPA)),
             LEU  = parse_number(as.character(LEU))) |>
      arrange(Date) |>
      mutate(Time    = X9 - first(X9), #elapsed time like strobl python loader
             PSA_raw = PSA,     #normalise
             PSA     = PSA_raw / first(PSA_raw),
             segment = consecutive_id(DrugConcentration)) #for solving ODE by segment rather than whole trajec in Stan
    
}
# 
#Returns an df of treatment segments Stan with rows for each segment start and end date, treatment and number of PSA measurements within segment
make_segment_df <- function(patient_data_df){
  #drop rows after last row with PSA
  patient_data_df <- patient_data_df |>
    filter(Time <= max(Time[!is.na(PSA)]))
  #make def where each row is start and end day of segment, and what treatment was given
  segments <- patient_data_df |> 
    group_by(segment) |> 
    summarize(segment_start_time = first(Time),
              Treatment= first(DrugConcentration)) |> 
    mutate(segment_end_time = lead(segment_start_time, default = max(patient_data_df$Time))) |> 
    # some patients have final visit where does changes, but no follow up obs, so a segment would be created but not ended. 
    filter(segment_end_time > segment_start_time)
  #count how many rowws with PSA obs fall inside each section
  counts <- patient_data_df |>
    mutate(solve_segment = lag(segment)) |>
    filter(!is.na(PSA), !is.na(solve_segment)) |>
    count(solve_segment, name = "num_psa_obs")
  #join the to get num psa obs in segment df
  segments <- segments |>
    left_join(counts, by = c("segment" = "solve_segment")) |>
    mutate(num_psa_obs = replace_na(num_psa_obs, 0L))
  #positions of each segment's first and last output time, for slicing output_times and ode_states in Stan
  output_times <- make_output_times(patient_data_df, segments)
  segments <- segments |>
    #get index of segments end time
    mutate(segment_to_idx   = match(segment_end_time, output_times),
           segment_from_idx = lag(segment_to_idx, default = 0) + 1)
 return(segments)
}

#times the ODE solver must report: every PSA day (to compare with data) and every segment end (to start the next segment from)
#sorted and without duplicates, since the solver needs strictly increasing times. Includes day 0 (start time trick in Stan)
make_output_times <- function(patient_data_df, segments){
  psa_times <- patient_data_df$Time[!is.na(patient_data_df$PSA)]
  sort(unique(c(psa_times, segments$segment_end_time)))
}


#loads a list of dfs, one per patient, named by patient id
load_bruchovsky_list <- function(patient_ids){
    out_list <- lapply(patient_ids, load_bruchovsky_patient)
    names(out_list) <- as.character(patient_ids)
    return(out_list)
}

#rebuilds treatment schedule from bruchowsky trial given a trial df
build_treatment_schedule <- function(patient_data) {

    df <- patient_data |> arrange(Time)

    df |>
      mutate(segment = consecutive_id(DrugConcentration)) |>
      group_by(segment) |>
      summarise(start = first(Time),
                dose  = first(DrugConcentration),
                .groups = "drop") |>
      mutate(end = lead(start, default = max(df$Time))) |>
      select(start, end, dose) |>
      filter(end > start) #drop zero-length segment when dose changes at last measurement

}

#function for simulating a set of patients
simulate_best_strobl_fits <- function(sigma, n){
  
  strobl_fits <- read_csv(here::here("AT_costOfResistance_LVModel/data/fits/4params/fitSummaryDf.csv"))
  
  top_ids <- strobl_fits |> 
    arrange(desc(RSquared)) |> 
    head(n) |> 
    pull(PatientId)
  
  clinical_data_list <- load_bruchovsky_list(top_ids)
  
  out_list <- list()
  
  # simulate data_sets for top_n best patients
  for (i in (1:n)){
    patient_id <- top_ids[i]
    state_params <- get_params_and_state(patient_id = patient_id, strobl_fits = strobl_fits)
    clinical_data <- clinical_data_list[[as.character(patient_id)]] # get the clinical data for id
    treatment_schedule <- build_treatment_schedule(clinical_data)
    #sim the trajectory and keep only timepoints in the actual data where a PSA measurement is present
    simulated_trajectory <- forward_sim_single_patient(pars=state_params$params, starting_state = state_params$starting_state, schedule = treatment_schedule, sigma = sigma) |>
      right_join(clinical_data, by = join_by(time==Time), suffix = c("_simulated","_observed")) |> 
      filter(!is.na(PSA_observed))
    out_list[[i]] <- simulated_trajectory
  }
  out_df = bind_rows(out_list)
  return(out_df)
}

#function for building a fit path from a patient Id
make_fit_path <- function(patient_id, dir) {
  file.path(dir, sprintf("fit_patient%03d.rds", patient_id))
}

#### BIOMARKER FUNCTIONS #####
## credit to Kit Gallagher (see A New Mathematical Biomarker paper, utils/myUtils.py)
calc_growth_time <- function(kappa, dR, K, rR, R0, upper = NULL, ...){
  if (is.null(upper)){
    upper <- 2 * R0
  }
  numerator <- upper * (rR * R0 + dR * K - (1 - kappa) * rR * K)
  denominator <- R0 * (upper * rR + dR * K - (1 - kappa) * rR * K)
  Time <- log(numerator / denominator) / (rR * (1 - kappa) - dR)
  return(Time)
}

calc_average_overshoot <- function(tau, n_crit, rS, dD, dR, K, ...){
  a <- rS * (1 - dD) - dR
  b <- rS * (1 - dD) / K
  denom <- b * n_crit - (b * n_crit - a) * exp(- a * tau)
  max_overshoot_pos <- (a * n_crit) / denom
  return((n_crit - max_overshoot_pos) / 2)
}

calc_critical_treatment_threshold <- function(tau, S0, K, rS, dS, prog = 1.2, ...){
  numerator <- K * (dS - rS)
  pre_exp <- (numerator / (prog * S0)) + rS
  denominator <- (pre_exp * exp(tau * (rS - dS))) - rS
  return(numerator / denominator)
}

calc_delta_benefit <- function(kappa, dR, K, rR, R0, ...){
  T_plus <- calc_growth_time(kappa, dR, K, rR, R0, ...)
  T_minus <- calc_growth_time(0, dR, K, rR, R0, ...)
  return((T_plus - T_minus) / T_minus)
}

calc_atx_benefit <- function(tau, n0, K, ...){
  n_crit <- calc_critical_treatment_threshold(tau, K = K, ...)
  n_overshoot <- calc_average_overshoot(tau, n_crit, K = K, ...)
  ave_n <- (n_crit + 1.2 * n0) / 2 - n_overshoot
  return(calc_delta_benefit(kappa = ave_n / K, K = K, ...))
}

predict_ct_ttp <- function(rR, dR, K, R0, n0, n_lim = NULL, ...){
  if (is.null(n_lim)){
    n_lim <- n0 * 1.2
  }
  factor <- rR - dR
  log_num <- rR - factor * (K / R0)
  log_den <- rR - factor * (K / n_lim)
  return(log(log_num / log_den) / factor)
}

predict_atx_ttp <- function(tau, ...){
  benefit <- calc_atx_benefit(tau, ...)
  ct_ttp <- predict_ct_ttp(...)
  return(ct_ttp * (1 + benefit))
}

#eq. (8) in Gallagher et al.: TRUE when progression under continuous therapy is possible (resistant equilibrium above 1.2 * n0)
#keeping only draws where this is TRUE gives the posterior conditioned on progression being possible
progression_possible <- function(cost, turnover, n0, K, prog = 1.2){
  turnover < (1 - prog * n0 / K) * (1 - cost)
}

### fitting ##
#make function to build fixed and non fit param values
make_fixed_val_data <- function(fixed_vals_input = list()){
  fixed_vals_input <- as.list(fixed_vals_input)
  #check if passed params correspond to params in stan model
  switchable_params <- c("cost", "turnover", "n0", "rFrac", "rS", "K", "dD")
  unknown_params <- setdiff(names(fixed_vals_input), switchable_params)
  if (length(unknown_params)) stop("not switchable: ", paste(unknown_params, collapse = ", ")) #complain on every unknown input
  #make flags so everything not in input is TRUE and fitted
  flags <- as.integer(!switchable_params %in% names(fixed_vals_input)) #stan takes 1/0
  flags <- setNames(as.list(flags), paste0("fit_", switchable_params))
  #set all to zero, and then fill in values from given list
  vals <- setNames(as.list(rep(0, length(switchable_params))), switchable_params)
  vals[names(fixed_vals_input)] <- fixed_vals_input
  vals <- setNames(vals, paste0(names(vals), '_fixed_val'))
  return(c(flags, vals))
}
  

## build the Stan data list for one patient
make_stan_data <- function(patient_id, first_cycle = FALSE, priors, fixed_vals = list(), run_diagnostics = 1, prior_only = 0) {
  
  patient_data <- load_bruchovsky_patient(patient_id)
  if (first_cycle) patient_data <- filter(patient_data, CycleId == 1)
  
  #make seperate dfs for treatment segment for the ODE Solver in stan and observation df for the likelihood to account for rows with no PSA but treatment info
  segments_df  <- make_segment_df(patient_data)
  output_times <- make_output_times(patient_data, segments_df)
  obs_df       <- patient_data |> filter(!is.na(PSA))
  
  
  #turn it all into list
  c(
    list(
      #data
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
    make_fixed_val_data(fixed_vals),
    prior_only = prior_only,
    priors,
    run_diagnostics = run_diagnostics)
}

# fit a set of patients and save one .rds (+ CSVs) per patient
fit_patients <- function(patient_ids=NULL,
                         stan_file,
                         priors,
                         fixed_vals  = list(dD = 1.5),
                         first_cycle = FALSE,
                         out_dir     = here::here("fits", paste0("free_rs_K_", if (first_cycle) "first_cycle" else "full_history")),
                         overwrite   = FALSE,
                         init        = NULL,
                         ...) {                       # extra args go to $sample(), e.g. chains, adapt_delta, seed
  ##if null fit all
  if (is.null(patient_ids)) {
    patient_ids <- list.files(here::here("dataTanaka/Bruchovsky_et_al"), pattern = "^patient\\d+\\.txt$") |>
      parse_number() |>
      sort()
  }
  
  
  mod <- cmdstan_model(stan_file)
  csv_dir <- file.path(out_dir, "csv")
  dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)
  
  message("model: ", basename(stan_file), " | out: ", out_dir,
          " | patients: ", paste(patient_ids, collapse = ", "), " | overwrite: ", overwrite)
  
  fits <- list()
  for (patient_id in patient_ids) {
    fit_path <- make_fit_path(patient_id, dir = out_dir)
    if (file.exists(fit_path) && !overwrite) { message("patient ", patient_id, ": exists, skipping"); next }
    
    stan_data <- make_stan_data(patient_id, first_cycle, priors, fixed_vals)
    
    if (stan_data$n_obs < 5) next
    
    fit <- tryCatch(
      mod$sample(data = stan_data, init = init,
                 output_dir = csv_dir, output_basename = sprintf("patient%03d", patient_id), ...),
      error = function(e) { message("patient ", patient_id, " failed: ", conditionMessage(e)); NULL }
    )
    if (is.null(fit)) next
    
    if (all(fit$return_codes() == 0)) {
      fit$save_object(fit_path)
    } else {
      message("patient ", patient_id, ": ", sum(fit$return_codes() != 0), " chain(s) failed, not saved")
    }
    fits[[as.character(patient_id)]] <- fit
  }
  invisible(fits)
}


### plots  ###
#plot posteriors and priors
plot_prior_posterior <- function(patient_id, n = 4000, kappa = .2, filter_steady_state) {
  first_fit <- readRDS(make_fit_path(patient_id, here::here("fits/segmented_first_cycle")))
  full_fit  <- readRDS(make_fit_path(patient_id, here::here("fits/segmented_full_history")))
  
  prior_draws <- tibble(
    cost     = rbeta(n, 5, 10),
    turnover = rbeta(n, 2, 2),
    n0       = rbeta(n, 2, 2),
    rFrac    = rbeta(n, 2, 30),
    K        = rlnorm(n, log(2), .4),
    rS       = rlnorm(n, log(0.027), .05),
    sigma    = rlnorm(n, -2, 1)
  ) |>
    filter(K > n0) #stan constrains K > n0
  
  params <- names(prior_draws)
  get_post <- function(fit) {
    fit$draws(params, format = "draws_df") |> as_tibble() |> select(all_of(params))
  }
  
  draws <- bind_rows(prior            = prior_draws,
                     `first cycle`    = get_post(first_fit),
                     `full history`   = get_post(full_fit),
                     .id = "dist") |>
    mutate(dist = factor(dist, c("prior", "first cycle", "full history")),
           #AT score from the same transforms as the stan model
           AT_score = calc_delta_benefit(kappa = kappa, dR = turnover * rS, K = K,
                                         rR = (1 - cost) * rS, R0 = rFrac * n0))
  
  #count posterior draws before and after filtering on eq. (8). The prior is left unfiltered
  n_before <- count(draws, dist, name = "before")
  if (filter_steady_state) draws <- filter(draws, dist == "prior" | turnover < (1-1.2*n0/K) * (1-cost) )
  n_after <- count(draws, dist, name = "after")
  draw_counts <- left_join(n_before, n_after, by = "dist") |> mutate(after = replace_na(after, 0L)) |>
    filter(dist != "prior")
  
  #number of divergent transitions in a fit
  n_div <- function(fit) sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent)
  #largest rhat across the plotted parameters
  max_rhat <- function(fit) round(max(fit$summary(params, "rhat")$rhat, na.rm = TRUE), 3)
  
  fill_colours <- c("prior" = "grey70", "first cycle" = "#2a78d6", "full history" = "#eb6834")
  subtitle_text <- paste0(if (filter_steady_state) "Draws kept after steady state filter: " else "Draws: ",
                          paste(draw_counts$dist, paste(draw_counts$after, "of", draw_counts$before), collapse = ", "),
                          "\nDivergences: first cycle ", n_div(first_fit), ", full history ", n_div(full_fit),
                          "\nMax Rhat: first cycle ", max_rhat(first_fit), ", full history ", max_rhat(full_fit))
  
  #parameters
  params_plot <- draws |>
    select(-AT_score) |>
    pivot_longer(-dist, names_to = "parameter") |>
    ggplot(aes(value, fill = dist)) +
    geom_density(alpha = .5, colour = NA) +
    facet_wrap(~ parameter, scales = "free") +
    scale_fill_manual(values = fill_colours) +
    theme_classic() +
    labs(title = paste("Prior and posteriors, patient", patient_id), subtitle = subtitle_text, fill = NULL)
  
  #AT score, with the paper's value as a dashed line. Non-finite scores and scores outside [-4, 4] are left out
  at_plot <- draws |>
    filter(between(AT_score, -4, 4)) |>
    ggplot(aes(AT_score, fill = dist)) +
    geom_density(alpha = .5, colour = NA) +
    geom_vline(data = filter(paper_at_scores, patient == patient_id), aes(xintercept = AT_score), linetype = "dashed") +
    scale_fill_manual(values = fill_colours) +
    coord_cartesian(xlim = c(-2, 2)) +
    theme_classic() +
    labs(title = paste("AT score, patient", patient_id),
         subtitle = subtitle_text)
  
  list(params = params_plot, at = at_plot)
}
