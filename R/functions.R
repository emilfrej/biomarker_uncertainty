
#get n best fits from Strobl fit
get_n_best_fits <- function(n){
  strobl_fit_results <- read_csv(here::here("AT_costOfResistance_LVModel/data/fits/4params/fitSummaryDf.csv"))

  top_n_fits <- strobl_fit_results |>
    arrange(desc(RSquared)) |>
    head(n) |>
    select(PatientId, RSquared, R0, S0, dR, dS, n0, rR, turnover, cost)
}


#ODE system that describes tumour trajectory
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

#loads a list of dfs with patient level observations from trial
load_bruchowsky_patients <- function(patient_ids){

    out_list <- list()
    for (id in patient_ids){

      file_path <- here::here("dataTanaka/Bruchovsky_et_al", sprintf("patient%03d.txt", id))

      #load in the df and clean
      patient_data <- read_csv(file_path, col_names = FALSE) |>
        rename(PatientID = 1, Date = 2, CPA = 3, LEU = 4, PSA = 5, Testosterone = 6, CycleId = 7, DrugConcentration = 8) |>
        mutate(Date = ymd_hms(Date),
               CPA  = parse_number(as.character(CPA)),  
               LEU  = parse_number(as.character(LEU))) |>
        arrange(Date) |>
        mutate(Time    = X9 - first(X9), #elapsed time like strobl python loader
               PSA_raw = PSA,     #normalise
               PSA     = PSA_raw / first(PSA_raw))

      #store the whole df as one list element
      out_list[[as.character(id)]] <- patient_data
    }

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
      select(start, end, dose)

}

#function for simulating a set of patients
simulate_patients <- function(best_fit_params, sigma){

  #get bruchowsky patients
  patient_data_list <- load_bruchowsky_patients(best_fit_params$PatientId)

  out_list <- list()

  #loop over all entries in in best fits
  for (i in 1:nrow(best_fit_params)){
    params <- c(rS = 0.027, rR = best_fit_params$rR[i], d = best_fit_params$dS[i], dD = 1.5, K = 1)
    starting_state <- c(S = best_fit_params$S0[i], R = best_fit_params$R0[i])
    PatientId <- best_fit_params$PatientId[i]
    patient_data <- patient_data_list[[as.character(PatientId)]] #get right data from list of bruchowsky dfs
    treatment_schedule <- build_treatment_schedule(patient_data)

    #simulate and store
    simulated_trajectory <- forward_sim_single_patient(pars=params, starting_state = starting_state, schedule = treatment_schedule, sigma = sigma) |>
      right_join(patient_data, by = join_by(time==Time), suffix = c("_simulated","_observed")) #only keeps obs for when observations were made

    out_list[[i]] <- simulated_trajectory

  }

  out_df = bind_rows(out_list)
  return(out_df)

}
