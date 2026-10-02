#script for simulating a data set of single patients

n_best_patients <- 3

### Helper Functions ###

#function to get top n best fits from strobl analysis according to r^2
get_n_best_fits <- function(n){
  strobl_fit_results <- read_csv("../AT_costOfResistance_LVModel/data/fits/4params/fitSummaryDf.csv") 
  
  top_n_fits <- strobl_fit_results |> 
    arrange(desc(RSquared)) |> 
    head(n) |> 
    select(PatientId, RSquared, R0, S0, dR, dS, n0, rR, turnover, cost)
}

#make a list of dataframes with bruchowsky data from vec of patient IDs
load_bruchowsky_patients <- function(patient_ids){
  
  out_list <- list()
  for (id in patient_ids){
    
    file_path <- paste0("dataTanaka/Bruchovsky_et_al/", sprintf("patient%03d.txt", id))
    
    #load in the df and clean
    patient_data <- read_csv(file_path, col_names = FALSE) |>
      rename(PatientID = 1, Date = 2, CPA = 3, LEU = 4, PSA = 5, Testosterone = 6, CycleId = 7, DrugConcentration = 8) |>
      mutate(Date = ymd_hms(Date)) |>
      arrange(Date) |>                                
      mutate(Time    = X9 - first(X9), #elapsed time like strobl python loader
             PSA_raw = PSA,     #normalise
             PSA     = PSA_raw / first(PSA_raw))
    
    #store the whole df as one list element
    out_list[[as.character(id)]] <- patient_data
  }
  
  return(out_list)
}

#build schedule from PatientID 
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

### SIMULATOR FUNCTIONS ###
#define noise form for tumour size
noise_func <- function(N_vec) {
  rnorm(length(N_vec), mean=N_vec, sd= sigma) 
}

#function for taking one step of forward patient trajectory and emitting a PSA signal
forward_sim_single_patient <- function(pars, starting_state, schedule){
  
  state = starting_state
  
  out_dfs <- list()
  
  # Loop over treatment schedule. Might need to vectorize **
  for (i in 1:nrow(schedule)) {
    
    #extract values needed for ode
    D = schedule$dose[i]
    
    #setup time grid for ode
    Time = seq(schedule$start[i], schedule$end[i], by = ode_res)
    
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
           PSA = noise_func(N_non_normalized))
  
  # return dataframe of SR
  return(out_df_combined)
}

#set seed
set.seed(1)

#get best fitted patients
top_patients <- get_n_best_fits(3)

#for patient id
## for patient id in best


