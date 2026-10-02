#script for simulating a data set of single patients

#functionget top n best fits from strobl analysis according to r^2
get_n_best_fits <- function(n){
  strobl_fit_results <- read_csv("../AT_costOfResistance_LVModel/data/fits/4params/fitSummaryDf.csv") 
  
  top_n_fits <- strobl_fit_results |> 
    arrange(desc(RSquared)) |> 
    head(n) |> 
    select(PatientId, RSquared, R0, S0, dR, dS, n0, rR, turnover, cost)
}

#set seed
set.seed(1)

n_patients <- 3

param_df <- get_n_best_fits(n_patients)

#function for getting a param for single patient and their treatment schedule




#give them treatment schedules


#generate patient data

 # make sure folders exist

 # make schedule

 # loop over patients, run ODE system
 # check that python agrees
 # 

#check that python sim agrees


