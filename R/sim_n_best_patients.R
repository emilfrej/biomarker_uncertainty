### For quickly plotting simulated norm PSA against actual values for best fitting patients according to Strobl et al's analysis

source("R/functions.R")
set.seed(1)

#number of best fits from strobl analysis
n <- 10

#set gaussian noise
sigma <- 0.025 

#run func
sim_data <- simulate_best_strobl_fits(n=n, sigma=sigma) |> 
  pivot_longer(cols = c(PSA_simulated, PSA_observed), names_to="Metric", values_to="PSA")

#build treatment schedules
blocks <- sim_data |>
  distinct(PatientID, Time = time, DrugConcentration) |>
  group_by(PatientID) |>
  group_modify(~ build_treatment_schedule(.x)) |>
  filter(dose > 0)


#plot it
ggplot(sim_data, aes(x = time, y = PSA, color = Metric)) +
  geom_rect(data = blocks, aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "grey75", alpha = 0.4) +
  geom_point() +
  theme_minimal() +
  ylim(-0.1, 2) +
  facet_wrap(~PatientID)





