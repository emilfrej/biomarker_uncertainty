To do
Calc eTTP

Make hierarchical model

Show priors are sensible 
Show SBC doesn't imply I'm sampling from a different posterior
Show how sensitive inference is to prior

Find out why model breaks.
Find out when it does so.

Main goal:
Get a delta AT and eTTP posteriors I trust for the patients in "mathematical biomarker"" paper

Generative assumptions for simulator:
Treatment decisions are taken directly from trial.
Missingness is not random / measurement process is not modelled. Neither in forward simulator

Treatment protocol: https://acsjournals.onlinelibrary.wiley.com/doi/10.1002/cncr.21989

Questions
what about likelihood around observations
How to go from raw PSA to measurements and back
Why 1.5 death rate
How do you go about this workflow. Maybe find a paper that does the same thing?
is k = .2 still used? can find it in the arxiv says, but cant find in the publication

#
stories you could tell:
- how to compare different mathematical models of tumor trajectories for a single patient?
- how certain can we get of Delta AT and eTTP after one round of treatment? Hierarchichal model.
- how to get parameter values for building simulators of tumour trajectories. 
- how identifiable is the lotka volterra model?
- how to improve the lotka volterra model? what noise likelihood to assume around observations?
  - simulate values from normed / non_normed values and see how identifiable they are. 
  - relax fixed values / non-dimensionality
  
Notes:
Maybe doing all of this in python would have been easier. Given all the work kit and strobl has already done. There would have been a lot of functions I could have reused. But then the stan stuff I'm familiar with in R.


  
  


