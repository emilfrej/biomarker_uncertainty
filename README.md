**Main goal**

Get a delta AT and eTTP posteriors I trust for the patients in "mathematical biomarker"" paper


**Questions**

what about likelihood around observations
How to go from raw PSA to measurements and back
Why 1.5 death rate
How do you go about this workflow. Maybe find a paper that does the same thing?
is k = .2 still used? can find it in the arxiv says, but cant find in the publication
What about n0


**stories you could tell**
how to compare different mathematical models of tumor trajectories for a single patient?
how certain can we get of Delta AT and eTTP after one round of treatment? Hierarchichal model.
how to get parameter values for building simulators of tumour trajectories. 
how identifiable is the lotka volterra model?
how to improve the lotka volterra model? what noise likelihood to assume around observations?
  - simulate values from normed / non_normed values and see how identifiable they are. 
  - relax fixed values / non-dimensionality
  
  
**Notes**

Maybe doing all of this in python would have been easier. Given all the work kit and strobl has already done. There would have been a lot of functions I could have reused. But then the stan stuff I'm familiar with in R.
currently passing starting vals directly which likely isn't ideal

  
  


