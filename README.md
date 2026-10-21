**Main goal**

Find out how well we can calculate biomarkers using different scheduling approaches. 


**Project outline**

~~Get biomarker estimates to compare against. ~~

~~Find out if Stan gets similar param values as biomarker study. ~~

Simulate AT schedules.
- Get hierarchichal model to run
- Decide on prior
- SBC to validate
- Show how well biomarkers are calculated
  
Get param estimates for a few patients using hierarchical model from Zhang.
- decide on hierarchichal fit or population level prior
- show adequacy of model
- show sensitivity of hyper priors

Simulate responses on different protocols for these params.

Compare against adaptive scheduler. 


**Questions**
what about likelihood around observations
How to go from raw PSA to measurements and back
Why 1.5 death rate
How do you go about this workflow. Maybe find a paper that does the same thing?
is k = .2 still used? can find it in the arxiv says, but cant find in the publication
how much init reduction is too little


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
currently passing starting vals directly which likely isn't ideal.
need a check on what params are free and what not 




