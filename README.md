**Main goal**

Find out how well we can calculate biomarkers using different scheduling approaches. 


**Project outline**

~~Get biomarker estimates to compare against.~~

~~Find out if Stan gets similar param values as biomarker study.~~

~~Get Stan working on a single patient~~

Simulate hierarchichal stan model 

SBC to validate
  
Hierarchical model from Zhang.

Simulate responses on different protocols for these params.

Compare against adaptive scheduler. 


**Questions**
How to go from raw PSA to measurements and back
How do you go about this workflow. Maybe find a paper that does the same thing?
is k = .2 still used? can find it in the arxiv says, but cant find in the publication
how much init reduction is too little

**stories you could tell**
how to compare different mathematical models of tumor trajectories for a single patient?
how to get parameter values for building simulators of tumour trajectories. 
how identifiable is the lotka volterra model?
how to improve the lotka volterra model? what noise likelihood to assume around observations?
  - simulate values from normed / non_normed values and see how identifiable they are. 
  - relax fixed values / non-dimensionality
  
**Notes**
