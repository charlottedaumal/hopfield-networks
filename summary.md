
# BIO-210-team-26

## Report of v7 of the Hopfield Network project

### Aim of the project
The aim of the project is to simulate the evolution of a dynamical system for associative memory, the Hopield Network. It models a simple mechanism to explain how a neural network could store representations in the form of weighted connections between neurons.

Here, we implemented the iterative process which allows to retrieve one of the stored/memorized patterns starting from the representation of a new pattern.

In this report, we will explain the experiment we have performed and the results we obtained.

### The experiment
In this final version of the project, the aim was to empirically estimate the capacity of the Hopfield network, trained with the Hebbian and Storkey rules, and simulated with the synchronous update rule.

For this experiment, we considered 10 networks of size ranging from 10 to 2500. Then, for each network size, we ran 10 trials of a defined experiment with different initial patterns. We could estimate in this way the network capacity as a function of the network size and compare it with the theoretical asymptotic estimate.

The 10 trials were ran on an function called 'experiment'. For each size of network, we did the following steps 10 times: we generated a random number of patterns, then we perturbed 20% of each pattern and we checked if the k-ieth element of each perturbed pattern matched with the original one.
This gives us a fraction of convergence, which is the number of patterns retrieved over the total number of patterns for one single network. If this fraction of convergence is above 0.9, we store the number of patterns.

Finally, we returned a dictionary for each network size containing the following keys: "network_size", "weight_rule", "num_patterns", "num_perturb" and "" (for the 10 trials). 

These dictionaries are saved in a list called 'results' and is converted to a pandas dataframe.

Regarding our implementation, you can see that the even indices of this list correspond to the results calculated with the Hebbian rule, whereas the odd indices correspond to those calculated with the Storkey rule.

Below you can see the capacity curves for each size of simulated network and a given learning rule, plotting the fraction of retrieved patterns vs. the number of patterns in the network.

We decided to plot our figures with dots instead of a curve because we assumed it was more clear to read and less ambiguous as we are dealing with discrete data.


### </ Plots with Hebbian rule>

![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B10%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B18%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B34%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B63%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B116%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B215%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B397%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B733%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B1354%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B2500%5D_Rule%5B'hebbian'%5D_CapacityCurve.png)


### </Plots with Storkey rule>

![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B10%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B18%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B34%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B63%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B116%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B215%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B397%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B733%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B1354%5D_Rule%5B'storkey'%5D_CapacityCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B2500%5D_Rule%5B'storkey'%5D_CapacityCurve.png)


We can see on the plots for sizes 10,18 and 34 that we can have 2 values for the fraction of retrieved patterns for one single number of patterns. This is because we sometimes have twice the same number of patterns. 


We also plotted a summary capacity plot showing the number of patterns that can be retrieved with a propbability higher than 0.9 vs. the number of neurons for both learning rules:

### plot with Hebbian rule
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Empirical_Capacity_hebbian.png)
### plot with Storkey rule
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Empirical_Capacity_storkey.png)

Finally, we saved robustness curves for our experiments for each size of simulated network and learning rule, plotting the fraction of retrieved patterns vs. the number of perturbations.

### Robustness plots with Hebbian rule

![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B10%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B18%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B34%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B63%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B116%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B215%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B397%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B733%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B1354%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B2500%5D_Rule%5B'hebbian'%5D_RobustnessCurve.png)


### Robustness plots with Storkey rule

![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B10%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B18%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B34%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B63%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B116%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B215%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B397%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B733%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B1354%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)
![alt text](https://github.com/EPFL-BIO-210/BIO-210-team-26/blob/main/Graphs/Size%5B2500%5D_Rule%5B'storkey'%5D_RobustnessCurve.png)








### Credits

ClemenceKiehl & charlottedaumal

