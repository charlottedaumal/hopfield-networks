
# BIO-210-team-26

## Report of v7 of the Hopfield Network project

### Aim of the project
The aim of the project is to simulate the evolution of a dynamical system for associative memory, the Hopield Network. It models a simple mechanism to explain how a neural network could store representations in the form of weighted connections between neurons.

Here, we implemented the iterative process which allows to retrieve one of the stored/memorized patterns starting from the representation of a new pattern.

In this report, we will explain the experiment we have performed and the results we obtained.

### The experiment
In this final version of the project, we empirically estimated the capacity of the Hopfield network, trained with the Hebbian and Storkey rules, and simulated with the synchronous update rule.

For this experiment, we considered 10 networks of size ranging from 10 to 2500. Then, for each network size, we ran 10 trials of a defined experiment with different initial patterns. We could estimate in this way the network capacity as a function of the network size and compare it with the theoretical asymptotic estimate.

The 10 trials were ran on an function called 'experiment'. For each size of network, we did the following steps 10 times: we generated a random number of patterns with the command 'linspace', then we perturbed 20% of each pattern and we checked if the k-ieth element of each perturbed pattern matched with the original one.
This gives us a fraction of convergence, which is the number of patterns retrieved over the total number of patterns for one single network. If this fraction of convergence is above 0.9, we store the number of patterns.

Finally, we return a dictionary for each network size containing... (for the 10 trials). 

These dictionaries are saved in a list called 'results' and is converted to a pandas dataframe.

Below you can see the capacity curves for each size of simulated network and learning rule, plotting the fraction of retrieved patterns vs. the number of patterns in the network.


#### Plots with Hebbian rule

![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)


#### Plots with Storkey rule

![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)


We also plotted a summary capacity plot showing the number of patterns that can be retrieved with a propbability higher than 0.9 vs. the number of neurons:

![alt text](lien image)


Finally, we saved robustness curves for our experiments for each size of simulated network and learning rule, plotting the fraction of retrieved patterns vs. the number of perturbations.

#### Robustness Plots with Hebbian rule

![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)


#### Robustness plots with Storkey rule

![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)
![alt text](lien image)

