from experiment import *
import math
import pandas as pd 

sizes = np.round(np.logspace(1, math.log(2500, 10), 10)).astype(int)  # definition of the networks' sizes
results = []

for size in sizes:

    num_perturb = int(0.2 * size)  # definition of the number of perturbations applied to all the base patterns
    successful_t_values = []  # initialization of the list that will be containing the integers representing the number 
    # of patterns where we have a system's convergence
    unsuccessful_t_values = []  # initialization of the list that wil be containing the integers representing the number
    # of patterns where we don't have a system's convergence

    weight_rule = "hebbian"
    capacity_h = size/(2*math.log(size))  # computation of the capacity with the Hebbian rule
    num_patterns_h = np.linspace(0.5 * capacity_h, 2 * capacity_h, 10).astype(int)  # list containing 10 different
    # integers determined with the capacity of the network and representing the number of patterns in the matrix of 
    # random patterns

    results.append(experiment(size, num_patterns_h, weight_rule, num_perturb, successful_t_values,
                              unsuccessful_t_values))  # storing the results' dictionary determined with the Hebbian 
    # rule of the experiment in a list

    # comparing with a tolerance of 10% if the asymptotic bound is a good estimation of our experimental capacity
    if successful_t_values:
        comparison_asymptotic_estimate_and_experimental_capacity(size, weight_rule, max(successful_t_values), capacity_h)
    else:
        print(f"No patterns converged within the 10 trials with the size {size}, the number of patterns {num_patterns_h} "
              f"and the learning rule {weight_rule}.")

    # re-initialization of the two lists to only have the values of the number of patterns determined with the Storkey 
    # rule's capacity
    successful_t_values = []
    unsuccessful_t_values = []

    weight_rule = "storkey"
    capacity_s = size / math.sqrt((2 * math.log(size)))  # computation of the capacity with the Storkey rule
    num_patterns_s = np.linspace(0.5 * capacity_s, 2 * capacity_s, 10).astype(int)  # list containing 10 different
    # integers determined with the capacity of the network and representing the number of patterns in the matrix of 
    # random patterns

    results.append(experiment(size, num_patterns_s, weight_rule, num_perturb, successful_t_values,
                              unsuccessful_t_values))  # storing the results' dictionary determined with the Storkey 
    # rule of the experiment in a list

    # comparing with a tolerance of 10% if the asymptotic bound is a good estimation of our experimental capacity
    if successful_t_values:
        comparison_asymptotic_estimate_and_experimental_capacity(size, weight_rule, max(successful_t_values), capacity_h)
    else:
        print(f"No patterns converged within the 10 trials with the size {size}, the number of patterns "
              f"{num_patterns_s} and the learning rule {weight_rule}.")
    print()

# saving capacity curves of our experiments
for element in results:
    plot_capacity_curve(element["network_size"], element["weight_rule"], element["num_patterns"], element["match_frac"])

# splitting the results' list into two lists: one containing the results for the
heb_results = results.copy()[0::2]
sto_results = results.copy()[1::2]

# saving two plots with our empirical capacity curves including number of neurons vs. capacity for both learning rules
save_empirical_capacity(heb_results, "hebbian")
save_empirical_capacity(sto_results, "storkey")



#creating a panda DataFrame from our results dictionary
df = pd.DataFrame(results)

#saving the dataframe as an hdf5 file
df.to_hdf("./results_panda_dataframe", key='df')

#panda prints the table in mardown format
print(df.tomarkdown())
