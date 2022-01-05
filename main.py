from experiment import *
import math
import timeit

sizes = np.round(np.logspace(1, math.log(2500, 10), 10)).astype(int)  # definition of the networks' sizes
weight_rule = ""  # initialization of weight_rule variable

results = []
for size in sizes:
    
    num_perturb = int(0.2 * size)  # definition of the number of perturbations applied to all the base patterns
    successful_t_values = []  # initialization of the list containing the values of t where we have convergence
    unsuccessful_t_values = []  # initialization of the list containing the values of t where we don't have convergence

    weight_rule = "hebbian"
    capacity_h = size/(2*math.log(size))  # computation of the capacity with the Hebbian rule
    num_patterns_h = np.linspace(0.5 * capacity_h, 2 * capacity_h, 10).astype(int)  # list containing 10 different
    # numbers of random patterns according to the number of neurons

    start_h = timeit.default_timer()
    results_experiment_h = experiment(size, num_patterns_h, weight_rule, num_perturb, successful_t_values,
                                      unsuccessful_t_values)  # applying the experiment with the Hebbian rule
    stop_h = timeit.default_timer() - start_h
    print(f"Timer Hebbian {size}: {stop_h}")

    # storing the result of the experiment with the Hebbian rule in a list
    results.append(results_experiment_h)

    # comparing with a tolerance of 10% if the asymptotic bound is a good estimation of our experimental capacity
    comparison_asymptotic_estimate_and_experimental_capacity(size, weight_rule, max(successful_t_values), capacity_h)

    successful_t_values = []
    unsuccessful_t_values = []

    weight_rule = "storkey"
    capacity_s = size / math.sqrt((2 * math.log(size)))  # computation of the capacity with the Storkey rule
    num_patterns_s = np.linspace(0.5 * capacity_s, 2 * capacity_s, 10).astype(int)

    start_s = timeit.default_timer()
    results_experiment_s = experiment(size, num_patterns_s, weight_rule, num_perturb, successful_t_values,
                                      unsuccessful_t_values)  # applying the experiment with the Storkey rule
    stop_s = timeit.default_timer() - start_s
    print(f"Timer Storkey {size}: {stop_s}")

    # storing the result of the experiment with the Storkey rule in a list
    results.append(results_experiment_s)

    # comparing with a tolerance of 10% if the asymptotic bound is a good estimation of our experimental capacity
    comparison_asymptotic_estimate_and_experimental_capacity(size, weight_rule, max(successful_t_values), capacity_s)
    print()

# saving capacity curves of our experiments
for element in results:
    plot_capacity_curve(element["network_size"], element["weight_rule"], element["num_patterns"], element["match_frac"])

# splitting the results' list into two lists: one containg the results for the
heb_results = results.copy()[0::2]
sto_results = results.copy()[1::2]

# saving two plots with our empirical capacity curves including number of neurons vs. capacity for both learning rules
save_empirical_capacity(heb_results, "hebbian")
save_empirical_capacity(sto_results, "storkey")
