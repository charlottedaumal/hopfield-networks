from functions import *
from classes import *
import math


def experiment(size, num_patterns, weight_rule, num_perturb, successful_t_values, unsuccessful_t_values,
               num_trials=10, max_iter=100):

    # definition of the dictionary
    results_dict = {"network_size": [], "weight_rule": [], "num_patterns": [],
                    "num_perturb": [], "match_frac": []}

    # initialization of the network size, the weight rule and the number of perturbation in the dictionary
    # as they remain constant along the experiment
    results_dict["network_size"].append(size)
    results_dict["weight_rule"].append(weight_rule)
    results_dict["num_perturb"].append(num_perturb)

    for i in range(num_trials):  # for-loop to perform 10 run of the dynamical evolution system
        patterns = generate_patterns(num_patterns[i], size)  # definition of a matrix of random patterns
        network = HopfieldNetwork(patterns, weight_rule)  # definition of an HopfieldNetwork instance
        # + computation of the weights matrix according to the learning rule done automatically
        # inside the class instance
        saver = DataSaver()

        nb_rows = np.linspace(0, patterns.shape[0]-1, patterns.shape[0], dtype=int)  # list containing all the indices
        # for the rows of the random patterns matrix
        perturbed_patterns = [perturb_pattern(patterns[k], num_perturb) for k in nb_rows]  # perturbing all patterns of
        # the random pattern matrix and storing them in a list
        [network.dynamics(perturbed_state, saver, max_iter) for perturbed_state in perturbed_patterns]  # running the
        # dynamical evolution system on all perturbed patterns
        network_evolution = saver.get_data()["state"]  # accessing the list containing all the evolutions
        # of each perturbed pattern

        convergence_list = [pattern_match(patterns[k], network_evolution[k][-1]) for k in nb_rows]  # definition of a
        # convergence list containing whether the index if the evolved pattern has matched with one initial pattern
        # or a NoneType value
        convergence_nb = len(convergence_list) - convergence_list.count(None)  # computation of the total number of
        # patterns that converged
        convergence_fraction = convergence_nb / patterns.shape[0]  # computation of the convergence fraction
        if convergence_fraction >= 0.9:  # determining if the system has successfully converged + storing the
            # number of patterns in the corresponding list
            successful_t_values.append(num_patterns[i])
        else:
            unsuccessful_t_values.append(num_patterns[i])

        # storing the number of patterns and the convergence fraction in the dictionnary
        results_dict["num_patterns"].append(num_patterns[i])
        results_dict["match_frac"].append(convergence_fraction)

    return results_dict


def comparison_asymptotic_estimate_and_experimental_capacity(size, weight_rule, experimental_capacity,
                                                             asymptotic_estimate):

    # comparing the experimental network capacity to the theoretical asymptotic estimate with a chose tolerance
    if math.isclose(experimental_capacity, asymptotic_estimate, rel_tol=0.1):
        print(f"The asymptotic bound for the {weight_rule} learning rule is a good estimation of the capacity for "
              f"a network's size of {size} (tolerance = 10%).")
    else:
        print(f"The asymptotic bound for the {weight_rule} learning rule is a rough estimation of the capacity for"
              f" a network's size of {size} (tolerance = 10%).")


def plot_capacity_curve(size, weight_rule, num_patterns, match_frac):

    # formatting of the figure
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1.1)
    plt.xticks(num_patterns, num_patterns)
    plt.xlim(min(num_patterns)-0.1, max(num_patterns) + 0.1)
    plt.xlabel("Number of patterns")
    plt.ylabel("Fraction of retrieved patterns")
    plt.title(f"Capacity curve for a network of size {size} with the {weight_rule} rule")

    # plot points on the figure
    plt.scatter(num_patterns, match_frac)  #

    # save the figure in the current directory and close it afterwards
    plt.savefig(f"Size{size}_Rule{weight_rule}_CapacityCurve", format="jpg")
    plt.close()


def plot_empirical_capacity(size, num_patterns, match_frac, color):

    max_nb_patterns = 0
    for j in range(len(match_frac)):  # determine for which higher number of patterns we have a system's convergence
        if match_frac[j] >= 0.9 and num_patterns[j] >= max_nb_patterns:
            max_nb_patterns = num_patterns[j]

    # plot points
    plt.scatter(max_nb_patterns, size, c=color)

    return max_nb_patterns


def save_empirical_capacity(results, rule):

    # formatting the figure
    plt.figure(figsize=(10, 7))
    plt.xlabel("Maximum Empirical Capacity")
    plt.ylabel("Number of Neurons")
    plt.title("Empirical Capacity Curve including Number of Neurons vs. Capacity")
    color = "black"

    # plot points for each size of the network + store the maximum empirical capacity and the networks' sizes in a list
    max_nb_patterns = []
    sizes = []
    for i in range(len(results)):
        sizes.append(results[i]["network_size"])
        max_nb_patterns.append(plot_empirical_capacity(results[i]["network_size"], results[i]["num_patterns"],
                               results[i]["match_frac"], color))

    # plot a curve connecting all the points
    plt.plot(max_nb_patterns, sizes)

    # save the figure in the current directory and close it afterwards
    plt.savefig(f"Empirical_Capacity_{rule}", format="jpg")
    plt.close()
