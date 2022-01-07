from functions import *
from classes import *
import math


def experiment(size, num_patterns, weight_rule, num_perturb, successful_t_values, unsuccessful_t_values,
               num_trials=10, max_iter=100):
    """Runs 10 trials for each network size by running the dynamical system varying the initial pattern and perturbing
    20% of the values of one of the original patterns.

    Parameters:
    --------------
    size : int
    -> size of the network used for the experiment ( = size of the patterns)
    num_patterns: int
    -> number of patterns within the network used for the experiment
    weight_rule: string
    -> learning rule that will be used for the calculations during the experiment ("Hebbian" or "Storkey")
    num_perturb: int
    -> number of perturbations that will be applied to one pattern of the network
    successful_t_values: list of ints
    -> list containing the values of t where we have convergence
    unsuccessful_t_values: list of ints
    ->list containing the values of t where we don't have convergence
    num_trials: int
    -> number of times we will repeat the experiment (here num_trials = 10)
    max_iter: int
    -> maximum of iterations used for the call of function "dynamics" (here max_iter = 100)

    Output:
    --------------
    returns a dictionary called "results_dict" which has the following keys : "network_size", "weight_rule",
    "num_patterns", "num_perturb", "match_frac".

    CU: size >= 0, num_patterns >=0, weight_rule = "Hebbian" or weight_rule = "Storkey",  num_perturb >=0,
    successful_t_values >=0, unsuccessful_t_values >=0, num_trials >=0 and max_iter >=0
    """

    # definition of the dictionary
    results_dict = {"network_size": [], "weight_rule": [], "num_patterns": [],
                    "num_perturb": [], "match_frac": []}

    # initialization of the network size, the weight rule and the number of perturbation in the dictionary
    # as they remain constant along the experiment
    results_dict["network_size"].append(size)
    results_dict["weight_rule"].append(weight_rule)
    results_dict["num_perturb"].append(num_perturb)

    for num_pattern in num_patterns:
        patterns = generate_patterns(num_pattern, size)  # definition of a matrix of random patterns
        network = HopfieldNetwork(patterns, weight_rule)  # definition of an HopfieldNetwork instance
        # + computation of the weights matrix according to the learning rule done automatically
        # inside the class instance
        saver = DataSaver()

        convergence_nb = 0
        for j in range(num_trials):
            index_perturbed = rd.randint(0, patterns.shape[0]-1)
            perturbed_pattern = perturb_pattern(patterns[index_perturbed], num_perturb)  # perturbing one random pattern
            # of the random pattern matrix
            network.dynamics(perturbed_pattern, saver, max_iter)  # running the dynamical evolution system on all
            # perturbed patterns
            network_evolution = saver.get_data()["state"]  # accessing the list containing all the evolutions
            # of each perturbed pattern

            if pattern_match(patterns, network_evolution[-1]) == index_perturbed:
                convergence_nb += 1

        convergence_fraction = (convergence_nb / num_trials)  # computation of the convergence fraction
        if convergence_fraction >= 0.9:  # determining if the system has successfully converged + storing the
            # number of patterns in the corresponding list
            successful_t_values.append(num_pattern)
        else:
            unsuccessful_t_values.append(num_pattern)

        # storing the number of patterns and the convergence fraction in the dictionary
        results_dict["num_patterns"].append(num_pattern)
        results_dict["match_frac"].append(convergence_fraction)

    return results_dict


def comparison_asymptotic_estimate_and_experimental_capacity(size, weight_rule, experimental_capacity,
                                                             asymptotic_estimate):
    """Compares the experimental network capacity to the theoretical asymptotic estimate.

    Parameters:
    --------------
    size: int
    -> size of the network (= size of the patterns of the network)
    weight_rule: string
    -> learning rule that will be used for the calculations ("Hebbian" or "Storkey")
    experimental_capacity:
    -> maximum of the list successful_t_values
    asymptotic_estimate:
    -> asymptotic bound for the number of patterns that a Hopfield network can store (different depending on the 
    learning rule we use)

    Output:
    --------------
    prints :
            - "f"The asymptotic bound for the {weight_rule} learning rule is a good estimation of the capacity for "
              f"a network's size of {size} (tolerance = 10%)."
                           -> if the experimental network capacity and the theoretical asymptotic estimate are the same,
                           within a relative tolerance of 0.1

            - "f"The asymptotic bound for the {weight_rule} learning rule is a rough estimation of the capacity for"
              f" a network's size of {size} (tolerance = 10%)."
                         -> if the experimental network capacity and the theoretical asymptotic estimate are different,
                           within a relative tolerance of 0.1

    CU: size >=0, weight_rule = "Hebbian" or weight_rule = "Storkey", experimental_capacity >=0, asymptotic_estimate >=0
    """
    
    # comparing the experimental network capacity to the theoretical asymptotic estimate with a chosen tolerance
    if math.isclose(experimental_capacity, asymptotic_estimate, rel_tol=0.1):
        print(f"The asymptotic bound for the {weight_rule} learning rule is a good estimation of the capacity for "
              f"a network's size of {size} (tolerance = 10%).")
    else:
        print(f"The asymptotic bound for the {weight_rule} learning rule is a rough estimation of the capacity for"
              f" a network's size of {size} (tolerance = 10%).")


def plot_capacity_curve(size, weight_rule, num_patterns, match_frac):
    """Plots the capacity curve for a given size of simulated network and learning rule.

    Parameters:
    --------------
    size: int
    -> size of the network (= size of the patterns of the network)
    weight_rule: string
    -> learning rule that will be used for the calculations ("Hebbian" or "Storkey")
    num_patterns: int
    -> number of patterns within the network used for the experiment
    match_frac: floating point
    -> fraction of convergence, which is the number of patterns retrieved over the total number of patterns for
    the network

    Output:
    --------------
    capacity curve with the number of patterns on the x-axis and the fraction of retrieved patterns on the y-axis
    
    CU: size >= 0, weight_rule = "Hebbian" or weight_rule = "Storkey", num_patterns >=0, match_frac >= 0
    """
    
    # formatting of the figure
    plt.figure(figsize=(10, 6))
    plt.ylim(-0.1, 1.1)
    plt.xticks(num_patterns, num_patterns)
    plt.xlim(min(num_patterns)-0.5, max(num_patterns) + 0.5)
    plt.xlabel("Number of patterns")
    plt.ylabel("Fraction of retrieved patterns")
    plt.title(f"Capacity curve for a network of size {size} with the {weight_rule} rule")

    # plot points on the figure
    plt.scatter(num_patterns, match_frac)

    # save the figure in the current directory and close it afterwards
    plt.savefig(f"Size{size}_Rule{weight_rule}_CapacityCurve", format="jpg")
    plt.close()


def plot_empirical_capacity(size, num_patterns, match_frac, color):
    """Plots empirical capacity curves including number of neurons vs. capacity (defined as the number of patterns that 
    can be retrieved with a probability higher than 90%).
    
    Parameters:
    --------------
    size: int
    -> size of the network (= size of the patterns of the network)
    num_patterns: int
    -> number of patterns within the network 
    match_frac: floating point 
    -> fraction of convergence, which is the number of patterns retrieved over the total number of patterns for
    the network
    color: string
    -> color of the curve
    
    Output:
    --------------
    empirical capacity curve
    
    CU: size >=0, num_patterns >=0 and match_frac >=0
    """

    max_nb_patterns = 0
    for j in range(len(match_frac)):  # determine for which higher number of patterns we have a system's convergence
        if match_frac[j] >= 0.9 and num_patterns[j] >= max_nb_patterns:
            max_nb_patterns = num_patterns[j]

    # plot points
    plt.scatter(size, max_nb_patterns, c=color)

    return max_nb_patterns


def save_empirical_capacity(results, rule):
    """ Plots a curve corresponding to the empirical capacity and saves the figure in the current directory
    
    Parameters:
    --------------
    results: list 
    -> list of results returned by the function 'experiment'
    rule: string
    -> learning rule that will be used for the calculations ("Hebbian" or "Storkey")
    
    Output:
    --------------
    empirical capacity curve saved in the current directory
    """

    # formatting the figure
    plt.figure(figsize=(10, 7))
    plt.xlabel("Number of Neurons")
    plt.ylabel("Maximum Empirical Capacity")
    plt.title("Empirical Capacity Curve : Maximum Empirical Capacity vs. Number of Neurons")
    color = "black"

    # plot points for each size of the network + store the maximum empirical capacity and the networks' sizes in a list
    max_nb_patterns = []
    sizes = []
    for i in range(len(results)):
        sizes.append(results[i]["network_size"])
        max_nb_patterns.append(plot_empirical_capacity(results[i]["network_size"], results[i]["num_patterns"],
                               results[i]["match_frac"], color))

    # plot a curve connecting all the points
    plt.plot(sizes, max_nb_patterns)

    # save the figure in the current directory and close it afterwards
    plt.savefig(f"Empirical_Capacity_{rule}", format="jpg")
    plt.close()

