from storkey_rule_week6 import *
from function_week6 import *
import numpy as np

index_perturbed = 4
random_patterns = generate_patterns(80, 1000)
perturbed_pattern = perturb_pattern(random_patterns[index_perturbed], 80)

""" Uncomment the part of the test you want to run"""
#weights = hebbian_weights(random_patterns)
#weights = storkey_weights(random_patterns)
#updating_state = dynamics(perturbed_pattern, weights, 20)
#updating_state = dynamics_async(perturbed_pattern, weights, 20000, 3000)

print()
if pattern_match(random_patterns, perturbed_pattern) == index_perturbed:
    print("Pattern recognised")
