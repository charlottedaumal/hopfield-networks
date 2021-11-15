from functions_week6 import *
import numpy as np

random_patterns = generate_patterns(80, 100)
perturbed_pattern = perturb_pattern(random_patterns[4], 80)
weights = storkey_weights(random_patterns)
updating_state = dynamics(perturbed_pattern, weights, 20)

print()
if np.allclose(updating_state[len(updating_state)-1], random_patterns[4]):
    print("Pattern recognised")
else:
    print("Miskin")