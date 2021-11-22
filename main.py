# Import of packages/useful files
import numpy as np
import random as rd
from function_week6 import *

-----------------------------------------------------------------------------------------------------------------------------------------------------

# Code of week5


"""Determination of global parameters"""
N = 50
M = 3
base = [-1, 1]

"""Creation of M random binary patterns"""
patterns = np.empty(M*N).reshape(M, N)
for i in range(0, M):
    patterns[i] = np.array(rd.choices(base, k=N))

"""Creation of the weight matrix with the Hebbian rule"""
W = np.zeros(N * N).reshape(N, N)
for i in range(0, N):
    for j in range(0, N):
        if i != j:
            patterns_sum = 0
            for z in range(0, M):
                patterns_sum += patterns[z][i] * patterns[z][j]
                W[i][j] = (1 / M) * patterns_sum

"""Global verification on weight matrix
print(W)
   Verification symmetric matrix
W_transpose = np.transpose(W)
if W.all() == W_transpose.all():
   print("symmetric matrix")"""

"""Generation of pattern p0 by changing 3 random values of one of the memorized patterns"""
p0 = patterns[0]
counter = 0
while counter < 3:
    index = rd.choices(np.linspace(0, N-1, N, dtype=int))
    p0[index] = rd.choices(np.linspace(-N, N, N**2), k=1)
    counter += 1

"""Applying the update rule"""
T = 20
p_t = p0
for t in range(0, T):
    p_t1 = np.sign(np.dot(W, p_t))
    if np.allclose(p_t1, p_t):
        print(f"Convergence reached after {t} iteration(s)")
        break
    p_t = p_t1
    
-----------------------------------------------------------------------------------------------------------------------------------------------------

# Unit tests of week6


index_perturbed = 4
random_patterns = generate_patterns(80, 1000)
perturbed_pattern = perturb_pattern(random_patterns[index_perturbed], 80)

""" Uncomment the part of the test you want to run"""
#weights = hebbian_weights(random_patterns)
#weights = storkey_weights(random_patterns)
#updating_state = dynamics(perturbed_pattern, weights, 20)
#updating_state = dynamics_async(perturbed_pattern, weights, 20000, 3000)

"""Verifies if the network retrieve the original pattern"""
print()
if pattern_match(random_patterns, perturbed_pattern) == index_perturbed:
    print("Pattern recognised")
    
-----------------------------------------------------------------------------------------------------------------------------------------------------
