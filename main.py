# Import of packages/useful files
from functions import *

# ---------------------------------------------------------------------------------------------------------------------------------

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
        print()
        print("Results of unit tests of week 5 :")
        print(f"Convergence reached after {t} iteration(s)")
        break
    p_t = p_t1
    
# ---------------------------------------------------------------------------------------------------------------------------------

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
if pattern_match(random_patterns, perturbed_pattern) == index_perturbed:
    print()
    print("Results of unit tests of week 6 :")
    print("Pattern recognised")
    
# ---------------------------------------------------------------------------------------------------------------------------------

# Unit tests of week7

# Energy plotting tests

"""creates random patterns, perturbs one of them and stores them in a Hopfield network"""
index_perturbed = 4
random_patterns = generate_patterns(50, 100)
perturbed_pattern = perturb_pattern(random_patterns[index_perturbed], 10)
random_patterns[index_perturbed] = perturbed_pattern
hopfield_network = random_patterns


"""using the hebbian rule / with synchronous update rule or asynchronous update rule"""
weights_h = hebbian_weights(hopfield_network)
state_history_h = dynamics(perturbed_pattern, weights_h, 20)
state_history_h_a = dynamics_async(perturbed_pattern, weights_h, 3000, 1000)


"""using the storkey rule / with synchronous update rule or asynchronous update rule"""
weights_s = storkey_weights(hopfield_network)
state_history_s = dynamics(perturbed_pattern, weights_s, 20)
state_history_s_a = dynamics_async(perturbed_pattern, weights_s, 3000, 1000)


"""evaluating and comparing the energy of each of the states"""
"""trying with state_history_s"""

energy_list = []
for i in range(0, len(state_history_s)):
    F = energy(state_history_s[i], weights_s)
    energy_list.append(F)

plt.figure(figsize=(5,7))
plt.plot(np.arange(0,len(energy_list), step=1), energy_list, 'b')
plt.ylabel("energy")
plt.xlabel("time[s]")
plt.title("energy-time plot")
plt.show()

# Visualization tests

"""Initializing the checkerboard, the randome patterns and the perturbed pattern"""
checkerboard = create_checkerboard(50)
random_patterns = generate_patterns(50, 2500)
random_patterns[-1] = checkerboard.flatten()
perturbed_pattern = perturb_pattern(random_patterns[-1], 1000)

""" Uncomment the part of the test you want to run"""

"""Calculating the weight matrix according to Hebbian rule or Storkey rule"""
weights_heb = hebbian_weights(random_patterns)
# weights_sto = storkey_weights(random_patterns)
updating_state_dynamics = dynamics(perturbed_pattern, weights_heb, 20)
# updating_state_dynamics_async = dynamics_async(perturbed_pattern, weights_sto, 20000, 3000)

reshaped_updating_state_dynamics = []
for i in range(len(updating_state_dynamics)):
    reshaped_updating_state_dynamics.append(updating_state_dynamics[i].reshape(50, 50))

# reduced_updating_state_dynamics_async = updating_state_dynamics_async[::1000].copy()
# reshaped_updating_state_dynamics_async = []
# for i in range(len(updating_state_dynamics_async)):
    # reshaped_updating_state_dynamics_async.append(updating_state_dynamics_async[i].reshape(50, 50))

"""Saving the video in the current directory"""
save_video(reshaped_updating_state_dynamics, "./video_synchronous_experiment.mp4")
# save_video(updating_state_dynamics_async, "./video_asynchronous_experiment.mp4")

# ---------------------------------------------------------------------------------------------------------------------------------

