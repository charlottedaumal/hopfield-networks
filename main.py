from functions import *

# Generates the matrix of random patterns
random_patterns = generate_patterns(50, 2500)

# Creation of the checkerboard and insertion at the end of the patterns' matrix
checkerboard = create_checkerboard(50)
random_patterns[-1] = checkerboard.flatten()

# Choose randomly an index in the matrix of patterns and perturb the associated pattern
perturbed_pattern = perturb_pattern(random_patterns[-1], 1000)

# Computation of the two weights matrices using the Hebbian and the Storkey learning rules
# depending on what the user chooses through a keyboard interaction
weights = np.zeros(random_patterns.shape[1])
value = input("Please enter 'h' if you want to dot all the computations with the hebbian weights matrix "
              "or 's' if you want to do it with the storkey weights, "
              "then press 'Enter' :\n")
if value == 'h':
    weights = hebbian_weights(random_patterns)
elif value == 's':
    weights = storkey_weights(random_patterns)
else:
    print("ERROR : You didn't enter a proper character")

# Computation of the state history using the dynamics and the asynchronous dynamics function
state_history_sync = dynamics(perturbed_pattern, weights, 30000)
state_history_async = dynamics_async(perturbed_pattern, weights, 30000, 10000)

# Store the energy values associated to the different states into a list
# using the synchronous dynamics rule
energy_list_sync = []
for element in state_history_sync:
    energy_list_sync.append(energy(element, weights))

# Store the energy values associated to the different states into a list
# using the asynchronous dynamics rule
energy_list_async = []
reduced_state_history_async = state_history_async[0::1000]
for element in reduced_state_history_async:
    energy_list_async.append(energy(element, weights))

# Plotting the energy versus time
plt.figure(figsize=(5, 7))
plt.subplot(211)
plt.plot(np.arange(0, len(state_history_sync), step=1), energy_list_sync, color='blue')
plt.ylabel("Energy")
plt.title("Plot of the energy versus the time computed with the synchronous rule")
plt.subplot(212)
plt.plot(np.arange(0, len(reduced_state_history_async), step=1), energy_list_async, color='red')
plt.xlabel("Time [s]")
plt.ylabel("Energy")
plt.title("Plot of the energy versus the time computed with the asynchronous rule")
plt.show()

# Constituting a list of matrices of shape 50x50 to constitute the video
# (for the synchronous dynamics rule)
reshaped_state_history_sync = []
for element in state_history_sync:
    reshaped_state_history_sync.append(element.reshape(50, 50))

# Constituting a list of matrices of shape 50x50 to constitute the video
# (for the asynchronous dynamics rule with only storing one state every 1000)
reduced_state_history_async = state_history_async[0::1000]
reshaped_state_history_async = []
for element in reduced_state_history_async:
    reshaped_state_history_async.append(element.reshape(50, 50))

# Saving the video in the current directory
save_video(reshaped_state_history_sync, "./video_synchronous_experiment.mp4")
save_video(reshaped_state_history_async, "./video_asynchronous_experiment.mp4")
