from classes import *
from functions import *

# Generates the matrix of random patterns
random_patterns = generate_patterns(50, 2500)

# Creation of the checkerboard and insertion at the end of the patterns' matrix
checkerboard = create_checkerboard()

# Insertion of the checkerboard at the end of the patterns' matrix
random_patterns[-1] = checkerboard.flatten()

# Perturbing the checkerboard (last pattern of the random patterns' matrix)
perturbed_pattern = perturb_pattern(random_patterns[-1], 1000)

# Initialization of a HopfieldNetwork object
network = HopfieldNetwork(random_patterns)
value = input("Please enter 's' if you want to do all the computations with the storkey weights matrix "
              "then press 'Enter', otherwise just press directly 'Enter' :\n")
if value == 's':
    network = HopfieldNetwork(random_patterns, "storkey")

# Initialization of two DataSaver objects
saver_synchronous = DataSaver()
saver_asynchronous = DataSaver()

# Making the network evolve according to the synchronous or the asynchronous rule
network.dynamics(perturbed_pattern, saver_synchronous)
network.dynamics_async(perturbed_pattern, saver_asynchronous)

# Plotting the energy function first calculated with the synchronous rule and the the asynchronous one
saver_synchronous.plot_energy()
saver_asynchronous.plot_energy()

# Saving videos in the current directory
saver_synchronous.save_video("./video_synchronous_experiment.mp4", (50, 50))
saver_asynchronous.save_video("./video_asynchronous_experiment.mp4", (50, 50))
