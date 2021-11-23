import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def generate_patterns(num_patterns, pattern_size):
    """Generates random binary patterns that will be memorized"""
    return np.random.choice([-1,1], size=(num_patterns, pattern_size))


def perturb_pattern(pattern, num_perturb):
    """Randomly perturbs a given number of times a pattern (changes the sign of its elements)"""
    counter = 0
    while counter < num_perturb:
        index = rd.choices(np.linspace(0, len(pattern) - 1, len(pattern), dtype=int))
        pattern[index] = -pattern[index]
        counter += 1
    return pattern


def pattern_match(memorized_patterns, pattern):
    """Verifies if a given pattern is equal to one of the memorized ones"""
    for index in range(0, memorized_patterns.shape[0]):
        if np.allclose(memorized_patterns[index], pattern):
            return index


def hebbian_weights(patterns):
    """Creates the weight matrix by using the hebbian learning rule on given patterns"""
    w = np.zeros(patterns.shape[1]**2).reshape(patterns.shape[1], patterns.shape[1])
    for i in range(0, patterns.shape[1]):
        for j in range(0, patterns.shape[1]):
            if i != j:
                patterns_sum = 0
                for z in range(0, patterns.shape[0]):
                    patterns_sum += patterns[z][i] * patterns[z][j]
                w[i][j] = (1 / patterns.shape[0]) * patterns_sum
    return w


def update(state, weights):
    """Applies the update rule to a state pattern"""
    vector = np.dot(weights, state)
    vector = np.where(vector >= 0, 1, -1)
    return vector


def update_async(state, weights):
    """Applies the asynchronous update rule to a state pattern"""
    index = rd.choices(np.linspace(0, weights.shape[0] - 1, weights.shape[0], dtype=int))
    state_updated = np.dot(weights[index], state)
    state_updated = np.where(state_updated >= 0, 1, -1)
    return state_updated


def dynamics(state, weights, max_iter):
    """Runs the dynamical system from an initial state until convergence
    or until a maximum number of steps is reached"""
    state_history = [state]
    previous_state = state
    new_state = np.zeros_like(state)
    nb_iter = 0
    while (nb_iter < max_iter) and (previous_state != new_state).any():
        new_state = update(previous_state, weights)
        state_history.append(new_state)
        previous_state = new_state
        nb_iter += 1
    return state_history


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """Runs the dynamical system from an initial state until a maximum number
    of steps is reached or a convergence for a given number of steps is reached"""
    state_history = [state]
    previous_state = state
    nb_iter = 0
    nb_iter_convergence = 0
    while (nb_iter < max_iter) and (nb_iter_convergence < convergence_num_iter):
        new_state = update(previous_state, weights)
        state_history.append(new_state)
        previous_state = new_state
        nb_iter += 1
        if np.allclose(previous_state, new_state):
            nb_iter_convergence += 1
    return state_history


def storkey_weights(patterns):
    """Creates the weight matrix by using the storkey learning rule on given patterns"""
    dimension = patterns.shape[1]
    nb_patterns = patterns.shape[0]
    w = np.zeros(dimension**2).reshape(dimension, dimension)
    for mu in range(0, nb_patterns):
        new_w = np.zeros(dimension**2).reshape(dimension, dimension)
        for i in range(0, dimension):
            for j in range(0, dimension):
                h = np.zeros(dimension**2).reshape(dimension, dimension)
                for k in range(0, dimension):
                    if (k != i) and (k != j):
                        product = w[i][k] * patterns[mu][k]
                        h[i][j] += product
                product_patterns = patterns[mu][i] * patterns[mu][j]
                sub_products_pattern_h = patterns[mu][i] * h[j][i] - patterns[mu][j] * h[i][j]
                new_w[i][j] = w[i][j] + 1/dimension * (product_patterns - sub_products_pattern_h)
        w = new_w
    return w


def energy(state, weights):
    """returns the energy value associated to a given pattern"""
    e = 0
    for i in range(0, weights.shape[0]):
        for j in range(0, weights.shape[1]):
            e += (-1/2 * weights[i][j] * state[i] * state[j])
    return e


def create_checkerboard(n):
    """Function that prints the checkerboard pattern according to a given dimension"""

    # definition of the checkerboard
    dimension = (n, n)
    checkerboard = np.zeros(dimension, dtype=int)

    # arranges rows and columns
    list_numbers = []
    for k in range(5):
        for m in range(5):
            list_numbers.append(m + k * 10)

    for i in range(5):
        for j in range(n):
            if (i in list_numbers) and (j in list_numbers):
                checkerboard[i][j] = 1
            else:
                checkerboard[i][j] = -1

    for i in range(5, 50):
        checkerboard[i] = -checkerboard[i - 5]

    return checkerboard


def save_video(state_list, out_path):
    frames = []
    fig = plt.figure()
    writer = anim.writers['ffmpeg']
    writervideo = writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    for state in state_list:
        frames.append([plt.imshow(state, cmap='Greys')])

    video = anim.ArtistAnimation(fig, frames)
    video.save(out_path, writer=writervideo)
    
