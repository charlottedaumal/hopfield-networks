import numpy as np
import random as rd


def generate_patterns(num_patterns, pattern_size):
    dimensions = (num_patterns, pattern_size)
    patterns = np.zeros(dimensions).reshape(dimensions)
    for i in range(0, num_patterns):
        patterns[i] = np.array(rd.choices([1, -1], k=pattern_size))
    return patterns


def perturb_pattern(pattern, num_perturb):
    counter = 0
    while counter < num_perturb:
        index = rd.choices(np.linspace(0, len(pattern) - 1, len(pattern), dtype=int))
        pattern[index] = -pattern[index]
        counter += 1
    return pattern


def pattern_match(memorized_patterns, pattern):
    for index in range(0, memorized_patterns.shape[0]):
        if np.allclose(memorized_patterns[index], pattern):
            return index


def hebbian_weights(patterns):
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
    vector = np.dot(weights, state)
    vector = np.where(vector >= 0, 1, -1)
    return vector


def update_async(state, weights):
    index = rd.choices(np.linspace(0, weights.shape[0] - 1, weights.shape[0], dtype=int))
    state_updated = np.dot(weights[index], state)
    state_updated = np.where(state_updated >= 0, 1, -1)
    return state_updated


def dynamics(state, weights, max_iter):
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

