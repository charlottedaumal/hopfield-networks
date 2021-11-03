import numpy as np
import random as rd


def generate_patterns(num_patterns, pattern_size):
    patterns = np.empty(num_patterns * pattern_size).reshape(num_patterns, pattern_size)
    for i in range(0, num_patterns):
        patterns[i] = np.array(rd.choices([1, -1], k=pattern_size))
    return patterns


def pattern_match(memorized_patterns, pattern):
    for index in range(0, memorized_patterns.shape()[0]):
        if np.allclose(memorized_patterns[index], pattern):
            return index


def update(state, weights): return np.sign(np.dot(weights, state))


def dynamics(state, weights, max_iter):
    state_history = [state]
    previous_state = state
    nb_iter = 0
    while (nb_iter < max_iter) and (not np.allclose(previous_state, new_state)):
        new_state = update(previous_state, weights)
        state_history.append(new_state)
        previous_state = new_state
        nb_iter += 1
    return state_history
