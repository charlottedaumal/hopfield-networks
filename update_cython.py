import numpy as np
import random as rd


def update(state, weights):
    return np.where(np.dot(weights, state) >= 0, 1, -1)


def update_async(state, weights):
    index = rd.choices(np.linspace(0, weights.shape[0] - 1, weights.shape[0], dtype=int))  # chooses randomly an index
    pattern = state.copy()
    pattern[index] = np.where(np.dot(weights[index], state) >= 0, 1, -1) # applying the asynchronous update rule
    # (updates the i-th component of the state pattern)
    return pattern
