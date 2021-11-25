import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def generate_patterns(num_patterns, pattern_size):
    """Generates random binary patterns that will be memorized
    
    Parameters:
    --------------
    num_patterns:int
    -> number of patterns we want to generate.
    pattern_size : int
    -> size of the patterns we want to generate
    
    Output:
    --------------
    returns a pattern (2-dimensional numpy array)
    """
    """Generates random binary patterns that will be memorized"""
    
    return np.random.choice([-1,1], size=(num_patterns, pattern_size))


def perturb_pattern(pattern, num_perturb):
    """Randomly perturbs a given number of times a pattern (changes the sign of its elements)
    
    Parameters:
    --------------
    pattern : array
    -> pattern we want to be perturbed
    num_pertub : int
    -> number of elements from "pattern" we want to be perturbed
    
    Output:
    --------------
    returns the perturbed pattern (2-dimensional numpy array)
    """
    
    pattern_perturbed = pattern.copy()
    for i in range(num_perturb):
        index = rd.choices(np.linspace(0, len(pattern) - 1, len(pattern), dtype=int))
        pattern_perturbed[index] = -pattern_perturbed[index]
    return pattern_perturbed


def pattern_match(memorized_patterns, pattern):
    """Verifies if a given pattern is equal to one of the memorized ones
    
    Parameters:
    --------------
    memorized_pattern: array
    -> initially memorized pattern that was generated previously
    pattern: array
    -> another pattern to which we will compare the initially memorized pattern
    
    Output:
    --------------
    returns 'None' if no memorized pattern matchs
    otherwise, it returns the index of the row corresponding to the matching pattern (an integer).
    """
    
    for index in range(0, memorized_patterns.shape[0]):
        if np.allclose(memorized_patterns[index], pattern):
            return index


def hebbian_weights(patterns):
    """Creates the weight matrix by using the hebbian learning rule on given patterns
    
    Parameters:
    --------------
    patterns: array
    -> pattern to which the hebbian learning rule will be applied
    
    Output:
    --------------
    retuns the weight matrix (a multi-dimensional numpy array)
    """
    w = np.zeros([patterns.shape[1], patterns.shape[1]])
    for row in range(patterns.shape[0]):
        w += np.outer(patterns[row],patterns[row]) * 1/patterns.shape[0]
    np.fill_diagonal(w, 0)
    return w


def update(state, weights):
    """Applies the update rule to a state pattern
    
    Parameters:
    --------------
    state : array
    -> the network state to which we will apply the update rule
    weights : array
    -> weight matrix returned by the function hebbian_weights(patterns)
    
    Output:
    --------------
    returns the new state updated from the previous one (list of numpy arrays)
    """
    
    vector = np.dot(weights, state)
    vector = np.where(vector >= 0, 1, -1)
    return vector


def update_async(state, weights):
    """Applies the asynchronous update rule to a state pattern
    
    Parameters:
    --------------
    state : array
    -> the network state to which we will apply the asynchronous update rule
    weights : array
    -> weights matrix returned by the function hebbian_weights(patterns)
    
    Output:
    --------------
    returns the new state updated from the previous one (list of numpy arrays)
    
 """
    
    index = rd.choices(np.linspace(0, weights.shape[0] - 1, weights.shape[0], dtype=int))
    state_updated = np.dot(weights[index], state)
    state_updated = np.where(state_updated >= 0, 1, -1)
    return state_updated


def dynamics(state, weights, max_iter):
    """Runs the dynamical system from an initial state until convergence
    or until a maximum number of steps is reached
    
    Parameters:
    --------------
    state : array
    -> initial network state
    weights : array
    -> weights matrix returned by the function hebbian_weights(patterns)
    max_iter : int
    -> maximum number of steps reached
    
    Output:
    --------------
    returns the list of the state history 
    """
    
    state_history = [state]
    previous_state = state.copy()
    new_state = np.zeros_like(state)
    nb_iter = 0
    while (nb_iter < max_iter) and (previous_state != new_state).any():
        new_state = update(previous_state, weights)
        state_history.append(new_state)
        previous_state = new_state.copy()
        nb_iter += 1
    return state_history


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """Runs the dynamical system from an initial state until a maximum number
    of steps is reached or a convergence for a given number of steps is reached
    
    Parameters:
    --------------
    state : array
    -> initial network state
    weights : array
    -> matrix returned by the function hebbian_weights(patterns)
    max_iter : int
    -> maximum number of steps reached
    convergence_num_iter : int
    -> maximum number of iterations in which the algorithm can reach convergence
    
    Output:
    --------------
    returns the list of the state history
    """
    
    state_history = [state]
    previous_state = state.copy()
    nb_iter = 0
    nb_iter_convergence = 0
    while (nb_iter < max_iter) and (nb_iter_convergence < convergence_num_iter):
        new_state = update(previous_state, weights)
        state_history.append(new_state)
        previous_state = new_state.copy()
        nb_iter += 1
        if np.allclose(previous_state, new_state):
            nb_iter_convergence += 1
    return state_history


def storkey_weights(patterns):
   """Creates the weight matrix by using the storkey learning rule on given patterns
    
    Parameters:
    --------------
    patterns : array
    -> pattern to which the storkey learning rule will be applied
    
    
    Output:
    --------------
    retuns the weight matrix (a multi-dimensional numpy array)
    """

     w = np.zeros([patterns.shape[1], patterns.shape[1]])

    for mu in range(patterns.shape[0]):
        w_calculation_h = w.copy()
        np.fill_diagonal(w_calculation_h, 0)
        pattern_calculation_h = np.dot(patterns[mu].copy().reshape(patterns.shape[1],1), np.ones((1, patterns.shape[1])))
        np.fill_diagonal(pattern_calculation_h, 0)
        
        h = np.dot(w_calculation_h, pattern_calculation_h)
        w += np.outer(patterns[mu].copy(), patterns[mu].copy()) / patterns.shape[1]   

        product_1 = patterns[mu].copy() * h.copy()
        product_2 = product_1.T

        w -= np.add(product_2, product_1) / patterns.shape[1]
        
    return w


def energy(state, weights):
    """returns the energy value associated to a given pattern
    
    Parameters:
    --------------
    state : array
    -> network state composed of patterns to which we will associate energie values
    weights : array
    -> weights matrix returned by the function Hebbian_weights(patterns)
    
    Output:
    --------------
    returns the enery value associated to each patterns of the state
    """
    
    e = 0
    for i in range(0, weights.shape[0]):
        for j in range(0, weights.shape[1]):
            e += (-1/2 * weights[i][j] * state[i] * state[j])
    return e


def create_checkerboard(n):
   """Function that prints the checkerboard pattern according to a given dimension
    
    Parameters:
    --------------
    n : int
    -> size of the checkerboard 
    
    Output :
    --------------
    returns the checkerboard (multi-dimensional numpy array)
    """

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
    """ generates a video of the evolution of the system
    
    Parameters:
    --------------
    state_list : list of arrays
    -> list of network state
    out_path : 
    
    Output:
    --------------
    saves the video
    """
    
    frames = []
    fig = plt.figure()
    writer = anim.writers['ffmpeg']
    writervideo = writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    for state in state_list:
        frames.append([plt.imshow(state, cmap='Greys')])

    video = anim.ArtistAnimation(fig, frames)
    video.save(out_path, writer=writervideo)
    
