import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from update_cython import *
from dynamics_cython import *


def generate_patterns(num_patterns, pattern_size):
    """Generates random binary patterns that will be memorized
    
    Parameters:
    --------------
    num_patterns : int
    -> number of patterns we want to generate.
    pattern_size : int
    -> size of the patterns we want to generate
    
    Output:
    --------------
    returns a pattern (2-dimensional numpy array)
    
    CU : num_patterns >= 0 and pattern_size >= 0
    """
    
    return np.random.choice([-1,1], size=(num_patterns, pattern_size))  # generates 2-dimensional numpy array in which each row is a random binary pattern


def perturb_pattern(pattern, num_perturb):
    """Randomly perturbs a given number of times a pattern (changes the sign of its elements)
    
    Parameters:
    --------------
    pattern : array
    -> pattern we want to be perturbed
    num_perturb : int
    -> number of elements from "pattern" we want to be perturbed
    
    Output:
    --------------
    returns the perturbed pattern (2-dimensional numpy array)
    
    CU: num_perturb >= 0
    """
    
    pattern_perturbed = pattern.copy()
    indices = rd.choices(np.linspace(0, len(pattern) - 1, len(pattern), dtype=int), k=num_perturb)  # chooses randomly an index to perturb a random element of one given pattern
    pattern_perturbed[indices] = -pattern_perturbed[indices]  # inverse the sign of a single element of a given pattern
    return pattern_perturbed


def pattern_match(memorized_patterns, pattern):
    """Verifies if a given pattern is equal to one of the memorized ones
    
    Parameters:
    --------------
    memorized_patterns: array
    -> memorized patterns that was generated previously
    pattern: array
    -> another pattern to which we will compare the memorized pattern
    
    Output:
    --------------
    returns 'None' if no memorized pattern matches
    otherwise, it returns the index of the row corresponding to the matching pattern (an integer).
    
    Examples:
    --------------
    >>> pattern_match(np.array([[0,5,3,4], [0,0,0,0]]), np.array([[0,5,3,4]]))
    0
    """
    
    for index in range(0, memorized_patterns.shape[0]):
        if np.allclose(memorized_patterns[index], pattern):  # verifies if the pattern passed in parameters match to one of the memorized patterns
            return index


def hebbian_weights(patterns):
    """Creates the weight matrix by using the hebbian learning rule on given patterns
    
    Parameters:
    --------------
    patterns: array
    -> patterns randomly generated previously to which the hebbian learning rule will be applied
    
    Output:
    --------------
    returns the weight matrix (a multi-dimensional numpy array)
    
    Examples:
    --------------
    >>> hebbian_weights(np.array([[1,1,4,1], [1,4,5,4]]))
    array([[ 0. ,  2.5,  4.5,  2.5],
           [ 2.5,  0. , 12. ,  8.5],
           [ 4.5, 12. ,  0. , 12. ],
           [ 2.5,  8.5, 12. ,  0. ]])

    >>> hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    array([[ 0.        ,  0.33333333, -0.33333333, -0.33333333],
           [ 0.33333333,  0.        , -1.        ,  0.33333333],
           [-0.33333333, -1.        ,  0.        , -0.33333333],
           [-0.33333333,  0.33333333, -0.33333333,  0.        ]])
    """
    
    w = np.zeros([patterns.shape[1], patterns.shape[1]])  # initialisation of the weights matrix
    for row in range(patterns.shape[0]):
        w += np.outer(patterns[row],patterns[row]) * 1/patterns.shape[0]  # computation of the average contribution of each pattern to the synaptic weight
    np.fill_diagonal(w, 0)  # fill the diagonal of the matrix with zeros
    return w


#def update(state, weights):
#    """Applies the update rule to a state pattern
    
#    Parameters:
#    --------------
#    state : array
#    -> the network state to which we will apply the update rule
#    weights : array
#    -> weights matrix
    
#    Output:
#    --------------
#    returns the new state updated from the previous one (a numpy array)
 
#    Examples:
#    --------------
#    >>> update(np.array([1, 1, -1, 1]), np.array([[1, 1, 1, -1], [1, 1, 1, -1]]))
#    [array([1, 1])]
#    """
    
#    return np.where(np.dot(weights, state) >= 0, 1, -1)  # applying the update rule to a state pattern 


#def update_async(state, weights):
#    """Applies the asynchronous update rule to a state pattern
    
#    Parameters:
#    --------------
#    state : array
#    -> the network state to which we will apply the asynchronous update rule
#    weights : array
#    -> weights matrix
    
#    Output:
#    --------------
#    returns the new state updated from the previous one (a numpy array)
    
#    Examples:
#    --------------
#    >>> update_async(np.array([-1, -1, -1, 1]), np.array([[1, 1, -1, -1], [1, 1, 1, 1]]))
#    [array([-1, -1, -1, 1])]
#    """
    
#    index = rd.choices(np.linspace(0, weights.shape[0] - 1, weights.shape[0], dtype=int))  # chooses randomly an index
#    pattern = state.copy()
#    pattern[index] = np.where(np.dot(weights[index], state) >= 0, 1, -1) # applying the asynchronous update rule
    # (updates the i-th component of the state pattern)
#    return pattern 


#def dynamics(state, weights, max_iter):
#    """Runs the dynamical system from an initial state until convergence
#    or until a maximum number of steps is reached
    
#    Parameters:
#    --------------
#    state : array
#    -> initial network state
#    weights : array
#    -> weights matrix
#    max_iter : int
#    -> maximum number of steps that can be reached
    
#    Output:
#    --------------
#    returns the list of the state history
    
#    CU : max_iter >= 0

#    Examples:
#    --------------
#    >>> dynamics(np.array([1, 8, 0, 9]), np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]), 10)
#    [array([1, 8, 0, 9]), array([1, 1, 1, 1]), array([1, 1, 1, 1])]
#    """
    
#    state_history = [state]
#    previous_state = state.copy()
#    for i in range(max_iter):
#        new_state = update(previous_state, weights)  # updating the state
#        state_history.append(new_state)  # adding the updated state to the state history list
#        if np.allclose(previous_state, new_state):  # verifies if the state before the update is equal to the updated state
#            break  # goes out of the for-loop because convergence is reached
#        previous_state = new_state.copy()  # iterative perspective of the dynamical evolution of the pattern 
#    return state_history


#def dynamics_async(state, weights, max_iter, convergence_num_iter):
#    """Runs the dynamical system from an initial state until a maximum number
#    of steps is reached or a convergence for a given number of steps is reached
    
#    Parameters:
#    --------------
#    state : array
#    -> initial network state
#    weights : array
#    -> weights matrix
#    max_iter : int
#    -> maximum number of steps that can be reached
#    convergence_num_iter : int
#    -> maximum number of iterations in which the algorithm can reach convergence
    
#    Output:
#    --------------
#    returns the list of the state history
    
#    CU: max_iter >= 0 and convergence_num_iter >= 0

#    Examples:
#    --------------
#    >>> dynamics_async(np.array([[1,0,9,7], [3,7,8,9]]), np.array([[1,5], [4,9]]), 10, 6)
#    [array([[1, 0, 9, 7],
#           [3, 7, 8, 9]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]]), array([[1, 1, 1, 1],
#           [1, 1, 1, 1]])]
#    """
    
#    state_history = [state]
#    previous_state = state.copy()
#    nb_iter = nb_iter_convergence = 0
#    while (nb_iter < max_iter) and (nb_iter_convergence < convergence_num_iter):  # two conditions to run the dynamical system : a maximum number of iterations and a minimum number of convergence iterations
#        new_state = update_async(previous_state, weights)  # updating the state
#        state_history.append(new_state)  # adding the updated state to the state history list
#        if np.allclose(previous_state, new_state):  # verifies if the state before the update is equal to the updated state
#            nb_iter_convergence += 1
#        previous_state = new_state.copy()  # affecting the updated pattern to the previous one to perform the next updating step
#        nb_iter += 1
#    return state_history


def storkey_weights(patterns):
    """Creates the weights matrix by using the storkey learning rule on given patterns
    
    Parameters:
    --------------
    patterns : array
    -> patterns randomly generated previously to which the storkey learning rule will be applied
    
    Output:
    --------------
    returns the weights matrix (a multi-dimensional numpy array)
    
    Examples:
    --------------
    >>> storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    array([[ 1.125,  0.25 , -0.25 , -0.5  ],
           [ 0.25 ,  0.625, -1.   ,  0.25 ],
           [-0.25 , -1.   ,  0.625, -0.25 ],
           [-0.5  ,  0.25 , -0.25 ,  1.125]])
    """

    w = np.zeros([patterns.shape[1], patterns.shape[1]])  # initialization of the weights matrix with zeros
    for mu in range(patterns.shape[0]):  # iterating on the number of patterns of the random patterns' matrix
        w_calculation_h = w.copy()  # definition of a matrix equal to w
        np.fill_diagonal(w_calculation_h, 0)  # fill the diagonal of the previous matrix with zeros
        pattern_calculation_h = np.dot(patterns[mu].copy().reshape(patterns.shape[1],1), np.ones((1, patterns.shape[1])))  # definition of a matrix in which all columns are a given pattern 
        np.fill_diagonal(pattern_calculation_h, 0)  # fill the diagonal of the previous matrix with zeros
        h = np.dot(w_calculation_h, pattern_calculation_h)  # computation of the matrix h for a given pattern with a matrices' product
        w += np.outer(patterns[mu].copy(), patterns[mu].copy()) / patterns.shape[1]  # computation of the term with the product of the given pattern with itself
        product_1 = patterns[mu].copy() * h.copy()  # computation of the term with the product of the given pattern with the h matrix
        product_2 = product_1.T  # computation of the second product of the given pattern with the h matrix (which is the transpose of the first one)
        w -= np.add(product_2, product_1) / patterns.shape[1]  # substracting the sum of the two products computed above
    return w


def energy(state, weights):
    """Returns the energy value associated to the network state
    
    Parameters:
    --------------
    state : array
    -> network state
    weights : array
    -> weights matrix
    
    Output:
    --------------
    returns the energy value associated to the network state
    
    Example:
    --------------
    >>> energy(np.array([[2, 5]]), np.array([[1, 1], [1, 1]]))
    array([[-24.5]])
    """
    
    return -1/2 * np.dot(state, np.dot(weights, state.T))  # computes the energy value associated to the state pattern


def create_checkerboard(n):
    """Prints the checkerboard pattern according to a given dimension
    
    Parameters:
    --------------
    n : int
    -> size of the checkerboard 
    
    Output :
    --------------
    returns the checkerboard (multi-dimensional numpy array)
    """

    checkerboard = np.zeros((n, n), dtype = int)

    vector1 = np.ones(5)
    vector2 = -vector1
    vector3 = np.concatenate((vector1, vector2))
    vector4 = np.tile(vector3, 5)
    vector5 = np.tile(vector4, 5)
    vector6 = np.concatenate((vector2, vector1))
    vector7 = np.tile(vector6, 5)
    vector8 = np.tile(vector7, 5)
    vector9 = np.concatenate((vector5, vector8))
    checkerboard = np.tile(vector9, 5).reshape(50,50)
    
    return checkerboard


def save_video(state_list, out_path):
    """Generates a video of the evolution of the system
    
    Parameters:
    --------------
    state_list : list of arrays
    -> list of network state
    out_path : string
    -> path where the video will be saved
    
    Output:
    --------------
    saves the video
    """
    
    frames = []  # initialising an empty list of frames 
    fig = plt.figure()  # definition of a figure (needed for the visualization)
    writer = anim.writers['ffmpeg']  # initialization of pipe-based ffmpeg writer
    writer_video = writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)  # initialization of the writer parameter for saving the video

    for state in state_list:  # adding all the state's configurations in the list of frames 
        frames.append([plt.imshow(state, cmap='Greys')])

    video = anim.ArtistAnimation(fig, frames)  # definition of the animation
    video.save(out_path, writer=writer_video)  # saving the video
    
