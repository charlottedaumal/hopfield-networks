import numpy as np
import random as rd


def update(state, weights):
    """Applies the update rule to a state pattern
    
    Parameters:
    --------------
    state : array
    -> the network state to which we will apply the update rule
    weights : array
    -> weights matrix
    
    Output:
    --------------
    returns the new state updated from the previous one (a numpy array)
 
    Examples:
    --------------
    >>> update(np.array([1, 1, -1, 1]), np.array([[1, 1, 1, -1], [1, 1, 1, -1]]))
    [array([1, 1])]
    """
    return np.where(np.dot(weights, state) >= 0, 1, -1)


def update_async(state, weights):
    """Applies the asynchronous update rule to a state pattern
    
    Parameters:
    --------------
    state : array
    -> the network state to which we will apply the asynchronous update rule
    weights : array
    -> weights matrix
    
    Output:
    --------------
    returns the new state updated from the previous one (a numpy array)
    
    Examples:
    --------------
    >>> update_async(np.array([-1, -1, -1, 1]), np.array([[1, 1, -1, -1], [1, 1, 1, 1]]))
     array([-1, -1, -1,  1])
    """
    index = rd.choices(np.linspace(0, weights.shape[0] - 1, weights.shape[0], dtype=int))  # chooses randomly an index
    pattern = state.copy()
    pattern[index] = np.where(np.dot(weights[index], state) >= 0, 1, -1) # applying the asynchronous update rule
    # (updates the i-th component of the state pattern)
    return pattern
