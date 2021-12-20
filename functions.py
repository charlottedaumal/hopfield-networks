import numpy as np
import random as rd


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
    
    return np.random.choice([-1, 1], size=(num_patterns, pattern_size))  # generates 2-dimensional numpy array
    # in which each row is a random binary pattern


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
    indices = rd.choices(np.linspace(0, len(pattern) - 1, len(pattern), dtype=int), k=num_perturb)  # chooses randomly
    # an index to perturb a random element of one given pattern
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
        if np.allclose(memorized_patterns[index], pattern):  # verifies if the pattern passed in parameters match
            # to one of the memorized patterns
            return index


def create_checkerboard():
    """Prints the checkerboard pattern according to a given dimension
    
    Output :
    --------------
    returns the checkerboard (multi-dimensional numpy array)
    """

    vector1 = np.ones(5)
    vector2 = -vector1
    vector3 = np.concatenate((vector1, vector2))
    vector4 = np.tile(vector3, 5)
    vector5 = np.tile(vector4, 5)
    vector6 = np.concatenate((vector2, vector1))
    vector7 = np.tile(vector6, 5)
    vector8 = np.tile(vector7, 5)
    vector9 = np.concatenate((vector5, vector8))
    checkerboard = np.tile(vector9, 5).reshape(50, 50)
    
    return checkerboard
    
