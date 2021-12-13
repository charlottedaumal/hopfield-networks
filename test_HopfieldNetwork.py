import functions
import doctest
from pathlib import Path
import numpy as np


def test_hopfield_network():
    """integrating the doctests in the pytest framework"""
    assert doctest.testmod(functions, raise_on_error=True)


def test_generate_patterns():
    """testing the function generate_patterns"""
    list_generate_patterns = [-1, 1]

    # testing the values of the generated patterns
    assert ((functions.generate_patterns(5, 10)).all() in list_generate_patterns)

    # testing the size of the generated patterns
    assert ((functions.generate_patterns(5, 10)).shape[0] == 5)
    assert ((functions.generate_patterns(5, 10)).shape[1] == 10)


def test_perturb_pattern():
    """testing the function perturb_pattern"""
    list_perturb_pattern = [-1, 1]

    # testing the values of the perturbed pattern
    assert ((functions.perturb_pattern(np.array([[1, 1, 2, 3]]), 7)).all() in list_perturb_pattern)

    # testing if the perturbed pattern is different from the original one
    assert ((functions.perturb_pattern(np.array([[1, 1, 2, 3]]), 7) != np.array([[1, 1, 2, 3]])).any())


def test_update():
    """testing the function update"""
    p = np.array([[2, 5, 6, 7], [4, 5, 6, 9]])
    q = np.array([[1, 1, 1, 1], [4, 5, 6, 9]])
    w = np.array([[1, 1], [1, 1]])
    list_update = [-1, 1]

    assert ((update_cython.update(p, w)).all() in list_update)  # testing the values of the updated pattern

    assert((update_cython.update(q, w) != q).any())  # testing if the updated pattern is different


def test_update_async():
    """testing the function update_async"""
    q = np.array([[2, 5, 6, 7], [4, 5, 6, 9]])
    w_ = np.array([[1, 1], [1, 1]])

    # testing the values of the updated pattern
    list_update_async = [-1, 1]
    assert((update_cython.update_async(q, w_)).all() in list_update_async)

    assert((update_cython.update_async(q, w_) != q).any())  # testing if the updated pattern is different


def test_hebbian_weights():
    """testing the function hebbian_weights"""
    weight_matrix = functions.hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    
    assert (np.allclose(weight_matrix, np.transpose(weight_matrix)))  # testing the symmetry of the matrix

    assert (weight_matrix.shape[0] == weight_matrix.shape[1])  # testing the size of the matrix

    # testing if the diagonal elements are equal to 0
    assert ((np.diagonal(functions.hebbian_weights(np.array(([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))))).all() 
            == 0)


def test_storkey_weights():
    """testing the function storkey_weights"""
    weight_matrix = functions.storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])) 

    assert(np.allclose(weight_matrix, np.transpose(weight_matrix)))  # testing the symmetry of the matrix

    assert(weight_matrix.shape[0] == weight_matrix.shape[1])  # testing the size of the matrix


def test_dynamics():
    """testing the function dynamics"""
    s = np.array([[1, 4, 6, 7], [5, 8, 9, 0]])
    w = np.array([[1, 1], [1, 1]])
    a = dynamics_cython.dynamics(s, w, 10)
    b = [np.array([[1, 4, 6, 7], [5, 8, 9, 0]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], 
                                                                                                   [1, 1, 1, 1]])]

    # testing the return value of the function dynamics for a specific input
    assert(np.allclose(np.array([a]), np.array([b])))


def test_dynamics_async():
    """testing the function dynamics_async"""
    s = np.array([[1, 0, 9, 7], [3, 7, 8, 9]])
    w = np.array([[1, 1], [1, 1]])
    a = dynamics_cython.dynamics_async(s, w, 10, 6)
    b = [np.array([[1, 0, 9, 7], [3, 7, 8, 9]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], 
                                                                                                   [1, 1, 1, 1]]), 
         np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], 
                                                                                                   [1, 1, 1, 1]]), 
         np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]])]

    # testing the return value of the functions dynamics_async for a specific input 
    assert(np.allclose(np.array([a]), np.array([b])))

    
def test_energy():
    """testing the function energy"""
    s = np.array([[2, 5]])
    w = np.array([[1, 1], [1, 1]])
    e = functions.energy(s, w)

    # testing the return value of the functions energy for a specific input
    assert(np.allclose(np.array([e]), np.array([[-24.5]])))


def test_create_checkerboard():
    """testing the function create_checkerboard"""
    list_checkerboard = [-1, 1]

    assert(functions.create_checkerboard(50).all() in list_checkerboard)  # testing the values of the checkerboard


def test_pattern_match():
    """testing the function pattern_match"""
    a = np.array([[0, 5, 3, 4], [0, 0, 0, 0]])
    b = np.array([[0, 5, 3, 4]])
    c = np.array([[0, 3, 3, 4]])

    # testing the return value of the pattern_match function for a specific input
    assert(functions.pattern_match(a, b) == 0) 
    assert(functions.pattern_match(a, c) is None)
    
       
def test_save_video():
    """testing the function save_video"""
    
    # testing if the video file exists and is saved where it should be
    random_patterns = functions.generate_patterns(2, 2500)
    checkerboard = functions.create_checkerboard(50)
    random_patterns[-1] = checkerboard.flatten()
    perturbed_pattern = functions.perturb_pattern(random_patterns[-1], 100)
    weights = functions.hebbian_weights(random_patterns)

    state_history_test = dynamics_cython.dynamics(perturbed_pattern, weights, 20)
    state_list_test = [np.reshape(test_state, (50, 50)) for test_state in state_history_test]
    path_test = Path("./video_saved_test.mp4")
    functions.save_video(state_list_test, path_test)
    assert(path_test.is_file())
