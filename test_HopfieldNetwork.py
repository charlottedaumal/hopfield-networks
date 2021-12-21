from classes import *
import functions
import doctest
from pathlib import Path
import numpy as np
import update_cython
import dynamics_cython


def test_hopfield_network():
    """integrating the doctests in the pytest framework"""
    assert doctest.testmod(functions, raise_on_error=True)


def test_generate_patterns():
    """testing the function generate_patterns"""
    list_generate_patterns = [-1, 1]

    # testing the values of the generated patterns
    assert (functions.generate_patterns(5, 10)).all() in list_generate_patterns

    # testing the size of the generated patterns
    assert (functions.generate_patterns(5, 10)).shape[0] == 5
    assert (functions.generate_patterns(5, 10)).shape[1] == 10


def test_perturb_pattern():
    """testing the function perturb_pattern"""
    list_perturb_pattern = [-1, 1]

    # testing the values of the perturbed pattern
    assert (functions.perturb_pattern(np.array([[1, 1, 2, 3]]), 7)).all() in list_perturb_pattern

    # testing if the perturbed pattern is different from the original one
    assert (functions.perturb_pattern(np.array([[1, 1, 2, 3]]), 7) != np.array([[1, 1, 2, 3]])).any()

    
def test_update(benchmark):
    """testing the function update"""
    p = np.array([-1, 1, -1, 1])
    w = np.array([[1, 1, -1, -1], [1, 1, 1, 1], [1, 1, -1, 1], [1, -1, -1, 1]])
    list_update = [-1, 1]
    p_updated = benchmark.pedantic(update_cython.update, args=(p, w), iterations=100)

    network = HopfieldNetwork(functions.generate_patterns(50, 4))
    p_updated_in_class = network.update(p)

    assert p_updated.all() in list_update  # testing the values of the updated pattern
    assert (update_cython.update(p, w) != p).any()  # testing if the updated pattern is different
    assert p_updated_in_class.all() in list_update  # testing the values of the updated pattern
    assert p_updated_in_class is not None  # testing whether the function update_async has a return type


    
def test_update_async(benchmark):
    """testing the function update_async"""
    p = np.array([-1, 1, -1, 1])
    w = np.array([[1, 1, -1, -1], [1, 1, 1, 1], [1, 1, -1, 1], [1, -1, -1, 1]])
    p_updated = benchmark.pedantic(update_cython.update_async, args=(p, w), iterations=100)
    list_update_async = [-1, 1]

    network = HopfieldNetwork(functions.generate_patterns(50, 4))
    p_updated_in_class = network.update_async(p)

    assert p_updated.all() in list_update_async  # testing the values of the updated pattern
    assert p_updated is not None  # testing whether the function update_async has a return type
    assert p_updated_in_class.all() in list_update_async  # testing the values of the updated pattern
    assert p_updated_in_class is not None  # testing whether the function update_async has a return type


def test_hebbian_weights(benchmark):
    """testing the function hebbian_weights"""
    weights = benchmark.pedantic(HopfieldNetwork.hebbian_weights, args=(HopfieldNetwork, functions.generate_patterns(5, 25)), iterations=5)

    assert (np.allclose(weights, np.transpose(weights)))  # testing the symmetry of the matrix

    # testing if the diagonal elements are equal to 0
    assert np.diagonal(weights).all() == 0
    assert weights.shape[0] == weights.shape[1]


def test_storkey_weights(benchmark):
    """testing the function storkey_weights"""
    weights = benchmark.pedantic(HopfieldNetwork.storkey_weights, args=(HopfieldNetwork, functions.generate_patterns(5, 25)), iterations=5)

    assert np.allclose(weights, np.transpose(weights))  # testing the symmetry of the matrix
    assert weights.shape[0] == weights.shape[1]  # testing the size of the matrix


def test_dynamics():
    """testing the function dynamics"""
    s = np.array([1, 8, 0, 9])
    w = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    a = dynamics_cython.dynamics(s, w, 10)
    b = [np.array([np.array([1, 8, 0, 9]), np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])])]

    network_test = HopfieldNetwork(functions.generate_patterns(50, 4))
    saver_test = DataSaver()
    network_test.dynamics_async(s, saver_test)

    assert np.allclose(np.array([a]), np.array([b]))  # testing the return value of the function dynamics
    # for a specific input
    assert saver_test.data != ([], [])  # testing that the method dynamics_async saves the states in the saver

def test_dynamics_async():
    """testing the function dynamics_async"""
    s = np.array([-1, -1, -1, 1])
    w = np.array([[1, 1, -1, -1], [1, 1, 1, 1]])
    a = dynamics_cython.dynamics_async(s, w, 10, 6)
    b = [np.array([-1, -1, -1,  1]), np.array([-1, -1, -1,  1]), np.array([-1, -1, -1,  1]), np.array([-1, -1, -1,  1]),
         np.array([-1, -1, -1,  1]), np.array([-1, -1, -1,  1]), np.array([-1, -1, -1,  1])]

    network_test = HopfieldNetwork(functions.generate_patterns(50, 4))
    saver_test = DataSaver()
    network_test.dynamics_async(s, saver_test)

    assert np.allclose(np.array([a]), np.array([b]))  # testing the return value of the functions dynamics_async
    # for a specific input
    assert saver_test.data != ([], [])  # testing that the method dynamics_async saves the states in the saver


    
def test_energy(benchmark):
    """testing the function energy"""
    s = np.array([[2, 5]])
    w = np.array([[1, 1], [1, 1]])
    saver_test = DataSaver()
    e = benchmark.pedantic(DataSaver.compute_energy, args=(saver_test, s, w), iterations=5)

    # testing the return value of the functions energy for a specific input
    assert np.allclose(np.array([e]), np.array([[-24.5]]))


def test_create_checkerboard():
    """testing the function create_checkerboard"""
    list_checkerboard = [-1, 1]
    checkerboard = functions.create_checkerboard()

    assert checkerboard.all() in list_checkerboard  # testing the values of the checkerboard
    assert checkerboard.shape == (50, 50)


def test_pattern_match():
    """testing the function pattern_match"""
    a = np.array([[1, 1, -1, -1]])
    b = np.array([[1, 1, -1, -1]])
    c = np.array([[1, 1, 1, 1]])

    # testing the return value of the pattern_match function for a specific input
    assert functions.pattern_match(a, b) == 0
    assert functions.pattern_match(a, c) is None
    
       
def test_save_video():
    """testing the function save_video"""
    random_patterns = functions.generate_patterns(2, 2500)
    checkerboard = functions.create_checkerboard()
    random_patterns[-1] = checkerboard.flatten()
    perturbed_pattern = functions.perturb_pattern(random_patterns[-1], 100)
    network_test = HopfieldNetwork(random_patterns)
    saver_test = DataSaver()
    network_test.dynamics(perturbed_pattern, saver_test)
    path_test = Path("./video_saved_test.mp4")
    saver_test.save_video(path_test, (50, 50))

    assert path_test.is_file()  # testing if the video file exists and is saved where it should be
    
    
def test_arguments_class_HopfieldNetwork():
    """testing if all the arguments of the class HopfieldNetWork are well initialized"""
    network_h = HopfieldNetwork(functions.generate_patterns(2, 50))
    network_s = HopfieldNetwork(functions.generate_patterns(2, 50), "storkey")

    assert network_h.w is not None  # testing if the network has a weights matrix
    assert network_s.w is not None  # testing whether the network has a weights matrix
    assert network_h.rule == "hebbian"  # testing whether the weights matrix of the network is computed using
    # the hebbian learning rule
    assert network_s.rule == "storkey"  # testing whether the weights matrix of the network is computed using
    # the storkey learning rule
    
    
def test_reset_method_class_DataSaver():
    """testing if the reset method of the class DataSaver resets all the arguments"""
    saver_test = DataSaver()
    network_test = HopfieldNetwork(functions.generate_patterns(50, 10))
    state_test = np.array([-1, 1, -1, 1, 1, -1, 1, 1, 1, -1])
    network_test.dynamics(state_test, saver_test)
    saver_test.reset()

    assert saver_test.get_data()["state"] == []  # testing that the argument relative to the key "state" does not 
    # contain any value
    assert saver_test.get_data()["energy"] == [] # testing that the argument relative to the key "energy" does not 
    # contain any value
    
    
def test_plot_energy():
    """testing if the plot of energy is done"""
    saver_test = DataSaver()
    s1 = np.array([1, 8, 0, 9])
    s2 = np.array([1, -3, 6, -5])
    w = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    saver_test.store_iter(s2, w)
    saver_test.store_iter(s1, w)
    saver_test.plot_energy()  # displays a test curve to see if the plotting method works

