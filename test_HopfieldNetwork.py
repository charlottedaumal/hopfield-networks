from functions import *
import os


def test_generate_patterns():
    list_generate_patterns = [-1, 1]

    assert ((generate_patterns(5, 10)).all() in list_generate_patterns)  # testing the values of the  generated patterns

    # testing the size of the generated patterns
    assert ((generate_patterns(5, 10)).shape[0] == 5)
    assert ((generate_patterns(5, 10)).shape[1] == 10)


def test_perturb_pattern():
    list_perturb_pattern = [-1, 1]

    '''testing the values of the perturbed pattern'''
    assert ((perturb_pattern(np.array([[1, 1, 2, 3]]), 7)).all() in list_perturbed_pattern)

    '''testing if the perturbed pattern is different from the original one'''
    assert ((perturb_pattern(np.array([[1, 1, 2, 3]]), 7) != np.array([[1, 1, 2, 3]])).any())


def test_update():
    p = np.array([[2, 5, 6, 7], [4, 5, 6, 9]])
    q = np.array([[1, 1, 1, 1], [4, 5, 6, 9]])
    w = np.array([[1, 1], [1, 1]])
    u = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    list_update = [-1, 1]

    assert(np.allclose(update(p, w), u))

    '''testing the values of the updated pattern'''
    assert ((update(p, w)).all() in list_update)

    '''testing if the updated pattern is different'''
    assert((update(q, w) != q).any())


def test_update_async():
    p = np.array([[8, 9], [0, 0]])
    w = np.array([[1, 1], [2, 2]])
    q = np.array([[2, 5, 6, 7], [4, 5, 6, 9]])
    w_ = np.array([[1, 1], [1, 1]])

    assert(np.allclose(update_async(p, w), np.array([[1, 1]])))

    '''testing the values of the updated pattern'''
    list_update_async = [-1, 1]
    assert((update_async(q, w_)).all() in list_update_async)

    '''testing if the updated pattern is different'''
    assert((update_async(q, w_) != q).any())


def test_hebbian_weights():
    weight_matrix = hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))

    '''testing the symmetry of the matrix'''
    assert (np.allclose(weight_matrix, np.transpose(weight_matrix)))

    '''testing the size of the matrix'''
    weight_matrix = hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    assert (weight_matrix.shape[0] == weight_matrix.shape[1])

    '''testing if the diagonal elements are equal to 0'''
    assert ((np.diagonal(hebbian_weights(np.array(([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))))).all() == 0)


def test_storkey_weights():
    weight_matrix = storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))

    '''testing the symmetry of the matrix'''
    assert(np.allclose(weight_matrix, np.transpose(weight_matrix)))

    '''testing the size of the matrix'''
    assert(weight_matrix.shape[0] == weight_matrix.shape[1])


def test_dynamics():
    s = np.array([[1, 4, 6, 7], [5, 8, 9, 0]])
    w = np.array([[1, 1], [1, 1]])
    a = dynamics(s, w, 10)
    b = [np.array([[1, 4, 6, 7], [5, 8, 9, 0]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]])]

    assert(np.allclose(np.array([a]), np.array([b])))


def test_dynamics_async():
    s = np.array([[1, 0, 9, 7], [3, 7, 8, 9]])
    w = np.array([[1, 5], [4, 9]])
    a = dynamics_async(s, w, 10, 6)
    b = [np.array([[1, 0, 9, 7], [3, 7, 8, 9]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]])]

    assert(np.allclose(np.array([a]), np.array([b])))

    
def test_energy():
    s = np.array([[2, 5]])
    w = np.array([[1, 1], [1, 1]])
    e = energy(s, w)

    assert(np.allclose(np.array([e]), np.array([[-24.5]])))


def test_create_checkerboard():
    list_checkerboard = [-1, 1]

    '''testing the values of the checkerboard'''
    assert(create_checkerboard(50).all() in list_checkerboard)


def test_pattern_match():
    a = np.array([[0, 5, 3, 4], [0, 0, 0, 0]])
    b = np.array([[0, 5, 3, 4]])
    c = np.array([[0, 3, 3, 4]])

    assert(pattern_match(a, b) == 0)
    assert(pattern_match(a, c) == None)
    
       
def test_save_video():
    assert(os.path.exists("./video_synchronous_experiment.mp4"))
    assert(os.path.exists("./video_asynchronous_experiment.mp4"))

