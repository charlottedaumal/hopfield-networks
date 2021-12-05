from functions import *


def test_generate_patterns():
    '''testing the values of the  generated patterns '''
    list = [-1, 1]
    assert((generate_patterns(5, 10)).all() in list)

    '''testing the size of the generated patterns'''
    assert((generate_patterns(5, 10)).shape[0] == 5)
    assert((generate_patterns(5, 10)).shape[1] == 10)


def test_perturb_pattern():
    '''testing the values of the perturbed patterns'''
    list = [-1, 1]
    assert((perturb_pattern(np.array([[1, 1, 2, 3]]), 7)).all() in list)

    ''' testing if perturbed pattern is different '''
    assert((perturb_pattern(np.array([[1, 1, 2, 3]]), 7) != np.array([[1, 1, 2, 3]])).any())



def test_update():
    p = np.array([[2, 5, 6, 7], [4, 5, 6, 9]])
    q = np.array([[1, 1, 1, 1], [4, 5, 6, 9]])
    w = np.array([[1, 1], [1, 1]])
    u = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    list = [-1, 1]

    assert(np.allclose(update(p, w), u))

    '''testing the values of the updated pattern'''

    assert((update(p, w)).all() in list)

    """testing if updated pattern is different"""
    assert((update(q, w) != q).any())


def test_update_async():

    assert (update_async(np.array([[8, 9], [0, 0]]), np.array([[1, 1], [2, 2]])).all() == (np.array([[1, 1]])).all())

    """testing the values of the updated pattern"""
    list = [-1, 1]
    assert((update_async(np.array([[2, 5, 6, 7], [4, 5, 6, 9]]), np.array([[1, 1], [1, 1]]))).all() in list)

    """testing if updated pattern is different"""
    assert((update_async(np.array([[1, 1, 1, 1], [4, 5, 6, 9]]), np.array([[1, 1], [1, 1]])) != np.array(
        [[1, 1, 1, 1], [4, 5, 6, 9]])).any())


def test_hebbian_weights():
    weight_matrix = hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))

    '''testing the symmetry of the matrix'''
    assert(np.allclose(weight_matrix, np.transpose(weight_matrix)))

    '''testing the size of the matrix'''
    weight_matrix = hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    assert (weight_matrix.shape[0] == weight_matrix.shape[1])

    '''testing the diagonal elements are equal to 0'''
    assert((np.diagonal(hebbian_weights(np.array(([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))))).all() == 0)


def test_storkey_weights():
    weight_matrix = storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))

    '''testing the symmetry of the matrix'''
    assert(np.allclose(weight_matrix, np.transpose(weight_matrix)))

    '''testing the size of the matrix'''
    assert(weight_matrix.shape[0] == weight_matrix.shape[1])


def test_dynamics():
    a = dynamics(np.array([[1, 4, 6, 7], [5, 8, 9, 0]]), np.array([[1, 1], [1, 1]]), 10)
    b = [np.array([[1, 4, 6, 7]]), np.array([[5, 8, 9, 0]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]])]
    assert(a == b)


def test_dynamics_async():
    a = dynamics_async(np.array([[1, 0, 9, 7], [3, 7, 8, 9]]), np.array([[1, 5], [4, 9]]), 10, 6)
    b = [np.array([[1, 0, 9, 7]]), np.array([[3, 7, 8, 9]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]]), np.array([[1, 1, 1, 1]])]
    assert(a == b)


def test_energy():
    a = np.array([[2, 5, 6, 7], [4, 5, 6, 9]])
    w = np.array([[1, 1], [1, 1]])
    assert(energy(a, w) == [(np.array([[-18.,  -50.,  -72., -128.]]), np.array([[-18.,  -50.,  -72., -128.]]), np.array([[-18.,  -50.,  -72., -128.]]), np.array([[-18.,  -50.,  -72., -128.]]).all())])

    assert(energy())
