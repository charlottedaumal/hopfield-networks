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

def test_update():
    assert(update(np.array([[2, 5, 6, 7], [4, 5, 6, 9]]), np.array([[1, 1], [1, 1]])).all() == (np.array([[1, 1, 1, 1], [1, 1, 1, 1]])).all())

    '''testing the values of the updated pattern'''
    list = [-1, 1]
    assert((update(np.array([[2, 5, 6, 7], [4, 5, 6, 9]]), np.array([[1, 1], [1, 1]]))).all() in list)


def test_update_async():
    assert(update_async(np.array([[8, 9], [0, 0]]), np.array([[1, 1], [2, 2]])).all() == (np.array([[1, 1]])).all())


def test_hebbian_weights():
    '''testing the symmetry of the matrix'''
    assert(np.allclose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])), np.transpose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])))))

    '''testing the size of the matrix'''
    weight_matrix = hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    assert (weight_matrix.shape[0] == weight_matrix.shape[1])

    '''testing the diagonal elements are equal to 0'''
    assert((np.diagonal(hebbian_weights(np.array(([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))))).all() == 0)


def test_storkey_weights():
    '''testing the symmetry of the matrix'''
    assert(np.allclose(storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])),
                        np.transpose(storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])))))

    '''testing the size of the matrix'''
    weight_matrix = storkey_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]]))
    assert(weight_matrix.shape[0] == weight_matrix.shape[1])
