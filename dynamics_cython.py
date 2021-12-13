from update_cython import*


def dynamics(state, weights, max_iter):
    """Runs the dynamical system from an initial state until convergence
    or until a maximum number of steps is reached
    
    Parameters:
    --------------
    state : array
    -> initial network state
    weights : array
    -> weights matrix
    max_iter : int
    -> maximum number of steps that can be reached
    
    Output:
    --------------
    returns the list of the state history
    
    CU : max_iter >= 0

    Examples:
    --------------
    >>> dynamics(np.array([1, 8, 0, 9]), np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]), 10)
    [array([1, 8, 0, 9]), array([1, 1, 1, 1]), array([1, 1, 1, 1])]
    """
    
    state_history = [state]
    previous_state = state.copy()
    for i in range(max_iter):
        new_state = update(previous_state, weights)  # updating the state
        state_history.append(new_state)  # adding the updated state to the state history list
        if np.allclose(previous_state,
                       new_state):  # verifies if the state before the update is equal to the updated state
            break  # goes out of the for-loop because convergence is reached
        previous_state = new_state.copy()  # iterative perspective of the dynamical evolution of the pattern
    return state_history


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """Runs the dynamical system from an initial state until a maximum number
    of steps is reached or a convergence for a given number of steps is reached
    
    Parameters:
    --------------
    state : array
    -> initial network state
    weights : array
    -> weights matrix
    max_iter : int
    -> maximum number of steps that can be reached
    convergence_num_iter : int
    -> maximum number of iterations in which the algorithm can reach convergence
    
    Output:
    --------------
    returns the list of the state history
    
    CU: max_iter >= 0 and convergence_num_iter >= 0

    Examples:
    --------------
    >>> dynamics_async(np.array([-1, -1, -1, 1]), np.array([[1, 1, -1, -1], [1, 1, 1, 1]]), 10, 6)
   [array([-1, -1, -1,  1]), array([-1, -1, -1,  1])]
    """
    
    state_history = [state]
    previous_state = state.copy()
    nb_iter = nb_iter_convergence = 0
    while (nb_iter < max_iter) and (
            nb_iter_convergence < convergence_num_iter):  # two conditions to run the dynamical system : a maximum number of iterations and a minimum number of convergence iterations
        new_state = update_async(previous_state, weights)  # updating the state
        state_history.append(new_state)  # adding the updated state to the state history list
        if np.allclose(previous_state,
                       new_state):  # verifies if the state before the update is equal to the updated state
            nb_iter_convergence += 1
        previous_state = new_state.copy()  # affecting the updated pattern to the previous one to perform the next updating step
        nb_iter += 1
    return state_history
