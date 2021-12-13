from update_cython import*


def dynamics(state, weights, max_iter):
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
    state_history = [state]
    previous_state = state.copy()
    nb_iter = nb_iter_convergence = 0
    while (nb_iter < max_iter) and (
            nb_iter_convergence < convergence_num_iter):  # two conditions to run the dynamical system : a maximum number of iterations and a minimum number of convergence iterations
        new_state = update(previous_state, weights)  # updating the state
        state_history.append(new_state)  # adding the updated state to the state history list
        if np.allclose(previous_state,
                       new_state):  # verifies if the state before the update is equal to the updated state
            nb_iter_convergence += 1
        previous_state = new_state.copy()  # affecting the updated pattern to the previous one to perform the next updating step
        nb_iter += 1
    return state_history
