import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.animation as anim


class HopfieldNetwork:

    def __init__(self, patterns, rule="hebbian"):
        """Initialize the attributes "patterns" and "rule"

        Output:
        --------------
        Initialization of all the attributes with or depending on parameters
        """
        self.patterns = patterns
        self.rule = rule
        if rule == "hebbian":
            self.w = self.hebbian_weights(patterns)
        else:
            self.w = self.storkey_weights(patterns)

    def hebbian_weights(self, patterns):
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
            w += np.outer(patterns[row], patterns[row]) * 1 / patterns.shape[
                0]  # computation of the average contribution of each pattern to the synaptic weight
        np.fill_diagonal(w, 0)  # fill the diagonal of the matrix with zeros
        return w

    def storkey_weights(self, patterns):
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
        w = np.zeros([patterns.shape[1], patterns.shape[1]])
        for mu in range(patterns.shape[0]):
            w_calculation_h = w.copy()
            np.fill_diagonal(w_calculation_h, 0)
            pattern_calculation_h = np.dot(patterns[mu].copy().reshape(patterns.shape[1], 1), np.ones(
                (1, patterns.shape[1])))
            np.fill_diagonal(pattern_calculation_h, 0)
            h = np.dot(w_calculation_h,
                       pattern_calculation_h)
            w += np.outer(patterns[mu].copy(), patterns[mu].copy()) / patterns.shape[
                1]
            product_1 = patterns[
                            mu].copy() * h.copy()
            product_2 = product_1.T
            w -= np.add(product_2, product_1) / patterns.shape[
                1]
        return w

    def update(self, state):
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
        return np.where(np.dot(self.w, state) >= 0, 1, -1)

    def update_async(self, state):
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
        index = rd.choices(np.linspace(0, self.w.shape[0] - 1, self.w.shape[0], dtype=int))  # chooses randomly an index
        pattern = state.copy()
        pattern[index] = np.where(np.dot(self.w[index], state) >= 0, 1, -1)  # applying the asynchronous update rule
        # (updates the i-th component of the state pattern)
        return pattern

    def dynamics(self, state, saver, max_iter=20):
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
        saver.store_iter(state, self.w)
        previous_state = state.copy()
        for i in range(max_iter):
            new_state = self.update(previous_state)  # updating the state
            saver.store_iter(new_state, self.w)  # adding the updated state to the state history list
            if np.allclose(previous_state,
                           new_state):  # verifies if the state before the update is equal to the updated state
                break  # goes out of the for-loop because convergence is reached
            previous_state = new_state.copy()  # iterative perspective of the dynamical evolution of the pattern

    def dynamics_async(self, state, saver, max_iter=1000, convergence_num_iter=100, skip=10):
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
        skip : int
        -> used to save only one every skip states during the evolution of the patterns

        Output:
        --------------
        returns the list of the state history

        CU: max_iter >= 0 and convergence_num_iter >= 0

        Examples:
        --------------
        >>> dynamics_async(np.array([-1, -1, -1, 1]), np.array([[1, 1, -1, -1], [1, 1, 1, 1]]), 10, 6)
        [array([-1, -1, -1,  1]), array([-1, -1, -1,  1])]
        """
        saver.store_iter(state, self.w)
        previous_state = state.copy()
        nb_iter = nb_iter_convergence = 0
        while (nb_iter < max_iter) and (nb_iter_convergence < convergence_num_iter):  # two conditions to run
            # the dynamical system : a maximum number of iterations and a minimum number of convergence iterations
            new_state = self.update_async(previous_state)  # updating the state
            if nb_iter % skip == 0:
                saver.store_iter(new_state, self.w)
            if np.allclose(previous_state, new_state):  # verifies if the state before the update is equal
                # to the updated state
                nb_iter_convergence += 1
            previous_state = new_state.copy()  # affecting the updated pattern to the previous one
            # to perform the next updating step
            nb_iter += 1


class DataSaver:

    def __init__(self):
        """Initialize the attribute "data"

        Output:
        --------------
        Initialization of the attribute "data" to empty lists
        """

        self.data = {"state": [], "energy": []}

    def reset(self):
        """Resets the attribute "data"

        Output:
        --------------
        Resets the attribute "data" with empty lists
        """

        self.data = {"state": [], "energy": []}

    def store_iter(self, state, weights):
        """Stores a given state and its associated energy in the attribute "data"

        Parameters:
        --------------
        state : array
        -> network state to which we will associate energy values
        weights : array
        -> weights matrix

        Output:
        --------------
        Stores a state and its energy value in the attribute "data"
        """

        self.data["state"].append(state.copy())
        self.data["energy"].append(self.compute_energy(state, weights).copy())

    def compute_energy(self, state, weights):
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
        """

        # computes the energy value associated to the state pattern
        return -1 / 2 * np.dot(state, np.dot(weights, state.T))

    def get_data(self):
        """Return the data attribute of the class DataSaver

        Output:
        --------------
        returns the data"""
        return self.data

    def save_video(self, out_path, img_shape):
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
        writer_video = writer(fps=15, metadata=dict(artist='Me'),
                              bitrate=1800)  # initialization of the writer parameter for saving the video

        state_history = self.get_data()["state"]
        for element in state_history:
            img = element.reshape(img_shape)
            frames.append([plt.imshow(img, cmap='Greys')])

        video = anim.ArtistAnimation(fig, frames)  # definition of the animation
        video.save(out_path, writer=writer_video)  # saving the video

    def plot_energy(self):
        """Generates a plot of the evolution of the energy function

        Output:
        --------------
        Plot of the energy function versus time
        """

        plt.figure(figsize=(5, 7))
        plt.plot(np.arange(0, len(self.get_data()["state"]), step=1), self.get_data()["energy"], color='red')
        plt.xlabel("Time [s]")
        plt.ylabel("Energy")
        plt.title("Plot of the energy versus the time")
        plt.show()
