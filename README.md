# BIO-210-team-26

## Hopfield Network 


The aim of this project is to simulate the evolution of biologically-inspired dynamical system, the Hopfield networks. The Hopfield network is a computational model for associative memory, proposed by John Hopfield in 1982. It models a simple mechanism to explain how a neural network could store representations (i.e., the neural activity corresponding to a certain concept) in the form of weighted connections between neurons. 

Here, we will implement the iterative process which allows to retrieve one of the stored (memorized) patterns starting from the representation of a new (unseen) pattern.


### Table of contents
* Requirements
* Files
* Instructions to use/run our project
* Credits


### Requirements
* Python >= 3.5
* numpy
* matplotlib
* random
* pathlib
* cython
* disultils.core


### Files 
1) main.py -> file containing the first version of the code and the unit tests
2) functions.py -> file containing all the functions of the project
3) test_HopfieldNetwork.py -> file containing pytests to test some functions of the file functions.py
4) .gitignore -> file containing all the files that Git was told to ignore
5) Graphs -> directory containing the pictures of the energy curves for the hebbian and the storkey weights matrices
6) update.cython.py -> cython optimization of the update and update_async functions
7) dynamics_cython.py -> cython optimization of the dynamics and dynamics_async functions
8) setup.py -> file used to build the modules update_cython.py and dynamics_cython.py
9) classes.py -> file containing all the object programming part of the project


### How to use our project

#### Instructions to run the project 

To enjoy the cython optimizations of the update and dynamics functions, you will need to build the two following modules : update_cython.py and dynamics_cython.py
To build these, you need to type on the terminal : "python setup.py build_ext --inplace"

You need to run the main.py file. 
First, you will need to choose the weights matrix you want to use to do all the further computations. 

-> To use the Hebbian weights matrix, you can push the 'h' keyboard key and then press 'Enter'. 

-> To use the Storkey weights matrix, push the 's' keyboard key and then press 'Enter'. 

-> If you didn't push any of those two keyboards keys, the code will not run further.

-> If you pushed another keyboard key, the code will raise an error. 

When it has finished to run, the code provides you the two curves of energy related to the weights matrix you have chosen and the two videos of the convergence of the checkerboard saved in the directory where the main.py file lies.


#### Instructions to run the tests

You need to run the test_HopfieldNetwork.py file to run all the pytests and the doctests.

-> In the terminal, type the command "pytest test_HopfieldNetwork.py", then, to see the coverage, type the command "coverage run -m pytest test_HopfieldNetwork.py" and then "coverage report -m".

-> To only run the doctests, click on "Run Doctests in functions" on PyCharm 



### Credits

ClemenceKiehl & charlottedaumal
