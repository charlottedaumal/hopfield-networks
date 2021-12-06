# BIO-210-team-26

## Hopfield Network 


### Credits
ClemenceKiehl & charlottedaumal


### Description
The aim of this project is to simulate the evolution of biologically-inspired dynamical system, the Hopfield networks. The Hopfield network is a computational model for associative memory, proposed by John Hopfield in 1982. It models a simple mechanism to explain how a neural network could store representations (i.e., the neural activity corresponding to a certain concept) in the form of weighted connections between neurons. 

Here, we will implement the iterative process which allows to retrieve one of the stored (memorized) patterns starting from the representation of a new (unseen) pattern.


### Table of contents 
1) main.py -> file containing the first version of the code and the unit tests
2) functions.py -> file containing all the functions of the project
3) test_HopfieldNetwork.py -> file containing pytests to test some functions of the file functions.py


### How to use our project

#### Instructions to run the project 
You need to run the main.py file. First, you will need to choose the weights matrix you want to use to do all the further computations. 
If you want to use the Hebbian weights matrix, you can push the 'h' keyboard key and then press 'Enter'. To use the Storkey weights matrix, push the 's' keyboard key and then press 'Enter'. 
If you didn't push any of those two keyboards keys, the code will not run further. If you pushed another keyboard key, the code will raise an error. 

When it has finished to run, the code provides you the two curves of energy related to the weights matrix you have chosen and the two videos of the convergence of the checkerboard saved in the directory where the main.py file lies.


#### Instructions to run the tests
You need to run the test_HopfieldNetwork.py file to run all the pytests.

