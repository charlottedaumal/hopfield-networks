# Associative Memory Engine with Hopfield Dynamics

**Author:** Charlotte Daumal & Clemence Kiehl  
**Context:** This project was completed as part of the *Projects in informatics for SV* course taught by Prof. Alexander Mathis    
**Language:** Python   
**Date:** December 2021  

---

## 📘 Project Overview

The goal of this project is to simulate the dynamics of a biologically inspired system known as the **Hopfield network** — a model of associative memory introduced by John Hopfield in 1982.

This recurrent neural network models how memory patterns can be stored as stable states in a network of interconnected neurons. Each memory is encoded through weighted connections, allowing the network to recall a stored pattern from a noisy or partial input.

In this project, we implement and analyze the iterative update mechanism that enables the network to converge toward the closest stored pattern — effectively retrieving a memory from an initial, possibly unseen, input.

---

## 📝 Requirements

This project requires Python ≥ 3.5 and the following packages:

- `numpy` — Numerical computing  
- `matplotlib` — Data visualization  
- `random` — Built-in Python module for random number generation  
- `pathlib` — Built-in Python module for filesystem paths  
- `cython` — For compiling optimized Python extensions  
- `distutils.core` — Used in setup scripts for building Cython modules

---

## 💻 Repository Structure 

### 1) Core Logic and Implementation
- `main.py` — Initial implementation of the Hopfield network + basic unit tests  
- `functions.py` — All core functions used in the project  
- `classes.py` — Object-oriented implementation of Hopfield network components  
- `experiment.py` — Final experimental pipeline and plotting functions (v7 release)

### 2) Optimizations with Cython
- `update_cython.py` — Optimized `update` and `update_async` functions  
- `dynamics_cython.py` — Optimized `dynamics` and `dynamics_async` functions  
- `setup.py` — Build script for compiling Cython modules

### 3) Testing
- `test_HopfieldNetwork.py` — Pytest-based unit tests for `functions.py`

### 4) Results and Outputs
- `Graphs/` — Contains energy curve plots (Hebbian and Storkey weight matrices)  
- `summary.md` — Results summary with tables and figures

### 5) Environment and Dependencies
- `requirements.py` — Script to install all necessary Python packages  
- `.gitignore` — Specifies files and folders ignored by Git

---

## ⚙️ How to use our project until v7 release

### 1) Instructions to run the project 

To enjoy the cython optimizations of the update and dynamics functions, you will need to build the two following modules : update_cython.py and dynamics_cython.py
To build these, you need to type on the terminal : `python setup.py build_ext --inplace`.

You need to run the main.py file. 
First, you will need to choose the weights matrix you want to use to do all the further computations. 

-> To use the Hebbian weights matrix, just directly press 'Enter'. 

-> To use the Storkey weights matrix, push the 's' keyboard key and then press 'Enter'. 

-> If you pushed another keyboard key, the computations will be done witn the Hebbian learning rule by default. 

When it has finished to run, the code provides you the two curves of energy related to the weights matrix you have chosen and the two videos of the convergence of the checkerboard saved in the directory where the main.py file lies.

### 2) Instructions to run the tests

You need to run the test_HopfieldNetwork.py file to run all the pytests and the doctests.

-> In the terminal, type the command `pytest test_HopfieldNetwork.py`, then, to see the coverage, type the command `coverage run -m pytest test_HopfieldNetwork.py` and then coverage report -m`.

-> To only run the doctests, click on "Run Doctests in functions" on PyCharm

---

## 🔧 How to use our project on v7 release

-> You need to run the "main.py" file to run our project.

All the graphs will be saved under the current directory.

---

## License

This project is for educational purposes.  
Content © Charlotte Daumal & Clemence Kiehl. Academic use only.
