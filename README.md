# Fundamental-Domain-Projections
Code for the CICY experiment, whose results are found in Table 2 in Section 3.2 of the paper *Aslan, Platt, Sheard: "Group invariant machine learning by fundamental domain projections"*, available at 
https://openreview.net/pdf?id=RLkbkAgNA58.


## Background

The idea is to apply a pre-processing step to existing machine learning architectures which renders them G-invariant. 
To achieve this the pre-processing step itself must be G-invariant. 
A simple example: Say we wanted to learn a function f: R->R and know a-priori that $f(x) = f(-x)$. 
Then the G-invariant preprocessing step, applied to both training and test data would map an element $x$ to its absolute value $|x|$. 
Any network trained on the new data is automatically invariant under the transformation $x \mapsto x$. 
In general the pre-processing step is geometrically a projection map. 


### The CICY experiment
The CICY experiment is a problem from string theory, where each bit of input data is a 12x15 matrix. 
The symmetries of this problem are more involved, any row or column permutation of the same matrix represents the same mathematical object. 
To achieve invariance under this symmetry we propose two G-invariant pre-processing steps. 
The first one is combinatorial and the second uses a discrete gradient descent algorithm to approximate a Dirichlet domain.


## How to use

Install packages in `requirements.txt`. 
Run `experiments/run_ensemble.py`.
This *should* download the raw text file containing the dataset, apply the necessary pre-processing, and run 12 training experiments.
This generates the output:
```
[('train_bull_he_jejjala_mishra_network', '', False, 0.8489226698875427), ('train_bull_he_jejjala_mishra_network', '', True, 0.329024076461792), ('train_hartford_network', '', False, 0.8846641182899475), ('train_hartford_network', '', True, 0.8884664177894592), ('train_he_network', '', False, 0.5938059687614441), ('train_he_network', '', True, 0.18587589263916016), ('train_erbin_finotello', '', False, 0.9808467626571655), ('train_erbin_finotello', '', True, 0.34856629371643066), ('train_erbin_finotello', 'dirichlet', False, 0.9737903475761414), ('train_erbin_finotello', 'dirichlet', True, 0.9833669066429138), ('train_erbin_finotello', 'combinatorial', False, 0.9765625), ('train_erbin_finotello', 'combinatorial', True, 0.7434476017951965)]
```
Accuracies vary slightly compared to the ones from the paper due to randomness when separating the data into training and test set and when making choices for defining the pre-processing maps.

Tested with Python 3.10.


## Description of the repository

 The repository consists of the folders 

`experiments`: trains and evaluates architectures from the literature with different pre-processing applied

`data`: all pre-processing is summarised in `load_data`, which returns numpy arrays with the desired pre-processing applied

`models`: implementation of architectures for CICY from the literature, as Keras models

`projection_maps`: implementation of Dirichlet and combinatorial projection maps. To try out these maps on examples, run projection_maps/dirichlet_project.py or projection_maps/combinatorial_project.py.
 They are called by `load_data`, in the `data` folder
