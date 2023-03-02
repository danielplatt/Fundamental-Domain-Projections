# Fundamental-Domain-Projections

## How to use

Run `experiments/run_ensemble.py`.
This *should* download the raw text file containing the dataset, apply the necessary pre-processing, and run 12 training experiments.

## Description of the repository

Code for the CICY experiment, whose results are found in Table 2 in Section 3.2 of the paper *Aslan, Platt, Sheard: "Group invariant machine learning by fundamental domain projections"*, available at 
https://openreview.net/pdf?id=RLkbkAgNA58.
The fundamental idea is to apply a pre-processing step to existing machine learning architectures which renders them G-invariant.
This pre-processing step is geometrically a projection map. The repository consists of the folders 

`experiments`: trains and evaluates architectures from the literature with different pre-processing applied

`data`: all pre-processing is summarised in `load_data`, which returns numpy arrays with the desired pre-processing applied

`models`: implementation of architectures for CICY from the literature, as Keras models

`projection_maps`: implementation of Dirichlet and combinatorial projection maps, they are called by `load_data`, in the `data` folder
