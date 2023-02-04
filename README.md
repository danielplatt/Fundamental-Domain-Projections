# Fundamental-Domain-Projections
Code for the CICY experiment, found in section 3.2 of our paper "Group invariant machine learning by fundamental domain projections", available under 
https://openreview.net/pdf?id=RLkbkAgNA58
The fundamental idea is to apply a pre-processing step, to existing machine learning architectures, which renders them G-invariant.
This pre-processing step is geometrically a projection map. The repository consists of the folders 

**experiments**: trains and evaluates architectures from the literature with different pre-processing applied, call load_data and thhe different models **NOT IMPLEMENTED YET**

**data**: all pre-processing is summarised in load_data, which returns numpy arrays with the desired pre-processing applied

**models**: implementation of architectures for CICY from the literature, as keras models

**projection_maps**: implementation of dirichlet and combinatorial projection maps, they are called by load_data, in the data folder
