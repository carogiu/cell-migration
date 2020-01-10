# cell-migration
- MPhil in Engineering - University of Cambridge, Department of Engineering
- Modelling the migration of cells (wound healing, tumor growth)

## Packages and environments
- Needs to be run in the Fenics Projet environment
- Requires Numpy, Matplotlib, Dolfin and Random

## Construction
- model_flow and model_phase contain the function to solve the model
- model_save_evolution saves the results in arrays (not yet in csv files)
- model_visu plots the results
- model_parameters_class saves the parameters
- model/main solves the problem
