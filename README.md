# cell-migration
- MPhil in Engineering - University of Cambridge, Department of Engineering
- Modelling the migration of cells (wound healing, tumor growth)

## Packages and environments
- Needs to be run in the Fenics Projet environment
- Requires Numpy, Matplotlib, Scipy

## Construction

- Run main.py to run the simlation
- extract_results and main_save_param NOT USED YET, IN PROGRESS

*In model*:
 
   - model_domains creates the domain and boundaries
   - model_flow and model_phase contain the function to solve the model
   - model_parameters_class saves the parameters in a txt file
   - model_save_evolution saves the results in plots
   - model/main solves the problem (with iterative loop over time)
   
   - model_common contains common functions (mesh generation, main solver) NOT USED YET, IN PROGRESS
   - model_visu NOT USED ANYMORE

*In results*:

   - main_results saves interesting parameters of the interface


