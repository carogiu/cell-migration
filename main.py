### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam
import numpy as np

### Parameters
config = ModelParam(
    # Grid parameters
    h=0.04,  # Smallest element of the grid. Should be 0.1 - 0.2 , but try also smaller
    dim_x=10,  # Dimensions of the grid,  should try to be at least 50 x 50 for 'biological' situation
    dim_y=10,
    # Time parameters
    n=2,  # Number of time increments
    dt=5e-2,  # Time increment (should not be bigger than 0.01)
    # Model parameters
    theta=10,  # Friction ratio beta'/beta , must be bigger than 1!!!!!!
    Cahn=0.1,  # Cahn number. Should be 0.05 - 0.4 (need 0.5h < Cahn < 2h)
    Pe=20,  # Peclet number, should be big (about 1 / Cahn) , can try up to 200
    # Initial perturbation parameters
    h_0=0.1,  # Amplitude of the initial perturbation, need to be bigger than Cahn,  try 1-5 for biological situation
    wave=np.sqrt((10-1)/3),  # Wave number of the initial perturbation # the growing one is sqrt((theta-1)/3)

)

### Call function
main_model(config)
