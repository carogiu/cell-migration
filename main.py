### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam
import numpy as np

### Parameters
config = ModelParam(
    # Grid parameters
    h=0.1,  # Smallest element of the grid # don't make it too small!!
    dim_x=15,  # Dimensions of the grid
    dim_y=20,
    # Time parameters
    n=200,  # Number of time increments
    dt=1e-2,  # Time increment (should not be bigger than 0.01)
    # Model parameters
    theta=10,  # Friction ratio beta'/beta , must be bigger than 1!!!!!!
    Cahn=0.6,  # Cahn number, Should be 0.05 - 0.4 (need 0.5h < Cahn < 2h)
    Pe=20,  # Peclet number, should be big (about 1 / Cahn) , can try up to 200
    # Initial perturbation parameters
    h_0=1,  # Amplitude of the initial perturbation, need to be bigger than Cahn # try 0.2
    k_wave=np.sqrt((10 - 1) / 3),  # Wave number of the initial perturbation # the growing one is sqrt((theta-1)/3)
)

### Call function
main_model(config)
