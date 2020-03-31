### Packages
import numpy as np

### Imports
from model.main import main_model
from model.model_parameter_class import ModelParam
from model.save_param import get_and_save_param

### Parameters
config = ModelParam(
    # Grid parameters
    h=(1 / 2) ** 5,  # Smallest element of the grid (power 6,7 or 8)
    dim_x=5,  # Dimensions of the grid
    dim_y=5,

    # Time parameters
    n=50,  # int(1e3),  # Number of time increments
    dt=4e-3*3,  # Time increment (should not be bigger than 0.01)

    # Make sure that dt<h*h*Pe/2 and dt<h/v

    # Model parameters
    theta=20,  # Friction ratio beta'/beta , must be bigger than 1!
    alpha=0.5,  # Dimensionless activity, must be <1
    Cahn=0.12,  # Cahn number (need 0.5h < Cahn < 2h)
    Pe=50,  # Peclet number, should be big (about 1 / Cahn) , can try up to 200
    starting_point=-1.5,  # where the interface is at the beginning

    # Initial perturbation parameters
    h_0=0.15,  # 0.15,  # Amplitude of the initial perturbation
    k_wave=np.sqrt((20 - 1 + 0.5) / 3),  # Wave number of the initial perturbation (theta-1+alpha)

    # Saving parameter
    folder_name='31-3-2020#7'
)

### Call function

# folder_name = get_and_save_param(config)
# config1.folder_name= folder_name  # try something like that to update the folder name?
# print(config1.folder_name)
main_model(config)
