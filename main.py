### Packages
import numpy as np

### Imports
from model.main import main_model
from model.model_parameter_class import ModelParam
from main_save_param import get_and_save_param

### Parameters
config = ModelParam(
    # Grid parameters
    h=(1 / 2) ** 5,  # Smallest element of the grid (power 6,7 or 8)
    dim_x=2,  # Dimensions of the grid
    dim_y=2,

    # Time parameters
    n=3,  # int(1e3),  # Number of time increments
    dt=4e-3,  # Time increment (should not be bigger than 0.01)

    # Make sure that dt<h*h*Pe/2 and dt<h/v

    # Model parameters
    theta=10,  # Friction ratio beta'/beta , must be bigger than 1!
    Cahn=0.12,  # Cahn number (need 0.5h < Cahn < 2h)
    Pe=10,  # Peclet number, should be big (about 1 / Cahn) , can try up to 200
    starting_point=0,  # where the interface is at the beginning

    # Initial perturbation parameters
    h_0=0.15,  # 0.15,  # Amplitude of the initial perturbation
    k_wave=np.sqrt((10 - 1) / 3),  # Wave number of the initial perturbation

    # Saving parameter
    folder_name='11-3-2020#1'
)

### Call function
# folder_name = get_and_save_param(config)
# print(config1.folder_name)        try something like that to update the folder name?
# config1.folder_name= folder_name
# print(config1.folder_name)
main_model(config)

