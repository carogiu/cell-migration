### Packages
import numpy as np

### Imports
from model.main import main_model
from model.model_parameter_class import ModelParam
from model.save_param import get_and_save_param

### Parameters

config1 = ModelParam(
    # Grid parameters
    h=(1 / 2) ** 5,  # Smallest element of the grid (power 6,7 or 8)
    dim_x=4,  # Dimensions of the grid
    dim_y=3 * (2 * np.pi) / (np.sqrt((20 - 1 + 0) / 3)),

    # Time parameters
    n=200,  # int(1e3),  # Number of time increments
    dt=5e-3,  # Time increment

    # Make sure that dt<h/v

    # Model parameters
    theta=20,  # Friction ratio beta'/beta , must be bigger than 1-alpha to grow
    alpha=0,  # Dimensionless activity, must be <1 to work
    vi=1,  # Inflow velocity, 0 or 1
    k=0,  # Growth parameter (0.1-10)
    Cahn=0.2,  # Cahn number
    Pe=50,  # Peclet number, should be big (about 1 / Cahn)
    starting_point=0,  # where the interface is at the beginning
    model_type='darcy',  # 'darcy' or 'toner-tu'

    # Initial perturbation parameters
    h_0=0.15,  # Amplitude of the initial perturbation
    k_wave=np.sqrt((20 - 1 + 0) / 3),  # Wave number of the initial perturbation
    # (theta-1+alpha) without division
    # (theta-1)*k*d_0 with division with d_0 = dim_x/2 + starting_point

    # Saving parameter
    folder_name='10-8-2020#1'
)

### Call function

# Save the parameters
folder_name1 = get_and_save_param(config1)
# folder_name2 = get_and_save_param(config2)
# folder_name3 = get_and_save_param(config3)
# folder_name4 = get_and_save_param(config4)
# folder_name5 = get_and_save_param(config5)
# folder_name6 = get_and_save_param(config6)
# folder_name7 = get_and_save_param(config7)
# folder_name8 = get_and_save_param(config8)

# Update folder_name to be the appropriate one
# This does not work when using MPI, must put the appropriate folder_name in ModelParam

config1.folder_name = folder_name1
# config2.folder_name = folder_name2
# config3.folder_name = folder_name3
# config4.folder_name = folder_name4
# config5.folder_name = folder_name5
# config6.folder_name = folder_name6
# config7.folder_name = folder_name7
# config8.folder_name = folder_name8

# Run the simulation
main_model(config1)
# main_model(config2)
# main_model(config3)
# main_model(config4)
# main_model(config5)
# main_model(config6)
# main_model(config7)
# main_model(config8)
