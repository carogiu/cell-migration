### Packages
from model.main import main_model
from model.model_parameter_class import ModelParam
import numpy as np

# import fenics

### Parameters
config = ModelParam(
    # Grid parameters
    h=(1 / 2) ** 5,  # Smallest element of the grid (power 6,7 or 8)
    dim_x=10,  # Dimensions of the grid
    dim_y=50,

    # Time parameters
    n=int(2e3),  # Number of time increments
    dt=4e-3,  # Time increment (should not be bigger than 0.01)

    # Make sure that dt<h*h*Pe/2 and dt<h/v

    # Model parameters
    theta=5,  # Friction ratio beta'/beta , must be bigger than 1!
    Cahn=0.12,  # Cahn number (need 0.5h < Cahn < 2h)
    Pe=10,  # Peclet number, should be big (about 1 / Cahn) , can try up to 200
    starting_point=-3.5,  # where the interface is at the beginning

    # Initial perturbation parameters
    h_0=0.15,  # 0.15,  # Amplitude of the initial perturbation
    k_wave=np.sqrt((5 - 1) / 3),  # Wave number of the initial perturbation
)

# fenics.list_linear_solver_methods()
# fenics.list_krylov_solver_preconditioners()

### Call function
main_model(config)
