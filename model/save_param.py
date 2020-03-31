### Packages
import dolfin

### Imports
from model.model_parameter_class import save_param


def get_and_save_param(config):
    # Retrieve parameters

    # Grid parameters
    h = config.h
    nx, ny = config.nx, config.ny
    dim_x, dim_y = config.dim_x, config.dim_y

    # Time parameters
    n, dt = config.n, config.dt

    # Model parameters
    theta = config.theta
    alpha = config.alpha
    Cahn = config.Cahn
    Pe = config.Pe
    Ca = config.Ca
    starting_point = config.starting_point

    # Initial perturbation parameters
    h_0, k_wave = config.h_0, config.k_wave

    folder_name = save_param(h=h, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, n=n, dt=dt, theta=theta, Cahn=Cahn, Pe=Pe,
                             Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave, alpha=alpha)
    print(folder_name)
    return folder_name
