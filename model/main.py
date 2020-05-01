### Packages
import dolfin
import time

### Imports
from model.model_common import mesh_from_dim, time_evolution, time_evolution_with_activity
from model.model_phase import space_phase
from model.model_flow import space_flow

### Constants
dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize"] = True


### Main function

def main_model(config):
    """
    Run complete model from global main parameters and observe results.
    """
    # Retrieve parameters

    # Grid parameters
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

    # Dimensionless parameters
    vi = config.vi

    # Saving parameter
    folder_name = config.folder_name

    # Create Mesh
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    space_ME = space_phase(mesh=mesh)
    w_flow = space_flow(mesh=mesh)

    print('Expected computation time = ' + str(nx * ny * n * 5E-4 / 60) + ' minutes')  # 5e-4 on Mac 2e-4 on big Linux
    t1 = time.time()

    # Compute the model

    if alpha == 0:
        time_evolution(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, space_ME=space_ME, w_flow=w_flow, theta=theta,
                       Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave, vi=vi,
                       folder_name=folder_name)
    else:
        time_evolution_with_activity(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, space_ME=space_ME, w_flow=w_flow,
                                     theta=theta, alpha=alpha, Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point,
                                     h_0=h_0, k_wave=k_wave, vi=vi, folder_name=folder_name)

    t2 = time.time()
    print('Total computation time = ' + str((t2 - t1) / 60) + ' minutes')

    return
