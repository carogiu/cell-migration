### Packages
import dolfin
import time

### Imports
from model.model_common import mesh_from_dim, initiate_functions, main_solver
from model.model_phase import space_phase
from model.model_flow import space_flow
from model.model_save_evolution import save_HDF5
from model.model_parameter_class import save_param

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
    h = config.h
    nx, ny = config.nx, config.ny
    dim_x, dim_y = config.dim_x, config.dim_y

    # Time parameters
    n, dt = config.n, config.dt

    # Model parameters
    theta = config.theta
    Cahn = config.Cahn
    Pe = config.Pe
    Ca = config.Ca
    starting_point = config.starting_point

    # Initial perturbation parameters
    h_0, k_wave = config.h_0, config.k_wave

    # Dimensionless parameters
    vi = config.vi
    mid = config.mid

    # Saving parameter
    folder_name = config.folder_name

    # Create Mesh
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    space_ME = space_phase(mesh=mesh)
    w_flow = space_flow(mesh=mesh)

    print('Expected computation time = ' + str(nx * ny * n * 5E-4 / 60) + ' minutes')  # 5e-4 on Mac 2e-4 on big Linux
    t1 = time.time()

    # Compute the model
    time_evolution(mesh=mesh, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, space_ME=space_ME, w_flow=w_flow,
                   theta=theta, Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave, vi=vi,
                   mid=mid,
                   folder_name=folder_name)
    t2 = time.time()
    print('Total computation time = ' + str((t2 - t1) / 60) + ' minutes')

    return


def time_evolution(mesh: dolfin.cpp.generation.RectangleMesh, nx: int, ny: int, dim_x: int, dim_y: int, dt: float,
                   n: int, space_ME: dolfin.function.functionspace.FunctionSpace,
                   w_flow: dolfin.function.functionspace.FunctionSpace, theta: float, Cahn: float, Pe: float, Ca: float,
                   starting_point: float, h_0: float, k_wave: float, vi: str, mid: float, folder_name: str) -> None:
    """
    :param mesh: dolfin mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_y: dimension in the direction of y
    :param dim_x: dimension in the direction of x
    :param n: number of time steps
    :param dt: time step
    :param space_ME: Function space, for the phase
    :param w_flow: Function space, for the flow
    :param theta: friction ratio
    :param Cahn: Cahn number
    :param Pe: Peclet number
    :param Ca: Capillary number
    :param starting_point : float, where the interface is at the beginning
    :param h_0: amplitude of the perturbation
    :param k_wave: wave number of the perturbation
    :param vi: Initial velocity
    :param mid: time scheme Crank Nicholson
    :param folder_name: name of the folder where files should be saved
    :return:
    """
    t_ini_1 = time.time()

    # Initiate the functions
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0, velocity, pressure = initiate_functions(space_ME=space_ME,
                                                                                                Cahn=Cahn, h_0=h_0,
                                                                                                k_wave=k_wave,
                                                                                                starting_point=starting_point,
                                                                                                dim_x=dim_x,
                                                                                                theta=theta, vi=vi)
    # Save the initial phase
    save_HDF5(function=u, function_name="Phi_and_mu", time_simu=0, folder_name=folder_name, mesh=mesh)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time
    for i in range(1, n):
        t_1 = time.time()

        # Solve everything
        u, phi, mu, u_flow, velocity, pressure = main_solver(space_ME=space_ME, w_flow=w_flow, dim_x=dim_x, dim_y=dim_y,
                                                             mesh=mesh, phi_test=phi_test, mu_test=mu_test, du=du, u=u,
                                                             phi=phi, mu=mu, phi_0=phi_0, u0=u0, mu_0=mu_0,
                                                             velocity=velocity, mid=mid, dt=dt, Pe=Pe, Cahn=Cahn,
                                                             theta=theta, Ca=Ca, vi=vi)

        # Save the solutions for the phase and the flow
        save_HDF5(function=u, function_name="Phi_and_mu", time_simu=i, folder_name=folder_name, mesh=mesh)
        save_HDF5(function=u_flow, function_name="V_and_P", time_simu=i, folder_name=folder_name, mesh=mesh)

        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return
