### Packages
import dolfin
import time

### Imports
from model.model_flow import problem_coupled, space_flow
from model.model_phase import initiate_phase, space_phase, problem_phase_with_epsilon, solve_phase
from model.model_save_evolution import main_save_fig, main_save_fig_interm
from model.model_parameter_class import save_param
from results.main_results import save_peaks  # , check_div_v, check_hydro

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

    # Create Mesh
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    space_ME = space_phase(mesh=mesh)
    w_flow = space_flow(mesh=mesh)

    print('Expected computation time = ' + str(nx * ny * n * 5E-4 / 60) + ' minutes')
    t1 = time.time()

    # save the parameters used
    folder_name = save_param(h=h, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, n=n, dt=dt, theta=theta,
                             Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave)

    # Compute the model
    time_evolution(mesh=mesh, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, space_ME=space_ME, w_flow=w_flow,
                   theta=theta, Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave, vi=vi,
                   mid=mid,
                   folder_name=folder_name)
    t2 = time.time()
    print('Total computation time = ' + str((t2 - t1) / 60) + ' minutes')

    return


### Initiate Mesh

def mesh_from_dim(nx: int, ny: int, dim_x: int, dim_y: int) -> dolfin.cpp.generation.RectangleMesh:
    """
    Creates mesh of dimension nx, ny of dimensions dim_x x dim_y

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :param dim_x : dimensions in x direction
    :param dim_y : dimensions in y direction
    :return: mesh
    """
    mesh = dolfin.RectangleMesh(dolfin.Point(-dim_x / 2, 0.0), dolfin.Point(dim_x / 2, dim_y), nx, ny)
    return mesh


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
    # initiate the phase and the instability
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME=space_ME, Cahn=Cahn, h_0=h_0,
                                                                        k_wave=k_wave, starting_point=starting_point)

    # initiate the velocity and the pressure field
    velocity = dolfin.Expression((vi, "0.0"), degree=2)
    pressure = dolfin.Expression("x[0]> start ? theta*(dim_x/2 - x[0]) : theta*(dim_x/2 - start) + start - x[0]",
                                 degree=1, dim_x=dim_x, theta=theta, start=starting_point)

    # save the solutions
    arr_interface = main_save_fig_interm(u=u, velocity=velocity, pressure=pressure, i=0, mesh=mesh, nx=nx, ny=ny,
                                         dim_x=dim_x, dim_y=dim_y, folder_name=folder_name, theta=theta)
    save_peaks(folder_name=folder_name, arr_interface=arr_interface, h_0=h_0)
    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # solve through time
    for i in range(1, n):
        t_1 = time.time()

        # First find phi(n+1) and mu(n+1) with the previous velocity

        F, J, u, bcs_phase = problem_phase_with_epsilon(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh,
                                                        phi_test=phi_test, mu_test=mu_test, du=du, u=u, phi=phi, mu=mu,
                                                        phi_0=phi_0, mu_0=mu_0, velocity=velocity, mid=mid, dt=dt,
                                                        Pe=Pe, Cahn=Cahn)
        u = solve_phase(F=F, J=J, u=u, bcs_phase=bcs_phase)  # solve phi, mu for the next time step

        # Update the value of phi(n), u(n), phi(n+1), mu(n+1)
        u0.vector()[:] = u.vector()
        phi_0, mu_0 = dolfin.split(u0)
        phi, mu = dolfin.split(u)

        # Then solve the flow
        u_flow = problem_coupled(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi, mu=mu, vi=vi, theta=theta,
                                 Ca=Ca)
        velocity, pressure = u_flow.split()

        # See div(v)
        # check_div_v(velocity=velocity, mesh=mesh, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y, time=i,
        #             folder_name=folder_name)
        # See hydrodynamics
        # check_hydro(velocity=velocity, pressure=pressure, u=u, theta=theta, Ca=Ca, mesh=mesh, nx=nx, ny=ny,
        #             dim_x=dim_x, dim_y=dim_y, folder_name=folder_name, time=i)

        # save figure in folder
        arr_interface = main_save_fig(u=u, u_flow=u_flow, i=i, mesh=mesh, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y,
                                      folder_name=folder_name, theta=theta)
        save_peaks(folder_name=folder_name, arr_interface=arr_interface, h_0=h_0)
        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return
