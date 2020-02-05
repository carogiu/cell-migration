### Import
import dolfin
import time
import numpy as np

from model.model_flow import problem_coupled, space_flow
from model.model_phase import initiate_phase, space_phase, problem_phase_with_epsilon, solve_phase
from model.model_save_evolution import main_save_fig, main_save_fig_interm
from model.model_parameter_class import save_param
from results.main_results import save_peaks

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
    # Initial perturbation parameters
    h_0 = config.h_0
    k_wave = config.k_wave
    # Dimensionless parameters
    vi = config.vi
    mid = config.mid

    # Create Mesh
    mesh = mesh_from_dim(nx, ny, dim_x, dim_y)
    space_ME = space_phase(mesh)
    w_flow = space_flow(mesh)

    # print('Expected computation time = ' + str(nx * ny * n * 5E-4 / 60) + ' minutes')
    t1 = time.time()
    # save the parameters used
    folder_name = save_param(h, dim_x, dim_y, nx, ny, n, dt, theta, Cahn, Pe, Ca, h_0, k_wave)
    # Compute the model
    time_evolution(mesh, nx, ny, dim_x, dim_y, dt, n, space_ME, w_flow, theta, Cahn, Pe, Ca, h_0, k_wave, vi, mid,
                   folder_name)
    t2 = time.time()
    print('Total computation time = ' + str((t2 - t1) / 60) + ' minutes')

    # main_distance_save(folder_name, arr_interface_tot, n)
    return


### Initiate Mesh

def mesh_from_dim(nx, ny, dim_x, dim_y):
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


def time_evolution(mesh, nx, ny, dim_x, dim_y, dt, n, space_ME, w_flow, theta, Cahn, Pe, Ca, h_0, k_wave, vi, mid,
                   folder_name):
    """
    :param mesh: dolfin mesh
    :param nx: int, grid dimension
    :param ny: int, grid dimension
    :param dim_y: int, dimension in the direction of y
    :param dim_x: int, dimension in the direction of x
    :param n: int, number of time steps
    :param dt: float, time step
    :param space_ME: Function space, for the phase
    :param w_flow: Function space, for the flow
    :param theta: float, friction ratio
    :param Cahn: float, Cahn number
    :param Pe: float, Peclet number
    :param Ca: float, Capillary number
    :param h_0: float, amplitude of the perturbation
    :param k_wave: float, wave number of the perturbation
    :param vi: Expression, initial velocity
    :param mid: float, time scheme Crank Nicholson
    :param folder_name: string, name of the folder where files should be saved
    :return: arrays, contain all the values of vx, vy, p and phi for all the intermediate times
    """
    t_ini_1 = time.time()
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME, Cahn, h_0, k_wave)
    # initiate the velocity and the pressure field
    velocity = dolfin.Expression((vi, "0.0"), degree=2)
    pressure = dolfin.Expression("dim_x/2 - x[0]", degree=1, dim_x=dim_x)
    # save the solutions
    arr_interface = main_save_fig_interm(u, velocity, pressure, 0, mesh, nx, ny, dim_x, dim_y, folder_name)
    save_peaks(folder_name, arr_interface, h_0)
    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')
    for i in range(1, n):
        t_1 = time.time()

        # First solve the flow and the pressure with phi_0 and mu_0
        u_flow = problem_coupled(mesh, dim_x, dim_y, w_flow, phi, mu, vi, theta, Ca)
        velocity, pressure = dolfin.split(u_flow)

        # Then solve the phase to get phi and mu (time n+1)
        F, J, u, bcs_phase = problem_phase_with_epsilon(space_ME, dim_x, dim_y, mesh, phi_test, mu_test, du, u, phi, mu,
                                                        phi_0, mu_0, velocity, mid, dt, Pe, Cahn)
        u = solve_phase(F, J, u, bcs_phase)  # solve u for the next time step

        # Finally update the value of phi_0 and u_0
        u0.vector()[:] = u.vector()
        phi_0, mu_0 = dolfin.split(u0)

        # save figure in folder
        arr_interface = main_save_fig(u, u_flow, i, mesh, nx, ny, dim_x, dim_y, folder_name)
        save_peaks(folder_name, arr_interface, h_0)
        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return
