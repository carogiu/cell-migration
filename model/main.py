### Import
import dolfin
import numpy as np
import time

from model.model_flow import problem_coupled, space_flow
from model.model_phase import initiate_phase, space_phase, problem_phase_with_epsilon, solve_phase
from model.model_save_evolution import main_save_fig, main_save_fig_interm
from model.model_parameter_class import save_param

### Constants
dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize"] = True


### Main function

def main_model(config):
    """
    Run complete model from global main parameters and observe results.
    """
    # retrieve parameters
    nx, ny = config.nx, config.ny
    dim_x, dim_y = config.dim_x, config.dim_y
    n = config.n
    theta = config.theta
    factor = config.factor
    vi = config.vi
    epsilon = config.epsilon
    mid = config.mid
    dt = config.dt
    mob = config.mob

    # Create Mesh
    mesh = mesh_from_dim(nx, ny, dim_x, dim_y)
    space_ME = space_phase(mesh)
    w_flow = space_flow(mesh)

    print('Expected computation time = ' + str(nx * ny * n * (5E-4) / 60) + ' minutes')
    t1 = time.time()
    # save the parameters used
    folder_name = save_param(nx, ny, n, dim_x, dim_y, theta, epsilon, dt, mob, vi)
    # Compute the model
    time_evolution(space_ME, w_flow, vi, theta, factor, epsilon, mid, dt, mob, n, mesh,
                   nx, ny, dim_x, dim_y, folder_name)
    t2 = time.time()
    print('Total computation time = ' + str((t2 - t1) / 60) + ' minutes')

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


def time_evolution(space_ME, w_flow, vi, theta, factor, epsilon, mid, dt, mob, n, mesh, nx, ny, dim_x, dim_y,
                   folder_name):
    """

    @param space_ME: Function space, for the phase
    @param w_flow: Function space, for the flow
    @param vi: Expression, initial velocity
    @param theta: float, friction ratio
    @param factor: float, numerical factor
    @param epsilon: float, length scale ratio
    @param mid: float, time discretization Crank Nicholson
    @param dt: float, time step
    @param mob: float, energy value
    @param n: int, number of time steps
    @param mesh: dolfin mesh
    @param nx: int, grid dimension
    @param ny: int, grid dimension
    @param dim_y: int, dimension in the direction of y
    @param dim_x: int, dimension in the direction of x
    @return: arrays, contain all the values of vx, vy, p and phi for all the intermediate times
    """
    t_ini_1 = time.time()
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME, epsilon)
    # initiate the velocity and the pressure field
    velocity = dolfin.Expression((vi, "0.0"), degree=2)
    pressure = dolfin.Expression("dim_x/2-x[0]", degree=1, dim_x=dim_x)
    # save the solutions
    main_save_fig_interm(u, velocity, pressure, 0, mesh, nx, ny, dim_x, dim_y, folder_name)
    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')
    for i in range(1, n):
        t_1 = time.time()
        F, J, u = problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, mob,
                                             epsilon, factor)
        u0.vector()[:] = u.vector()  # save the previous solution of u
        u = solve_phase(F, J, u, space_ME, dim_x, mesh)  # solve u for the next time step
        u_flow = problem_coupled(mesh, dim_x, dim_y, w_flow, phi, mu, vi, theta, factor,
                                 epsilon)  # solve the new velocity and pressure fields
        velocity, pressure = dolfin.split(u_flow)
        # save figure in folder
        main_save_fig(u, u_flow, i, mesh, nx, ny, dim_x, dim_y, folder_name)
        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return
