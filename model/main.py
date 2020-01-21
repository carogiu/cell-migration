### Import
import dolfin
import numpy as np

from model.model_flow import problem_coupled, space_flow
from model.model_phase import initiate_phase, space_phase, problem_phase_with_epsilon, solve_phase
from model.model_save_evolution import main_save, save_phi
from model.model_visu import main_visu

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

    # Compute the model
    phi_tot, vx_tot, vy_tot, p_tot = time_evolution(space_ME, w_flow, vi, theta, factor, epsilon, mid, dt, mob, n, mesh,
                                                    nx, ny, dim_x, dim_y)

    # TODO : save the arrays as csv files in the future

    # Plots
    main_visu(phi_tot, 'Phase', dim_x, dim_y)
    main_visu(vx_tot, 'Vx', dim_x, dim_y)
    # main_visu(vy_tot, 'Vy', dim_x, dim_y)
    # main_visu(p_tot, 'Pressure', dim_x, dim_y)

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
    # Unit square mesh
    # mesh = dolfin.UnitSquareMesh.create(nx, ny, dolfin.CellType.Type.quadrilateral)
    # Square mesh of dimension dim_x x dim_y
    mesh = dolfin.RectangleMesh(dolfin.Point(-dim_x / 2, 0.0), dolfin.Point(dim_x / 2, dim_y), nx, ny)
    return mesh


def time_evolution(space_ME, w_flow, vi, theta, factor, epsilon, mid, dt, mob, n, mesh, nx, ny, dim_x, dim_y):
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
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME, epsilon)
    # initiate the velocity and the pressure field
    velocity = dolfin.Expression((vi, "0.0"), degree=2)
    pressure = dolfin.Expression("1-x[0]", degree=1)
    # save the solutions
    phi_tot = np.zeros((nx + 1, ny + 1, n))
    vx_tot = np.zeros((nx + 1, ny + 1, n))
    vy_tot = np.zeros((nx + 1, ny + 1, n))
    p_tot = np.zeros((nx + 1, ny + 1, n))
    phi_tot, vx_tot, vy_tot, p_tot = interm_save(phi_tot, vx_tot, vy_tot, p_tot, u, velocity, pressure, 0, mesh, nx, ny)
    for i in range(1, n):
        F, J, u = problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, mob,
                                             epsilon, factor)
        u0.vector()[:] = u.vector()  # save the previous solution of u
        u = solve_phase(F, J, u, space_ME, dim_x, mesh)  # solve u for the next time step
        u_flow = problem_coupled(mesh, dim_x, dim_y, w_flow, phi, mu, vi, theta, factor,
                                 epsilon)  # solve the new velocity and pressure fields
        velocity, pressure = dolfin.split(u_flow)
        phi_tot, vx_tot, vy_tot, p_tot = main_save(phi_tot, vx_tot, vy_tot, p_tot, u, u_flow, i, mesh, nx, ny)
        print('Progress = ' + str(i + 1) + '/' + str(n))

    return phi_tot, vx_tot, vy_tot, p_tot


def interm_save(phi_tot, vx_tot, vy_tot, p_tot, u, velocity, pressure, i, mesh, nx, ny):
    """
    For a time i, extracts ux, uy, p from U_flow and the phase from u and saves them at the position i in the
    corresponding arrays
    @param phi_tot: array, contains all the values of phi for all the intermediate times
    @param vx_tot: array, contains all the values of vx for all the intermediate times
    @param vy_tot: array, contains all the values of vy for all the intermediate times
    @param p_tot: array, contains all the values of p for all the intermediate times
    @param u: dolfin Function
    @param pressure: dolfin Function
    @param velocity: dolfin Function
    @param i: int, number of the time step
    @param mesh: dolfin mesh
    @param nx: int, grid dimension
    @param ny: int, grid dimension
    @return: arrays
    """
    phi_tot = save_phi(phi_tot, u, i, mesh, nx, ny)
    arr_p = pressure.compute_vertex_values(mesh).reshape(nx + 1, ny + 1)
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, nx + 1, ny + 1))
    arr_ux = arr_u[0][::-1]
    arr_uy = arr_u[1][::-1]
    vx_tot[:, :, i] = arr_ux
    vy_tot[:, :, i] = arr_uy
    p_tot[:, :, i] = arr_p

    return phi_tot, vx_tot, vy_tot, p_tot
