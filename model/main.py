### Import
import dolfin
import numpy as np

from model.model_flow import problem_coupled, space_flow
from model.model_phase import initiate_phase, space_phase, problem_phase_with_epsilon, solve_phase
from model.model_save_evolution import main_save
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
    n = config.n
    theta = config.theta
    factor = config.factor
    vi = config.vi
    epsilon = config.epsilon
    mid = config.mid
    dt = config.dt
    M = config.M

    # Create Mesh
    mesh = mesh_from_dim(nx, ny)
    ME = space_phase(mesh)
    W_flow = space_flow(mesh)

    # Compute the model
    phi_tot, vx_tot, vy_tot, p_tot = time_evolution(ME, W_flow, vi, theta, factor, epsilon, mid, dt, M, n, mesh)

    # TODO : save the arrays as csv files in the future

    # Plots
    main_visu(phi_tot, 'Phase')
    # main_visu(vx_tot, 'Vx')
    # main_visu(vy_tot, 'Vy')
    # main_visu(p_tot, 'Pressure')

    return


### Initiate Mesh

def mesh_from_dim(nx, ny):
    """
    Creates mesh of dimension nx, ny of type quadrilateral

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :return: mesh
    """
    mesh = dolfin.UnitSquareMesh.create(nx, ny, dolfin.CellType.Type.quadrilateral)
    return mesh


def time_evolution(ME, W_flow, vi, theta, factor, epsilon, mid, dt, M, n, mesh):
    """

    @param ME: Function space, for the phase
    @param W_flow: Function space, for the flow
    @param vi: Expression, initial velocity
    @param theta: float, friction ratio
    @param factor: float, numerical factor
    @param epsilon: float, length scale ratio
    @param mid: float, time discretization Crank Nicholson
    @param dt: float, time step
    @param M: float, energy value
    @param n: int, number of time steps
    @param mesh: dolfin mesh
    @return: arrays, contain all the values of vx, vy, p and phi for all the intermediate times
    """
    nx = ny = int(np.sqrt(mesh.num_cells()))
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(ME)
    U_flow = problem_coupled(W_flow, phi, mu, vi, theta, factor, epsilon)
    velocity, pressure = dolfin.split(U_flow)
    # save the solutions
    phi_tot = np.zeros((nx + 1, ny + 1, n))
    vx_tot = np.zeros((nx + 1, ny + 1, n))
    vy_tot = np.zeros((nx + 1, ny + 1, n))
    p_tot = np.zeros((nx + 1, ny + 1, n))
    phi_tot, vx_tot, vy_tot, p_tot = main_save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, 0, mesh)
    for i in range(1, n):
        a, L, u = problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, M,
                                             epsilon)
        u0.vector()[:] = u.vector()
        u = solve_phase(a, L, u)
        U_flow = problem_coupled(W_flow, phi, mu, vi, theta, factor, epsilon)
        velocity, pressure = dolfin.split(U_flow)
        phi_tot, vx_tot, vy_tot, p_tot = main_save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i, mesh)

    return phi_tot, vx_tot, vy_tot, p_tot
