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
    mob = config.mob

    # Create Mesh
    mesh = mesh_from_dim(nx, ny)
    space_ME = space_phase(mesh)
    w_flow = space_flow(mesh)

    # Compute the model
    phi_tot, vx_tot, vy_tot, p_tot = time_evolution(space_ME, w_flow, vi, theta, factor, epsilon, mid, dt, mob, n, mesh, nx, ny)

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
    # Unit square mesh
    mesh = dolfin.UnitSquareMesh.create(nx, ny, dolfin.CellType.Type.quadrilateral)
    # Square mesh of dimension 100x100
    # mesh = dolfin.RectangleMesh(dolfin.Point(0, 0), dolfin.Point(5, 5), nx, ny)
    return mesh


def time_evolution(space_ME, w_flow, vi, theta, factor, epsilon, mid, dt, mob, n, mesh, nx, ny):
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
    @return: arrays, contain all the values of vx, vy, p and phi for all the intermediate times
    """
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME, epsilon)
    u_flow = problem_coupled(w_flow, phi, mu, vi, theta, factor, epsilon)
    velocity, pressure = dolfin.split(u_flow)
    # save the solutions
    phi_tot = np.zeros((nx + 1, ny + 1, n))
    vx_tot = np.zeros((nx + 1, ny + 1, n))
    vy_tot = np.zeros((nx + 1, ny + 1, n))
    p_tot = np.zeros((nx + 1, ny + 1, n))
    phi_tot, vx_tot, vy_tot, p_tot = main_save(phi_tot, vx_tot, vy_tot, p_tot, u, u_flow, 0, mesh, nx, ny)
    for i in range(1, n):
        a, L, u = problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, mob,
                                             epsilon)
        u0.vector()[:] = u.vector()
        u = solve_phase(a, L, u)
        u_flow = problem_coupled(w_flow, phi, mu, vi, theta, factor, epsilon)
        velocity, pressure = dolfin.split(u_flow)
        phi_tot, vx_tot, vy_tot, p_tot = main_save(phi_tot, vx_tot, vy_tot, p_tot, u, u_flow, i, mesh, nx, ny)
        print('Progress = ' + str(i+1)+'/'+str(n))

    return phi_tot, vx_tot, vy_tot, p_tot
