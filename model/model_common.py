### Packages
import dolfin
import time

### Imports
from model.model_phase import initiate_phase, problem_phase_with_epsilon, solve_phase
from model.model_flow import problem_coupled, flow_with_activity
from model.model_save_evolution import save_HDF5


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


# No activity

def initiate_functions(space_ME: dolfin.function.functionspace.FunctionSpace, Cahn: float, h_0: float, k_wave: float,
                       starting_point: float, dim_x: int, theta: float, vi: str):
    """
    Initiates the test and trial functions for the phase, and the initial flow
    :param space_ME: function space of the phase
    :param Cahn: Cahn number
    :param h_0: amplitude of the initial instability
    :param k_wave: wave number of the initial instability
    :param starting_point: initial position of the interface
    :param dim_x: dimension in the direction of x
    :param theta: friction ratio
    :param vi: initial velocity
    :return: initiated functions
    """
    # Initiate the phase and the instability
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME=space_ME, Cahn=Cahn, h_0=h_0,
                                                                        k_wave=k_wave, starting_point=starting_point)
    # Initiate the velocity and pressure field
    velocity = dolfin.Expression((vi, "0.0"), degree=2)
    pressure = dolfin.Expression("x[0]> start ? theta*(dim_x/2 - x[0]) : theta*(dim_x/2 - start) + start - x[0]",
                                 degree=1, dim_x=dim_x, theta=theta, start=starting_point)
    return phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0, velocity, pressure


def main_solver(space_ME: dolfin.function.functionspace.FunctionSpace,
                w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: int, dim_y: int,
                mesh: dolfin.cpp.generation.RectangleMesh, phi_test, mu_test, du, u, phi, mu, phi_0, u0, mu_0, velocity,
                mid: float, dt: float, Pe: float, Cahn: float, theta: float, Ca: float, vi: str):
    """
    Main solver for the phase and the flow
    :param space_ME: function space for the phase
    :param w_flow: function space for the flow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_test: test function
    :param mu_test: test function
    :param du: trial function
    :param u: function
    :param phi: function
    :param mu: function
    :param phi_0: function
    :param u0: function
    :param mu_0: function
    :param velocity: function
    :param mid: for time scheme
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param theta: viscosity ratio
    :param Ca: Capillary number
    :param vi: initial velocity
    :return: solutions for the next step
    """

    t_1 = time.time()

    # First solve the phase
    F, J, u, bcs_phase = problem_phase_with_epsilon(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh,
                                                    phi_test=phi_test, mu_test=mu_test, du=du, u=u, phi=phi, mu=mu,
                                                    phi_0=phi_0, mu_0=mu_0, velocity=velocity, mid=mid, dt=dt,
                                                    Pe=Pe, Cahn=Cahn)
    u = solve_phase(F=F, J=J, u=u, bcs_phase=bcs_phase)  # solve phi, mu for the next time step

    # Update the value of phi(n), u(n), phi(n+1), mu(n+1)
    u0.vector()[:] = u.vector()
    phi_0, mu_0 = dolfin.split(u0)
    phi, mu = dolfin.split(u)
    t_3 = time.time()
    print('Time to solve phase = ' + str(t_3 - t_1) + ' seconds')

    # Then solve the flow
    u_flow = problem_coupled(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi, mu=mu, vi=vi, theta=theta,
                             Ca=Ca)
    velocity, pressure = u_flow.split()
    t_4 = time.time()
    print('Time to solve flow = ' + str(t_4 - t_3) + ' seconds')

    return u, phi, mu, u_flow, velocity, pressure


def time_evolution(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int, dt: float, n: int,
                   space_ME: dolfin.function.functionspace.FunctionSpace,
                   w_flow: dolfin.function.functionspace.FunctionSpace, theta: float, Cahn: float, Pe: float, Ca: float,
                   starting_point: float, h_0: float, k_wave: float, vi: str, mid: float, folder_name: str) -> None:
    """
    :param mesh: dolfin mesh
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

        ### Temporary
        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return


# With activity

def initiate_with_activity(space_ME: dolfin.function.functionspace.FunctionSpace, Cahn: float, h_0: float,
                           k_wave: float, starting: float, dim_x: int, theta: float, alpha: float, vi: str):
    """
    Initiates the test and trial functions for the phase, and the initial flow
    :param space_ME: function space of the phase
    :param Cahn: Cahn number
    :param h_0: amplitude of the initial instability
    :param k_wave: wave number of the initial instability
    :param starting: initial position of the interface
    :param dim_x: dimension in the direction of x
    :param theta: friction ratio
    :param alpha: activity
    :param vi: initial velocity
    :return: initiated functions
    """
    # Initiate the phase and the instability
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(space_ME=space_ME, Cahn=Cahn, h_0=h_0,
                                                                        k_wave=k_wave, starting_point=starting)

    # Initiate the velocity and pressure field
    velocity = dolfin.Expression((vi, "0.0"), degree=2)
    pressure = dolfin.Expression(
        "x[0]> start ? theta*(dim_x/2 - x[0]) : ((1-alpha)*(start-x[0]) + theta*(dim_x/2 - start))",
        degree=1, dim_x=dim_x, theta=theta, start=starting, alpha=alpha)
    return phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0, velocity, pressure


def main_solver_with_activity(space_ME: dolfin.function.functionspace.FunctionSpace,
                              w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: int, dim_y: int,
                              mesh: dolfin.cpp.generation.RectangleMesh, phi_test, mu_test, du, u, phi, mu, phi_0, u0,
                              mu_0, velocity, mid: float, dt: float, Pe: float, Cahn: float, theta: float, alpha: float,
                              Ca: float, vi: str):
    """
    Main solver for the phase and the flow
    :param space_ME: function space for the phase
    :param w_flow: function space for the flow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_test: test function
    :param mu_test: test function
    :param du: trial function
    :param u: function
    :param phi: function
    :param mu: function
    :param phi_0: function
    :param u0: function
    :param mu_0: function
    :param velocity: function
    :param mid: for time scheme
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param theta: viscosity ratio
    :param alpha: activity
    :param Ca: Capillary number
    :param vi: initial velocity
    :return: solutions for the next step
    """

    t_1 = time.time()

    # First solve the phase
    F, J, u, bcs_phase = problem_phase_with_epsilon(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh,
                                                    phi_test=phi_test, mu_test=mu_test, du=du, u=u, phi=phi, mu=mu,
                                                    phi_0=phi_0, mu_0=mu_0, velocity=velocity, mid=mid, dt=dt, Pe=Pe,
                                                    Cahn=Cahn)
    u = solve_phase(F=F, J=J, u=u, bcs_phase=bcs_phase)  # solve phi, mu for the next time step

    # Update the value of phi(n), u(n), phi(n+1), mu(n+1)
    u0.vector()[:] = u.vector()
    phi_0, mu_0 = dolfin.split(u0)
    phi, mu = dolfin.split(u)
    t_3 = time.time()
    print('Time to solve phase = ' + str(t_3 - t_1) + ' seconds')

    # Then solve the flow
    u_flow = flow_with_activity(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi, mu=mu, vi=vi, theta=theta,
                                alpha=alpha, Ca=Ca, v_previous=velocity)
    velocity, pressure = u_flow.split()
    t_4 = time.time()
    print('Time to solve flow = ' + str(t_4 - t_3) + ' seconds')

    return u, phi, mu, u0, phi_0, mu_0, u_flow, velocity, pressure


def time_evolution_with_activity(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int, dt: float, n: int,
                                 space_ME: dolfin.function.functionspace.FunctionSpace,
                                 w_flow: dolfin.function.functionspace.FunctionSpace, theta: float, alpha: float,
                                 Cahn: float, Pe: float, Ca: float, starting_point: float, h_0: float, k_wave: float,
                                 vi: str, mid: float, folder_name: str) -> None:
    """
    :param mesh: dolfin mesh
    :param dim_y: dimension in the direction of y
    :param dim_x: dimension in the direction of x
    :param n: number of time steps
    :param dt: time step
    :param space_ME: Function space, for the phase
    :param w_flow: Function space, for the flow
    :param theta: friction ratio
    :param alpha: activity
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
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0, velocity, pressure = initiate_with_activity(space_ME=space_ME,
                                                                                                    Cahn=Cahn, h_0=h_0,
                                                                                                    k_wave=k_wave,
                                                                                                    starting=starting_point,
                                                                                                    dim_x=dim_x,
                                                                                                    theta=theta,
                                                                                                    alpha=alpha, vi=vi)
    # Save the initial phase
    save_HDF5(function=u, function_name="Phi_and_mu", time_simu=0, folder_name=folder_name, mesh=mesh)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time
    for i in range(1, n):
        t_1 = time.time()

        # Solve everything
        u, phi, mu, u0, phi0, mu0, u_flow, velocity, pressure = main_solver_with_activity(space_ME=space_ME,
                                                                                          w_flow=w_flow,
                                                                                          dim_x=dim_x, dim_y=dim_y,
                                                                                          mesh=mesh, phi_test=phi_test,
                                                                                          mu_test=mu_test, du=du, u=u,
                                                                                          phi=phi, mu=mu, phi_0=phi_0,
                                                                                          u0=u0,
                                                                                          mu_0=mu_0,
                                                                                          velocity=velocity, mid=mid,
                                                                                          dt=dt, Pe=Pe,
                                                                                          Cahn=Cahn,
                                                                                          theta=theta, alpha=alpha,
                                                                                          Ca=Ca, vi=vi)

        # Save the solutions for the phase and the flow
        save_HDF5(function=u, function_name="Phi_and_mu", time_simu=i, folder_name=folder_name, mesh=mesh)
        save_HDF5(function=u_flow, function_name="V_and_P", time_simu=i, folder_name=folder_name, mesh=mesh)

        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return
