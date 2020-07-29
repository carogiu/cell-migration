### Packages
import dolfin
import time

### Imports
from model.model_phase import initiate_phase, problem_phase_implicit
from model.model_flow import flow_passive_darcy, flow_active_darcy, flow_passive_toner_tu, flow_active_toner_tu
from model.model_save_evolution import save_HDF5


def mesh_from_dim(nx: int, ny: int, dim_x: float, dim_y: float) -> dolfin.cpp.generation.RectangleMesh:
    """
    Creates mesh of dimension nx, ny of dimensions dim_x x dim_y

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :param dim_x : dimensions in x direction
    :param dim_y : dimensions in y direction
    :return: mesh
    """
    mesh = dolfin.RectangleMesh(dolfin.Point(-dim_x / 2, -dim_y / 2), dolfin.Point(dim_x / 2, dim_y / 2), nx, ny)
    return mesh


def initiate_functions(space_ME: dolfin.function.functionspace.FunctionSpace, Cahn: float, h_0: float, k_wave: float,
                       starting_point: float, vi: int):
    """
    Initiates the test and trial functions for the phase, and the initial flow
    :param space_ME: function space of the phase
    :param Cahn: Cahn number
    :param h_0: amplitude of the initial instability
    :param k_wave: wave number of the initial instability
    :param starting_point: initial position of the interface
    :param vi: initial velocity
    :return: initiated functions
    """
    # Initiate the phase and the instability

    u0, phi_0 = initiate_phase(space_ME=space_ME, Cahn=Cahn, h_0=h_0, k_wave=k_wave, starting_point=starting_point)

    # Initiate the velocity and pressure field
    velocity = dolfin.Expression(("vi", "0.0"), degree=2, vi=vi)

    return u0, phi_0, velocity


### All these functions are only used in the case where we solve separately the phase and the flow

# DARCY
## Passive
def main_solver_passive_darcy(space_ME: dolfin.function.functionspace.FunctionSpace,
                              w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                              mesh: dolfin.cpp.generation.RectangleMesh, phi_0, u0, velocity, dt: float, Pe: float,
                              Cahn: float, theta: float, Ca: float, vi: int):
    """
    Main solver for the phase and the flow
    :param space_ME: function space for the phase
    :param w_flow: function space for the flow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: function
    :param u0: function
    :param velocity: function
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param theta: viscosity ratio
    :param Ca: Capillary number
    :param vi: initial velocity
    :return: solutions for the next step
    """

    # First, solve the phase
    u_phase = problem_phase_implicit(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0,
                                     velocity=velocity, dt=dt, Pe=Pe, Cahn=Cahn)

    # Then, update the value of phi(n), u(n), phi(n+1), mu(n+1) (the values are stored in u0)
    u0.vector()[:] = u_phase.vector()
    phi_0, mu_0 = dolfin.split(u0)

    # Finally, solve the flow with the new values of phi and mu
    u_flow = flow_passive_darcy(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi_0, mu=mu_0, vi=vi,
                                theta=theta, Ca=Ca)
    velocity, pressure = u_flow.split()

    return u0, phi_0, u_flow, velocity, pressure


def time_evolution_passive_darcy_2_solvers(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int, dt: float,
                                           n: int, space_ME: dolfin.function.functionspace.FunctionSpace,
                                           w_flow: dolfin.function.functionspace.FunctionSpace, theta: float,
                                           Cahn: float, Pe: float, Ca: float, starting_point: float, h_0: float,
                                           k_wave: float, vi: int, folder_name: str) -> None:
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
    :param folder_name: name of the folder where files should be saved
    :return:
    """
    t_ini_1 = time.time()

    # Initiate the functions
    u0, phi_0, velocity = initiate_functions(space_ME=space_ME, Cahn=Cahn, h_0=h_0, k_wave=k_wave,
                                             starting_point=starting_point, vi=vi)

    # Save the initial phase
    save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=0, folder_name=folder_name, mesh=mesh)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time
    for i in range(1, n):
        t_1 = time.time()

        # Solve everything
        u0, phi_0, u_flow, velocity, pressure = main_solver_passive_darcy(space_ME=space_ME, w_flow=w_flow, dim_x=dim_x,
                                                                          dim_y=dim_y, mesh=mesh, phi_0=phi_0, u0=u0,
                                                                          velocity=velocity, dt=dt, Pe=Pe, Cahn=Cahn,
                                                                          theta=theta, Ca=Ca, vi=vi)

        # Save the solutions for the phase and the flow
        save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=i, folder_name=folder_name, mesh=mesh)
        save_HDF5(function=u_flow, function_name="V_and_P", time_simu=i, folder_name=folder_name, mesh=mesh)

        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return


## Active

def main_solver_active_darcy(space_ME: dolfin.function.functionspace.FunctionSpace,
                             w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: int, dim_y: int,
                             mesh: dolfin.cpp.generation.RectangleMesh, phi_0, u0, velocity, dt: float, Pe: float,
                             Cahn: float, theta: float, alpha: float, Ca: float, vi: int):
    """
    Main solver for the phase and the flow
    :param space_ME: function space for the phase
    :param w_flow: function space for the flow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: function
    :param u0: function
    :param velocity: function
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param theta: viscosity ratio
    :param alpha: activity
    :param Ca: Capillary number
    :param vi: initial velocity
    :return: solutions for the next step
    """

    # First, solve the phase
    u_phase = problem_phase_implicit(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0,
                                     velocity=velocity, dt=dt, Pe=Pe, Cahn=Cahn)

    # Then, update the value of phi(n), u(n), phi(n+1), mu(n+1)
    u0.vector()[:] = u_phase.vector()
    phi_0, mu_0 = dolfin.split(u0)

    # Finally, solve the flow
    u_flow = flow_active_darcy(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi_0, mu=mu_0, vi=vi,
                               theta=theta, alpha=alpha, Ca=Ca)
    velocity, pressure = u_flow.split()

    return u0, phi_0, u_flow, velocity, pressure


def time_evolution_active_darcy_2_solvers(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int, dt: float,
                                          n: int, space_ME: dolfin.function.functionspace.FunctionSpace,
                                          w_flow: dolfin.function.functionspace.FunctionSpace, theta: float,
                                          alpha: float, Cahn: float, Pe: float, Ca: float, starting_point: float,
                                          h_0: float, k_wave: float, vi: int, folder_name: str) -> None:
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
    :param folder_name: name of the folder where files should be saved
    :return:
    """
    t_ini_1 = time.time()

    # Initiate the functions
    u0, phi_0, velocity = initiate_functions(space_ME=space_ME, Cahn=Cahn, h_0=h_0, k_wave=k_wave,
                                             starting_point=starting_point, vi=vi)
    # Save the initial phase
    save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=0, folder_name=folder_name, mesh=mesh)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time
    for i in range(1, n):
        t_1 = time.time()

        # Solve everything
        u0, phi_0, u_flow, velocity, pressure = main_solver_active_darcy(space_ME=space_ME, w_flow=w_flow, dim_x=dim_x,
                                                                         dim_y=dim_y, mesh=mesh, phi_0=phi_0, u0=u0,
                                                                         velocity=velocity, dt=dt, Pe=Pe, Cahn=Cahn,
                                                                         theta=theta, alpha=alpha, Ca=Ca, vi=vi)

        # Save the solutions for the phase and the flow
        save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=i, folder_name=folder_name, mesh=mesh)
        save_HDF5(function=u_flow, function_name="V_and_P", time_simu=i, folder_name=folder_name, mesh=mesh)

        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return


# TONER-TU
## Passive
def main_solver_passive_toner_tu(space_ME: dolfin.function.functionspace.FunctionSpace,
                                 w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                                 mesh: dolfin.cpp.generation.RectangleMesh, phi_0, u0, velocity_n1, velocity_n2,
                                 dt: float, Pe: float, Cahn: float, theta: float, Ca: float, vi: int):
    """
    Main solver for the phase and the flow

    :param space_ME: function space for the phase
    :param w_flow: function space for the flow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: function
    :param u0: function
    :param velocity_n1: solution for the time-step n-1
    :param velocity_n2: solution for the time-step n-2
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param theta: viscosity ratio
    :param Ca: Capillary number
    :param vi: initial velocity
    :return: solutions for the next step
    """

    # First, solve the phase
    u_phase = problem_phase_implicit(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0,
                                     velocity=velocity_n1, dt=dt, Pe=Pe, Cahn=Cahn)

    # Then, update the value of phi(n), u(n), phi(n+1), mu(n+1) (the values are stored in u0)
    u0.vector()[:] = u_phase.vector()
    phi_0, mu_0 = dolfin.split(u0)

    # Finally, solve the flow with the new values of phi and mu
    u_flow = flow_passive_toner_tu(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi_0, mu=mu_0, vi=vi,
                                   theta=theta, Ca=Ca, velocity_previous=velocity_n1, velocity_pprevious=velocity_n2)
    velocity, pressure = u_flow.split()

    return u0, phi_0, u_flow, velocity, pressure


def time_evolution_passive_toner_tu_2_solvers(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int,
                                              dt: float, n: int, space_ME: dolfin.function.functionspace.FunctionSpace,
                                              w_flow: dolfin.function.functionspace.FunctionSpace, theta: float,
                                              Cahn: float, Pe: float, Ca: float, starting_point: float, h_0: float,
                                              k_wave: float, vi: int, folder_name: str) -> None:
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
    :param folder_name: name of the folder where files should be saved
    :return:
    """
    t_ini_1 = time.time()

    # Initiate the functions
    u0, phi_0, velocity_n1 = initiate_functions(space_ME=space_ME, Cahn=Cahn, h_0=h_0, k_wave=k_wave,
                                                starting_point=starting_point, vi=vi)
    velocity_n2 = dolfin.Expression(("vi", "0.0"), degree=2, vi=vi)

    # Save the initial phase
    save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=0, folder_name=folder_name, mesh=mesh)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time
    for i in range(1, n):
        t_1 = time.time()

        # Solve everything
        u0, phi_0, u_flow, velocity, pressure = main_solver_passive_toner_tu(space_ME=space_ME, w_flow=w_flow,
                                                                             dim_x=dim_x, dim_y=dim_y, mesh=mesh,
                                                                             phi_0=phi_0, u0=u0,
                                                                             velocity_n1=velocity_n1,
                                                                             velocity_n2=velocity_n2, dt=dt, Pe=Pe,
                                                                             Cahn=Cahn, theta=theta, Ca=Ca, vi=vi)

        # Save the solutions for the phase and the flow
        save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=i, folder_name=folder_name, mesh=mesh)
        save_HDF5(function=u_flow, function_name="V_and_P", time_simu=i, folder_name=folder_name, mesh=mesh)

        velocity_n2 = velocity_n1
        velocity_n1 = velocity

        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return


## Active

def main_solver_active_toner_tu(space_ME: dolfin.function.functionspace.FunctionSpace,
                                w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: int, dim_y: int,
                                mesh: dolfin.cpp.generation.RectangleMesh, phi_0, u0, velocity, dt: float, Pe: float,
                                Cahn: float, theta: float, alpha: float, Ca: float, vi: int):
    """
    Main solver for the phase and the flow
    :param space_ME: function space for the phase
    :param w_flow: function space for the flow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: function
    :param u0: function
    :param velocity: function
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param theta: viscosity ratio
    :param alpha: activity
    :param Ca: Capillary number
    :param vi: initial velocity
    :return: solutions for the next step
    """

    # First, solve the phase
    u_phase = problem_phase_implicit(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0,
                                     velocity=velocity, dt=dt, Pe=Pe, Cahn=Cahn)

    # Then, update the value of phi(n), u(n), phi(n+1), mu(n+1)
    u0.vector()[:] = u_phase.vector()
    phi_0, mu_0 = dolfin.split(u0)

    # Finally, solve the flow
    u_flow = flow_active_toner_tu(mesh=mesh, dim_x=dim_x, dim_y=dim_y, w_flow=w_flow, phi=phi_0, mu=mu_0, vi=vi,
                                  theta=theta, alpha=alpha, Ca=Ca, velocity_previous=velocity)
    velocity, pressure = u_flow.split()

    return u0, phi_0, u_flow, velocity, pressure


def time_evolution_active_tuner_tu_2_solvers(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int,
                                             dt: float, n: int, space_ME: dolfin.function.functionspace.FunctionSpace,
                                             w_flow: dolfin.function.functionspace.FunctionSpace, theta: float,
                                             alpha: float, Cahn: float, Pe: float, Ca: float, starting_point: float,
                                             h_0: float, k_wave: float, vi: int, folder_name: str) -> None:
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
    :param folder_name: name of the folder where files should be saved
    :return:
    """
    t_ini_1 = time.time()

    # Initiate the functions
    u0, phi_0, velocity = initiate_functions(space_ME=space_ME, Cahn=Cahn, h_0=h_0, k_wave=k_wave,
                                             starting_point=starting_point, vi=vi)
    # Save the initial phase
    save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=0, folder_name=folder_name, mesh=mesh)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time
    for i in range(1, n):
        t_1 = time.time()

        # Solve everything
        u0, phi_0, u_flow, velocity, pressure = main_solver_active_toner_tu(space_ME=space_ME, w_flow=w_flow,
                                                                            dim_x=dim_x, dim_y=dim_y, mesh=mesh,
                                                                            phi_0=phi_0, u0=u0, velocity=velocity,
                                                                            dt=dt, Pe=Pe, Cahn=Cahn, theta=theta,
                                                                            alpha=alpha, Ca=Ca, vi=vi)

        # Save the solutions for the phase and the flow
        save_HDF5(function=u0, function_name="Phi_and_mu", time_simu=i, folder_name=folder_name, mesh=mesh)
        save_HDF5(function=u_flow, function_name="V_and_P", time_simu=i, folder_name=folder_name, mesh=mesh)

        t_2 = time.time()
        print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return
