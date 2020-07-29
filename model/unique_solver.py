### Packages
import dolfin
import numpy as np
from ufl import dot, grad
import ufl
import time

### Imports
from model.model_domains import dom_and_bound
from model.model_save_evolution import save_HDF5


def space_common(mesh: dolfin.cpp.generation.RectangleMesh) -> dolfin.function.functionspace.FunctionSpace:
    """
    Creates a function space for the functions
    First function is a vector (vx, vy)
    Second function is a scalar (p)
    Third function is a scalar (phi)
    Fourth function is a scalar (mu)
    :param mesh: mesh
    :return: Function space
    """
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = dolfin.MixedElement([P2, P1, P1, P1])  # v, p, phi, mu
    mixed_space = dolfin.FunctionSpace(mesh, element)
    return mixed_space


# Darcy
def problem_passive(mixed_space: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                    mesh: dolfin.cpp.generation.RectangleMesh, phi_0: ufl.indexed.Indexed, dt: float, Pe: float,
                    Cahn: float, Ca: float, vi: int, theta: float):
    """
    Creates the variational problem for one time step and solves it, in the passive case
    :param mixed_space: Function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: solution of the phase from the previous time-step
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param Ca: Capillary number
    :param vi: inflow velocity
    :param theta: viscosity ratio
    :return: function with the new solutions
    """
    # Define the domain
    bcs, domain, boundaries = boundary_conditions(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, vi=vi)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Functions
    du = dolfin.TrialFunction(V=mixed_space)  # Used only to calculate the Jacobian

    v_test, p_test, phi_test, mu_test = dolfin.TestFunctions(V=mixed_space)
    u = dolfin.Function(mixed_space)  # new solution
    velocity, pressure, phi, mu = dolfin.split(u)

    # Transformation
    theta_c = dolfin.Constant(.5) * ((dolfin.Constant(1) - phi) + (dolfin.Constant(1) + phi) * dolfin.Constant(theta))

    # Variational problem (implicit time scheme)

    L0 = ((phi * phi_test - phi_0 * phi_test) * dolfin.Constant(1 / dt)
          + dolfin.Constant(1 / Pe) * (dot(grad(mu), grad(phi_test))) + phi_test * dot(velocity, grad(phi))) * dx

    L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - dolfin.Constant(Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx

    L2 = (theta_c * dot(velocity, v_test) + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu))
          + dot(v_test, grad(pressure)) - dot(velocity, grad(p_test))) * dx - dolfin.Constant(vi) * p_test * ds(1)

    F = L0 + L1 + L2

    J = dolfin.derivative(form=F, u=u, du=du)

    # Problem
    problem = dolfin.NonlinearVariationalProblem(F=F, u=u, bcs=bcs, J=J)
    solver = dolfin.NonlinearVariationalSolver(problem)

    # Solver
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-7
    prm["newton_solver"]["relative_tolerance"] = 1E-4
    prm["newton_solver"]["maximum_iterations"] = 1000
    prm["newton_solver"]["relaxation_parameter"] = 1.0
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    dolfin.PETScOptions.set("ksp_type", "gmres")
    dolfin.PETScOptions.set("ksp_monitor")
    dolfin.PETScOptions.set("pc_type", "ilu")

    solver.solve()
    return u


def problem_active(mixed_space: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                   mesh: dolfin.cpp.generation.RectangleMesh, phi_0: ufl.indexed.Indexed, dt: float, Pe: float,
                   Cahn: float, Ca: float, vi: int, theta: float, alpha: float):
    """
    Creates the variational problem for one time step and solves it, in the active case
    :param mixed_space: Function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: solution of the phase from the previous time-step
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param Ca: Capillary number
    :param vi: inflow velocity
    :param theta: viscosity ratio
    :param alpha: activity
    :return: function with the new solutions
    """
    # Define the domain
    bcs, domain, boundaries = boundary_conditions(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, vi=vi)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Functions
    du = dolfin.TrialFunction(V=mixed_space)  # Used only to calculate the Jacobian

    v_test, p_test, phi_test, mu_test = dolfin.TestFunctions(V=mixed_space)
    u = dolfin.Function(mixed_space)  # new solution
    velocity, pressure, phi, mu = dolfin.split(u)

    # Transformations
    theta_c = dolfin.Constant(.5) * ((dolfin.Constant(1) - phi) + (dolfin.Constant(1) + phi) * dolfin.Constant(theta))
    alpha_c = dolfin.Constant(alpha) * dolfin.Constant(.5) * (dolfin.Constant(1) - phi)

    # Norm of the velocity
    norm_sq = dot(velocity, velocity)
    unit_vector_velocity = velocity * ((norm_sq + dolfin.DOLFIN_EPS) ** (-0.5))  # This does not work in high activity

    # Variational problem (implicit time scheme)

    L0 = ((phi * phi_test - phi_0 * phi_test) * dolfin.Constant(1 / dt)
          + dolfin.Constant(1 / Pe) * (dot(grad(mu), grad(phi_test))) + phi_test * dot(velocity, grad(phi))) * dx

    L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - dolfin.Constant(Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx

    L2 = (theta_c * dot(velocity, v_test) + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu))
          + dot(v_test, grad(pressure)) - dot(velocity, grad(p_test))
          - alpha_c * dot(unit_vector_velocity, v_test)) * dx - dolfin.Constant(vi) * p_test * ds(1)

    F = L0 + L1 + L2

    J = dolfin.derivative(form=F, u=u, du=du)

    # Problem
    problem = dolfin.NonlinearVariationalProblem(F=F, u=u, bcs=bcs, J=J)
    # solver = dolfin.NonlinearVariationalSolver(problem)
    solver = SolverClass(problem=problem)

    # Solver
    # prm = solver.parameters
    # prm["newton_solver"]["absolute_tolerance"] = 1E-7
    # prm["newton_solver"]["relative_tolerance"] = 1E-4
    # prm["newton_solver"]["maximum_iterations"] = 1000
    # prm["newton_solver"]["relaxation_parameter"] = 1.0
    # dolfin.parameters["form_compiler"]["optimize"] = True
    # dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    # dolfin.PETScOptions.set("ksp_type", "gmres")
    # dolfin.PETScOptions.set("ksp_monitor")
    # dolfin.PETScOptions.set("pc_type", "ilu")

    dolfin.parameters["form_compiler"]["optimize"] = True

    dolfin.parameters["linear_algebra_backend"] = "PETSc"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    solver.solve()
    return u


# Toner-Tu (does not work yet, solver does not converge)
def problem_passive_toner_tu(mixed_space: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                             mesh: dolfin.cpp.generation.RectangleMesh, phi_0: ufl.indexed.Indexed, dt: float,
                             Pe: float, Cahn: float, Ca: float, vi: int, theta: float):
    """
    Creates the variational problem for one time step and solves it, in the passive case
    :param mixed_space: Function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: solution of the phase from the previous time-step
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param Ca: Capillary number
    :param vi: inflow velocity
    :param theta: viscosity ratio
    :return: function with the new solutions
    """
    # Define the domain
    bcs, domain, boundaries = boundary_conditions(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, vi=vi)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Functions
    du = dolfin.TrialFunction(V=mixed_space)  # Used only to calculate the Jacobian

    v_test, p_test, phi_test, mu_test = dolfin.TestFunctions(V=mixed_space)
    u = dolfin.Function(mixed_space)  # new solution
    velocity, pressure, phi, mu = dolfin.split(u)

    # Transformation
    theta_c = dolfin.Constant(.5) * ((dolfin.Constant(1) - phi) + (dolfin.Constant(1) + phi) * dolfin.Constant(theta))
    norm_sq = dot(velocity, velocity)

    # Variational problem (implicit time scheme)

    L0 = ((phi * phi_test - phi_0 * phi_test) * dolfin.Constant(1 / dt)
          + dolfin.Constant(1 / Pe) * (dot(grad(mu), grad(phi_test))) + phi_test * dot(velocity, grad(phi))) * dx

    L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - dolfin.Constant(Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx

    L2 = (theta_c * norm_sq * dot(velocity, v_test) + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu))
          + dot(v_test, grad(pressure)) - dot(velocity, grad(p_test))) * dx - dolfin.Constant(vi) * p_test * ds(1)

    F = L0 + L1 + L2

    J = dolfin.derivative(form=F, u=u, du=du)

    # Problem
    problem = dolfin.NonlinearVariationalProblem(F=F, u=u, bcs=bcs, J=J)
    solver = SolverClass(problem=problem)

    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["linear_algebra_backend"] = "PETSc"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    solver.solve()
    return u


def problem_active_toner_tu(mixed_space: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                            mesh: dolfin.cpp.generation.RectangleMesh, phi_0: ufl.indexed.Indexed, dt: float, Pe: float,
                            Cahn: float, Ca: float, vi: int, theta: float, alpha: float):
    """
    Creates the variational problem for one time step and solves it, in the active case
    :param mixed_space: Function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: solution of the phase from the previous time-step
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :param Ca: Capillary number
    :param vi: inflow velocity
    :param theta: viscosity ratio
    :param alpha: activity
    :return: function with the new solutions
    """
    # Define the domain
    bcs, domain, boundaries = boundary_conditions(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, vi=vi)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Functions
    du = dolfin.TrialFunction(V=mixed_space)  # Used only to calculate the Jacobian

    v_test, p_test, phi_test, mu_test = dolfin.TestFunctions(V=mixed_space)
    u = dolfin.Function(mixed_space)  # new solution
    velocity, pressure, phi, mu = dolfin.split(u)

    # Transformations
    theta_c = dolfin.Constant(.5) * ((dolfin.Constant(1) - phi) + (dolfin.Constant(1) + phi) * dolfin.Constant(theta))
    alpha_c = dolfin.Constant(alpha) * dolfin.Constant(.5) * (dolfin.Constant(1) - phi)

    # Variational problem (implicit time scheme)

    L0 = ((phi * phi_test - phi_0 * phi_test) * dolfin.Constant(1 / dt)
          + dolfin.Constant(1 / Pe) * (dot(grad(mu), grad(phi_test))) + phi_test * dot(velocity, grad(phi))) * dx

    L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - dolfin.Constant(Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx

    L2 = (theta_c * dot(velocity, velocity) * dot(velocity, v_test)
          + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu)) + dot(v_test, grad(pressure))
          - dot(velocity, grad(p_test)) - alpha_c * dot(velocity, v_test)) * dx - dolfin.Constant(vi) * p_test * ds(1)

    F = L0 + L1 + L2

    J = dolfin.derivative(form=F, u=u, du=du)

    # Problem
    problem = dolfin.NonlinearVariationalProblem(F=F, u=u, bcs=bcs, J=J)
    solver = SolverClass(problem=problem)

    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["linear_algebra_backend"] = "PETSc"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    solver.solve()
    return u


class SolverClass(dolfin.NonlinearVariationalSolver):
    def __init__(self, problem):
        dolfin.NonlinearVariationalSolver.__init__(self, problem)
        self.NewtonDict = {}
        SolverClass.set_parameters()
        self.setup_solver()

    @staticmethod
    def set_parameters():
        SolverClass.NewtonDict = {"nonlinear_solver": "newton",
                                  "newton_solver": {"absolute_tolerance": 1E-7,
                                                    "relative_tolerance": 1E-4,
                                                    "maximum_iterations": 500,
                                                    "relaxation_parameter": 1.0,
                                                    "linear_solver": "lu",
                                                    # "preconditioner": "jacobi",
                                                    "report": True,
                                                    "error_on_nonconvergence": True,
                                                    # "krylov_solver": {"absolute_tolerance": 1E-7,
                                                    #                   "relative_tolerance": 1E-4,
                                                    #                   "maximum_iterations": 1000,
                                                    #                   "monitor_convergence": True,
                                                    #                   "nonzero_initial_guess": False,
                                                    #                   }
                                                    }
                                  }

    def setup_solver(self):
        dolfin.PETScOptions.set("ksp_type", "gmres")
        dolfin.PETScOptions.set("ksp_monitor")
        dolfin.PETScOptions.set("pc_type", "ilu")
        self.parameters.update(SolverClass.NewtonDict)


def boundary_conditions(mixed_space: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float, vi: int,
                        mesh: dolfin.cpp.generation.RectangleMesh) -> [list, dolfin.cpp.mesh.MeshFunctionSizet]:
    """
    Sets the Dirichlet boundary conditions for all the problem
    :param mixed_space: Function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param vi: velocity inflow
    :param mesh: mesh
    :return: array of boundary conditions
    """
    domain, boundaries = dom_and_bound(mesh=mesh, dim_x=dim_x, dim_y=dim_y)

    # Boundary conditions for the fluid (inflow)
    inflow = dolfin.Expression(("vi", "0.0"), degree=2, vi=vi)
    bc_v_left = dolfin.DirichletBC(mixed_space.sub(0), inflow, boundaries, 1)

    # Boundary condition for the pressure
    pressure_out = dolfin.Constant(0.0)
    bc_p_right = dolfin.DirichletBC(mixed_space.sub(1), pressure_out, boundaries, 2)

    # Boundary conditions for the phase
    bc_phi_left = dolfin.DirichletBC(mixed_space.sub(2), dolfin.Constant(-1.0), boundaries, 1)  # phi = -1 on the left
    bc_phi_right = dolfin.DirichletBC(mixed_space.sub(2), dolfin.Constant(1.0), boundaries, 2)  # phi = +1 on the right

    # Boundary conditions for mu
    bc_mu_left = dolfin.DirichletBC(mixed_space.sub(3), dolfin.Constant(0.0), boundaries, 1)  # mu = 0 on left and right
    bc_mu_right = dolfin.DirichletBC(mixed_space.sub(3), dolfin.Constant(0.0), boundaries, 2)

    if vi == 0:
        # No slip condition
        v_null = dolfin.Expression(("0.0", "0.0"), degree=2)
        bc_v_top = dolfin.DirichletBC(mixed_space.sub(0), v_null, boundaries, 3)
        bc_v_bot = dolfin.DirichletBC(mixed_space.sub(0), v_null, boundaries, 4)

        # Boundary conditions
        bcs = [bc_v_left, bc_p_right, bc_phi_left, bc_phi_right, bc_mu_left, bc_mu_right, bc_v_top, bc_v_bot]

    else:
        # Boundary conditions
        bcs = [bc_v_left, bc_p_right, bc_phi_left, bc_phi_right, bc_mu_left, bc_mu_right]

    return bcs, domain, boundaries


def initiation_of_the_phase(h_0, k_wave, Cahn, start):
    phi_0 = dolfin.Expression("tanh((x[0] - start + h_0 * cos(x[1] * k_wave)) / (Cahn * sqrt(2)))", degree=1, h_0=h_0,
                              Cahn=Cahn, k_wave=k_wave, start=start)
    # For a binary phase at the beginning
    # phi_0 = dolfin.Expression("x[0] < start + h_0 * cos(x[1] * k_wave) ? -1 : 1", degree=1, h_0=h_0, k_wave=k_wave,
    #                           start=start)

    return phi_0


def time_evolution_general(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int, dt: float, n: int,
                           mixed_space: dolfin.function.functionspace.FunctionSpace, theta: float, Cahn: float,
                           Pe: float, Ca: float, starting_point: float, h_0: float, k_wave: float, vi: int,
                           alpha: float, folder_name: str) -> None:
    t_ini_1 = time.time()

    # Initiate the phase
    phi_0 = initiation_of_the_phase(h_0=h_0, k_wave=k_wave, Cahn=Cahn, start=starting_point)
    starting_time = 0

    # If we have to re-run the simulation from a different time-step
    # starting_time = 136
    # u = dolfin.Function(mixed_space)
    # file_path = "results/Figures/" + folder_name + "/Solutions/Functions_" + str(starting_time) + ".h5"
    # input_file = dolfin.HDF5File(mesh.mpi_comm(), file_path, 'r')
    # input_file.read(u, 'Solution Functions')
    # input_file.close()
    # _, _, phi_0, _ = u.split()

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time

    if alpha == 0:
        print('Simulation without activity')
        for i in range(starting_time + 1, n):
            t_1 = time.time()

            # Solve everything
            u = problem_passive(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0, dt=dt, Pe=Pe,
                                Cahn=Cahn, Ca=Ca, vi=vi, theta=theta)

            _, _, phi_0, _ = dolfin.split(u)

            # Save the solutions for the phase and the flow
            save_HDF5(function=u, function_name="Functions", time_simu=i, folder_name=folder_name, mesh=mesh)

            t_2 = time.time()
            print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')
    else:
        print('Simulation with activity')
        for i in range(starting_time + 1, n):
            t_1 = time.time()

            # Solve everything
            u = problem_active(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0, dt=dt, Pe=Pe,
                               Cahn=Cahn, Ca=Ca, vi=vi, theta=theta, alpha=alpha)

            _, _, phi_0, _ = dolfin.split(u)

            # Save the solutions for the phase and the flow
            save_HDF5(function=u, function_name="Functions", time_simu=i, folder_name=folder_name, mesh=mesh)

            t_2 = time.time()
            print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return


def time_evolution_toner_tu(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int, dt: float, n: int,
                            mixed_space: dolfin.function.functionspace.FunctionSpace, theta: float, Cahn: float,
                            Pe: float, Ca: float, starting_point: float, h_0: float, k_wave: float, vi: int,
                            alpha: float, folder_name: str) -> None:
    t_ini_1 = time.time()

    # Initiate the phase
    phi_0 = initiation_of_the_phase(h_0=h_0, k_wave=k_wave, Cahn=Cahn, start=starting_point)

    t_ini_2 = time.time()
    print('Initiation time = ' + str(t_ini_2 - t_ini_1) + ' seconds')

    # Solve through time

    if alpha == 0:
        print('Simulation without activity')
        for i in range(1, n):
            t_1 = time.time()

            # Solve everything
            u = problem_passive_toner_tu(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0,
                                         dt=dt, Pe=Pe,
                                         Cahn=Cahn, Ca=Ca, vi=vi, theta=theta)

            _, _, phi_0, _ = dolfin.split(u)

            # Save the solutions for the phase and the flow
            save_HDF5(function=u, function_name="Functions", time_simu=i, folder_name=folder_name, mesh=mesh)

            t_2 = time.time()
            print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')
    else:
        print('Simulation with activity')
        for i in range(1, n):
            t_1 = time.time()

            # Solve everything
            u = problem_active_toner_tu(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, phi_0=phi_0,
                                        dt=dt, Pe=Pe,
                                        Cahn=Cahn, Ca=Ca, vi=vi, theta=theta, alpha=alpha)

            _, _, phi_0, _ = dolfin.split(u)

            # Save the solutions for the phase and the flow
            save_HDF5(function=u, function_name="Functions", time_simu=i, folder_name=folder_name, mesh=mesh)

            t_2 = time.time()
            print('Progress = ' + str(i + 1) + '/' + str(n) + ', Computation time = ' + str(t_2 - t_1) + ' seconds')

    return

# This was used to solve the problem assuming that phi = phi_0 + delta_phi, with phi_0 the equilibrium solution (tanh),
# and solving on delta_phi. The other field are solved as whole (no LSA).
# It gives the same results as the rest, does no solve the discrepancy in the growth rate.

# def problem_passive_LSA(mixed_space: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
#                         mesh: dolfin.cpp.generation.RectangleMesh, phi_0: ufl.indexed.Indexed, dt: float, Pe: float,
#                         Cahn: float, Ca: float, vi: int, theta: float, i: int, start: float):
#     """
#     Creates the variational problem for one time step and solves it, in the passive case
#     :param mixed_space: Function space
#     :param dim_x: dimension in the direction of x
#     :param dim_y: dimension in the direction of y
#     :param mesh: mesh
#     :param phi_0: solution of the phase from the previous time-step
#     :param dt: time step
#     :param Pe: Peclet number
#     :param Cahn: Cahn number
#     :param Ca: Capillary number
#     :param vi: inflow velocity
#     :param theta: viscosity ratio
#     :return: function with the new solutions
#     """
#     # Define the domain
#     bcs, domain, boundaries = boundary_conditions(mixed_space=mixed_space, dim_x=dim_x, dim_y=dim_y, mesh=mesh, vi=vi)
#     dx = dolfin.dx(subdomain_data=domain)
#     ds = dolfin.ds(subdomain_data=boundaries)
#
#     # Functions
#     du = dolfin.TrialFunction(V=mixed_space)  # Used only to calculate the Jacobian
#
#     v_test, p_test, phi_test, mu_test = dolfin.TestFunctions(V=mixed_space)
#     u = dolfin.Function(mixed_space)  # new solution
#     velocity, pressure, d_phi, mu = dolfin.split(u)
#
#     # Linear solution for phi
#     phi_lin = dolfin.Expression("tanh((x[0] - start - vi * i * dt) / (Cahn * sqrt(2)))", degree=1, Cahn=Cahn, vi=vi,
#                                 i=i, dt=dt, start=start)
#     phi = dolfin.interpolate(phi_lin, mixed_space.sub(2).collapse()) + d_phi
#
#     # Transformation
#     theta_c = dolfin.Constant(.5) * ((dolfin.Constant(1) - phi) + (dolfin.Constant(1) + phi) * dolfin.Constant(theta))
#
#     # Variational problem (implicit time scheme)
#
#     L0 = ((phi * phi_test - phi_0 * phi_test) * dolfin.Constant(1 / dt) + dolfin.Constant(1 / Pe) * (
#         dot(grad(mu), grad(phi_test))) + phi_test * dot(velocity, grad(phi))) * dx
#
#     L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - dolfin.Constant(Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx
#
#     L2 = (theta_c * dot(velocity, v_test) + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu)) + dot(v_test, grad(
#         pressure)) - dot(velocity, grad(p_test))) * dx - dolfin.Constant(vi) * p_test * ds(1)
#
#     F = L0 + L1 + L2
#
#     J = dolfin.derivative(form=F, u=u, du=du)
#
#     # Problem
#     problem = dolfin.NonlinearVariationalProblem(F=F, u=u, bcs=bcs, J=J)
#     solver = dolfin.NonlinearVariationalSolver(problem)
#
#     # Solver
#     prm = solver.parameters
#     prm["newton_solver"]["absolute_tolerance"] = 1E-7
#     prm["newton_solver"]["relative_tolerance"] = 1E-4
#     prm["newton_solver"]["maximum_iterations"] = 1000
#     prm["newton_solver"]["relaxation_parameter"] = 1.0
#     dolfin.parameters["form_compiler"]["optimize"] = True
#     dolfin.parameters["form_compiler"]["cpp_optimize"] = True
#
#     dolfin.PETScOptions.set("ksp_type", "gmres")
#     dolfin.PETScOptions.set("ksp_monitor")
#     dolfin.PETScOptions.set("pc_type", "ilu")
#
#     solver.solve()
#     return u
