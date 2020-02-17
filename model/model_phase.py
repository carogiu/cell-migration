import random
import dolfin
import numpy as np
from ufl import dot, grad
import ufl
from model.model_domains import dom_and_bound


### Initialise the phase
class InitialConditions(dolfin.UserExpression):  # result is a dolfin Expression
    """
    Creates the initial condition for the phase
    """

    def __init__(self, Cahn, h_0, k_wave, **kwargs):
        random.seed(2)  # + MPI.rank(MPI.comm_world))  # need to seed random number
        self.Cahn = Cahn
        self.h_0 = h_0
        self.k_wave = k_wave
        super().__init__(**kwargs)

    def eval(self, values, x):
        Cahn = float(self.Cahn)
        h_0 = float(self.h_0)
        k_wave = float(self.k_wave)
        if abs(x[0]) <= h_0 * 1.5:
            # random perturbation
            # h = np.random.randn(1) * Cahn
            # sin perturbation
            dx = h_0 * np.sin(x[1] * k_wave)
            values[0] = np.tanh((x[0] + dx) / (Cahn * np.sqrt(2)))  # phi(0)

        else:
            values[0] = np.tanh((x[0]) / (Cahn * np.sqrt(2)))

        if abs(x[0]) <= h_0 * 1.5:
            dx = h_0 * np.sin(x[1] * k_wave)
            dx_prime = h_0 * k_wave * np.cos(x[1] * k_wave)
            phi = np.tanh((x[0] + dx) / (Cahn * np.sqrt(2)))
            values[1] = (Cahn * dx / np.sqrt(2)) * (k_wave ** 2) * (1 - phi ** 2) + dx_prime ** 2 * phi * (1 - phi ** 2)

        else:
            values[1] = 0.0  # mu(0) outside of the perturbation

    def value_shape(self):
        return (2,)  # dimension 2 (phi,mu)


### Main functions
def space_phase(mesh: dolfin.cpp.generation.RectangleMesh) -> dolfin.function.functionspace.FunctionSpace:
    """
    Returns the function space for the phase and mu
    :param mesh: dolfin mesh
    :return: Function space
    """
    element_P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    space_ME = dolfin.FunctionSpace(mesh, element_P1 * element_P1)
    return space_ME


def initiate_phase(space_ME: dolfin.function.functionspace.FunctionSpace, Cahn: float, h_0: float, k_wave=float):
    """
    Initiate the phase : from the function space, creates trial functions and test functions, and applies the
    initial conditions
    :param space_ME: Function space
    :param Cahn: Cahn number
    :param h_0: amplitude of the sin
    :param k_wave: wave number of the sin
    :return: Functions
    """
    du = dolfin.TrialFunction(V=space_ME)
    phi_test, mu_test = dolfin.TestFunctions(V=space_ME)

    u = dolfin.Function(space_ME)  # current solution u
    u0 = dolfin.Function(space_ME)  # solution from previous converged step u0

    u_init = InitialConditions(degree=1, Cahn=Cahn, h_0=h_0, k_wave=k_wave)
    u.interpolate(u_init)
    u0.interpolate(u_init)

    phi, mu = dolfin.split(u)
    phi_0, mu_0 = dolfin.split(u0)

    return phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0


def problem_phase_with_epsilon(space_ME: dolfin.function.functionspace.FunctionSpace, dim_x: int, dim_y: int,
                               mesh: dolfin.cpp.generation.RectangleMesh, phi_test: ufl.indexed.Indexed,
                               mu_test: ufl.indexed.Indexed, du: dolfin.function.argument.Argument,
                               u: dolfin.function.function.Function, phi: ufl.indexed.Indexed, mu: ufl.indexed.Indexed,
                               phi_0: ufl.indexed.Indexed, mu_0: ufl.indexed.Indexed,
                               velocity: dolfin.function.function.Function, mid: float, dt: float, Pe: float,
                               Cahn: float) -> [ufl.form.Form, ufl.form.Form, dolfin.function.function.Function, list]:
    """
    Creates the variational problem
    :param space_ME: function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_test: Test function
    :param mu_test: Test function
    :param du: Trial function
    :param u: Function, current solution
    :param phi: Function, current solution
    :param mu: Function, current solution
    :param phi_0: Function, previous solution
    :param mu_0: Function, previous solution
    :param velocity: Expression, velocity of the flow for each point of the mesh
    :param mid: for the time scheme
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :return: Functions
    """
    # intermediate mu
    # mu_mid = mu_calc(mid=mid, mu=mu, mu_0=mu_0)
    # define the domain
    bcs_phase, domain = boundary_conditions_phase(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)
    # variational problem

    L0 = (phi * phi_test - phi_0 * phi_test +
          dt * mid * (dot(grad(mu), grad(phi_test))) / Pe +
          dt * mid * phi_test * dot(velocity, grad(phi)))*dx
          #dt * (1 - mid) * (dot(grad(mu_0), grad(phi_test))) / Pe +
          #dt * (1 - mid) * phi_test * dot(velocity, grad(phi_0))) * dx

    L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - (Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx

    F = L0 + L1

    J = dolfin.derivative(form=F, u=u, du=du)

    return F, J, u, bcs_phase


def solve_phase(F: ufl.form.Form, J: ufl.form.Form, u: dolfin.function.function.Function,
                bcs_phase: list) -> dolfin.function.function.Function:
    """
    Solves the variational problem
    @param F: Function (residual)
    @param J: Function (Jacobian)
    @param u: Function
    @param bcs_phase: boundary conditions
    @return: Function
    """
    # Problem
    problem_phase = dolfin.NonlinearVariationalProblem(F=F, u=u, bcs=bcs_phase, J=J)
    solver_phase = dolfin.NonlinearVariationalSolver(problem_phase)

    # Solver
    prm = solver_phase.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-7
    prm["newton_solver"]["relative_tolerance"] = 1E-4
    prm["newton_solver"]["maximum_iterations"] = 1000
    prm["newton_solver"]["relaxation_parameter"] = 1.0
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    solver_phase.solve()
    return u


### BOUNDARIES### CREATE BOUNDARIES
def boundary_conditions_phase(space_ME: dolfin.function.functionspace.FunctionSpace, dim_x: int, dim_y: int,
                              mesh: dolfin.cpp.generation.RectangleMesh) -> [list, dolfin.cpp.mesh.MeshFunctionSizet]:
    """
    Creates the boundary conditions  : no slip condition, velocity inflow, pressure out
    :param space_ME: Function space
    :param dim_x: dimensions in the direction of x
    :param dim_y: dimensions in the direction of y
    :param mesh: mesh
    :return: array of boundary conditions
    """
    domain, boundaries = dom_and_bound(mesh=mesh, dim_x=dim_x, dim_y=dim_y)
    # boundary conditions for the phase
    bc_phi_left = dolfin.DirichletBC(space_ME.sub(0), dolfin.Constant(-1.0), boundaries, 1)  # phi = -1 on the left (1)
    bc_phi_right = dolfin.DirichletBC(space_ME.sub(0), dolfin.Constant(1.0), boundaries, 2)  # phi = +1 on the right (2)
    # boundary conditions for mu
    bc_mu_left = dolfin.DirichletBC(space_ME.sub(1), dolfin.Constant(0.0), boundaries, 1)  # mu = 0 on left and right
    bc_mu_right = dolfin.DirichletBC(space_ME.sub(1), dolfin.Constant(0.0), boundaries, 2)
    bcs_phase = [bc_phi_left, bc_phi_right, bc_mu_left, bc_mu_right]

    return bcs_phase, domain


### Utilitarian functions
def mu_calc(mid: float, mu: ufl.indexed.Indexed, mu_0: ufl.indexed.Indexed):
    """
    Time discretization (Crank Nicholson method)
    :param mid: float, time scheme
    :param mu: Function, current solution
    :param mu_0: Function, previous solution
    :return: Function
    """
    return (1.0 - mid) * mu_0 + mid * mu


### NOT USED ANYMORE
"""
#     (For the example) Defines the potential
def potential(phi):
    phi = dolfin.variable(phi)
    f = 100 * phi ** 2 * (1 - phi) ** 2
    df_dphi = dolfin.diff(f, phi)

    return df_dphi


### TEST FUNCTIONS

#     Defines the problem for the example
def problem_phase_old(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, lmbda, dt):

    mu_mid = mu_calc(mid, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * dot(velocity, grad(phi_0)) * phi_test * dx + dt * dot(
        grad(mu_mid), grad(phi_test)) * dx
    L1 = mu * mu_test * dx - (phi ** 3 - phi) * mu_test * dx - lmbda * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = dolfin.derivative(L, u, du)
    return a, L, u
    
### Initialise Problem resolution (not used anymore)
class CahnHilliardEquation(dolfin.NonlinearProblem):

    def __init__(self, a, L):
        dolfin.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        dolfin.assemble(self.L, tensor=b)  # computes the residual vector b

    def J(self, A, x):
        dolfin.assemble(self.a, tensor=A)  # computes the Jacobian matrix A    
    
"""
