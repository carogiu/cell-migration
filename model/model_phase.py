import random
import dolfin
import numpy as np
from ufl import dx, dot, grad


### Initialise the phase
class InitialConditions(dolfin.UserExpression):  # result is a dolfin Expression
    """
    Creates the initial condition for the phase
    """

    def __init__(self, epsilon, **kwargs):
        random.seed(2)  # + MPI.rank(MPI.comm_world))  # need to seed random number
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def eval(self, values, x):
        ep = float(self.epsilon)
        if abs(x[0] - .5) < ep / 2:
            # random perturbation
            # values[0] = np.tanh(((x[0] - .5) * 2) / (ep * np.sqrt(2))) + np.random.randn(1) * 0.05 # phi(0)
            # sin perturbation
            values[0] = np.tanh((x[0] - .5) / (ep * np.sqrt(2))) + np.sin(x[1] * 30) * 0.3

        else:
            values[0] = np.tanh((x[0] - .5) / (ep * np.sqrt(2)))
        values[1] = 0.0  # mu(0)

    def value_shape(self):
        return (2,)  # dimension 2 (phi,mu)


### Initialise Problem resolution
class CahnHilliardEquation(dolfin.NonlinearProblem):
    """
    Creates the problem, initiates the residual vector and the Jacobian matrix
    """

    def __init__(self, a, L):
        dolfin.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        dolfin.assemble(self.L, tensor=b)  # computes the residual vector b

    def J(self, A, x):
        dolfin.assemble(self.a, tensor=A)  # computes the Jacobian matrix A


### Main functions
def space_phase(mesh):
    """
    Returns the function space for the phase and mu
    @param mesh: dolfin mesh
    @return: Function space
    """
    element_P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    space_ME = dolfin.FunctionSpace(mesh, element_P1 * element_P1)
    return space_ME


def initiate_phase(space_ME, epsilon):
    """
    Initiate the phase : from the function space, creates trial functions and test functions, and applies the
    initial conditions
    @param space_ME: Function space
    @param epsilon: float, length ratio
    @return: Functions
    """
    du = dolfin.TrialFunction(space_ME)
    phi_test, mu_test = dolfin.TestFunctions(space_ME)

    u = dolfin.Function(space_ME)  # current solution u
    u0 = dolfin.Function(space_ME)  # solution from previous converged step u0

    u_init = InitialConditions(degree=1, epsilon=epsilon)
    u.interpolate(u_init)
    u0.interpolate(u_init)

    phi, mu = dolfin.split(u)
    phi_0, mu_0 = dolfin.split(u0)

    return phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0


def problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, mob, epsilon):
    """
    Creates the variational problem
    @param phi_test: Test function
    @param mu_test: Test function
    @param du: Trial function
    @param u: Function, current solution
    @param phi: Function, current solution
    @param mu: Function, current solution
    @param phi_0: Function, previous solution
    @param mu_0: Function, previous solution
    @param velocity: Expression, velocity of the flow for each point of the mesh
    @param mid: float, for the time discretization
    @param dt: float, time step
    @param mob: float, energy factor
    @param epsilon: float, length ratio
    @return: Functions
    """
    mu_mid = mu_calc(mid, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * phi_test * dot(velocity, grad(phi_0)) * dx + dt * (
            mob * epsilon ** 2) * dot(
        grad(mu_mid), grad(phi_test)) * dx
    L1 = mu * mu_test * dx - (phi ** 3 - phi) * mu_test * dx - (epsilon ** 2) * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = dolfin.derivative(L, u, du)

    return a, L, u


def solve_phase(a, L, u):
    """
    Solves the variational problem
    @param a: Function
    @param L: Function
    @param u: Function
    @return: Function
    """
    problem_phase = CahnHilliardEquation(a, L)
    solver_phase = dolfin.NewtonSolver()
    solver_phase.parameters["linear_solver"] = "lu"
    solver_phase.parameters["convergence_criterion"] = "incremental"
    solver_phase.parameters["absolute_tolerance"] = 1e-7
    solver_phase.parameters["relative_tolerance"] = 1e-4
    solver_phase.parameters["maximum_iterations"] = 1000
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    solver_phase.solve(problem_phase, u.vector())
    return u


### Utilitarian functions
def mu_calc(mid, mu, mu_0):
    """
    Time discretization (Crank Nicholson method)
    @param mid: float, time discretization
    @param mu: Function, current solution
    @param mu_0: Function, previous solution
    @return: Function
    """
    return (1.0 - mid) * mu_0 + mid * mu


def potential(phi):
    """
    (For the example) Defines the potential
    @param phi: Function
    @return: Function
    """
    phi = dolfin.variable(phi)
    f = 100 * phi ** 2 * (1 - phi) ** 2
    df_dphi = dolfin.diff(f, phi)

    return df_dphi


### TEST FUNCTIONS
def problem_phase(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, lmbda, dt):
    """
    Defines the problem for the example
    @param phi_test: Test function
    @param mu_test: Test function
    @param du: Trial Function
    @param u: Function, current solution
    @param phi: Function, current solution
    @param mu: Function, current solution
    @param phi_0: Function, previous solution
    @param mu_0: Function, previous solution
    @param velocity: Expression
    @param mid: float
    @param lmbda: float
    @param dt: float
    @return: Functions
    """
    mu_mid = mu_calc(mid, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * dot(velocity, grad(phi_0)) * phi_test * dx + dt * dot(
        grad(mu_mid), grad(phi_test)) * dx
    L1 = mu * mu_test * dx - (phi ** 3 - phi) * mu_test * dx - lmbda * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = dolfin.derivative(L, u, du)
    return a, L, u
