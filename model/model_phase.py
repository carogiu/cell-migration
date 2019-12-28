import random
from dolfin import *
import numpy as np


### Initialise the phase
class InitialConditions(UserExpression):  # result is a dolfin Expression
    """
    TODO : comment
    """

    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))  # need to seed random number
        super().__init__(**kwargs)

    def eval(self, values, x):
        # values[0] = 0.63 + 0.02*(0.5 - random.random())
        epsilon = .2
        if abs(x[0] - .5) < epsilon / 2:
            # random perturbation
            # values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2))) + np.random.randn(1) * 0.05 # phi(0)
            # sin perturbation
            values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2))) + np.sin(x[1] * 30) * 0.1

        else:
            values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2)))
        values[1] = 0.0  # mu(0)

    def value_shape(self):
        return (2,)  # dimension 2 (phi,mu)


### Initialise Problem resulution
class CahnHilliardEquation(NonlinearProblem):
    """
    TODO : comment
    """

    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        assemble(self.L, tensor=b)  # computes the residual vector b

    def J(self, A, x):
        assemble(self.a, tensor=A)  # computes the Jacobian matrix A


### Main functions
def space_phase(mesh):
    """
    TODO : comment
    """
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1 * P1)
    return ME


def initiate_phase(ME):
    """
    TODO : comment
    """
    du = TrialFunction(ME)
    phi_test, mu_test = TestFunctions(ME)

    u = Function(ME)  # current solution u
    u0 = Function(ME)  # solution from previous converged step u0

    u_init = InitialConditions(degree=1)
    u.interpolate(u_init)
    u0.interpolate(u_init)

    phi, mu = split(u)
    phi_0, mu_0 = split(u0)

    return phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0


def problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, M, epsilon):
    """
    TODO : comment
    """
    mu_mid = mu_calc(mid, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * phi_test * dot(velocity, grad(phi_0)) * dx + dt * (
            M * epsilon ** 2) * dot(
        grad(mu_mid), grad(phi_test)) * dx
    L1 = mu * mu_test * dx - (phi ** 3 - phi) * mu_test * dx - (epsilon ** 2) * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = derivative(L, u, du)

    return a, L, u


def solve_phase(a, L, u):
    """
    TODO : comment
    """
    problem = CahnHilliardEquation(a, L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6
    solver.solve(problem, u.vector())
    return u


### Utilitaries functions
def mu_calc(mid, mu, mu_0):
    """
    TODO : comment
    """
    return (1.0 - mid) * mu_0 + mid * mu


def potential(phi):
    """
    TODO :comment
    """
    phi = variable(phi)
    f = 100 * phi ** 2 * (1 - phi) ** 2
    dfdphi = diff(f, phi)

    return dfdphi


### TEST FUNCTIONS
def problem_phase(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, lmbda, dt):
    """
    TODO : comment
    """
    mu_mid = mu_calc(mid, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * dot(velocity, grad(phi_0)) * phi_test * dx + dt * dot(
        grad(mu_mid), grad(phi_test)) * dx
    L1 = mu * mu_test * dx - (phi ** 3 - phi) * mu_test * dx - lmbda * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = derivative(L, u, du)
    return a, L, u