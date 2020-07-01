### Packages
import random
import dolfin
import numpy as np
from ufl import dot, grad
import ufl

### Imports
from model.model_domains import dom_and_bound


### Initialise the phase
class InitialConditions(dolfin.UserExpression):  # result is a dolfin Expression
    """
    Creates the initial condition for the phase
    """

    def __init__(self, Cahn, h_0, k_wave, starting_point, **kwargs):
        random.seed(2)  # need to seed random number
        self.Cahn = Cahn
        self.h_0 = h_0
        self.k_wave = k_wave
        self.start = starting_point
        super().__init__(**kwargs)

    def eval(self, values, x):
        Cahn = float(self.Cahn)
        h_0 = float(self.h_0)
        k_wave = float(self.k_wave)
        start = float(self.start)

        # For a binary initial condition
        """
        if x[0] < start + h_0 * np.cos(x[1] * k_wave):
            values[0] = -1
        else:
            values[0] = 1
        values[1] = 0
        """
        # For a tanh initial condition
        dx = h_0 * np.cos(x[1] * k_wave)
        dx_prime = - h_0 * k_wave * np.sin(x[1] * k_wave)
        phi = np.tanh((x[0] - start + dx) / (Cahn * np.sqrt(2)))
        values[0] = phi  # phi(0)
        values[1] = (Cahn * dx / np.sqrt(2)) * (k_wave ** 2) * (1 - phi ** 2) + dx_prime ** 2 * phi * (
                1 - phi ** 2)  # mu(0)

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


def initiate_phase(space_ME: dolfin.function.functionspace.FunctionSpace, Cahn: float, h_0: float, k_wave: float,
                   starting_point: float):
    """
    Initiate the phase : from the function space, creates trial functions and test functions, and applies the
    initial conditions
    :param space_ME: Function space
    :param Cahn: Cahn number
    :param h_0: amplitude of the sin
    :param k_wave: wave number of the sin
    :param starting_point : float, where the interface is at the beginning
    :return: Functions
    """

    u0 = dolfin.Function(space_ME)  # current solution

    u_init = InitialConditions(degree=1, Cahn=Cahn, h_0=h_0, k_wave=k_wave, starting_point=starting_point)
    u0.interpolate(u_init)

    phi_0, mu_0 = dolfin.split(u0)

    return u0, phi_0


def problem_phase_implicit(space_ME: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                           mesh: dolfin.cpp.generation.RectangleMesh, phi_0: ufl.indexed.Indexed,
                           velocity: dolfin.function.function.Function, dt: float, Pe: float, Cahn: float):
    """
    Creates the variational problem for the phase and solves it
    :param space_ME: function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :param phi_0: Function, previous solution
    :param velocity: Expression, velocity of the flow for each point of the mesh
    :param dt: time step
    :param Pe: Peclet number
    :param Cahn: Cahn number
    :return: Function with the new solutions
    """
    # Define the domain
    bcs_phase, domain = boundary_conditions_phase(space_ME=space_ME, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)

    # Functions
    du_phase = dolfin.TrialFunction(V=space_ME)  # Used only to calculate the Jacobian

    phi_test, mu_test = dolfin.TestFunctions(V=space_ME)
    u_phase = dolfin.Function(space_ME)  # new solution
    phi, mu = dolfin.split(u_phase)

    # Variational problem (implicit time scheme)

    L0 = ((phi * phi_test - phi_0 * phi_test) * dolfin.Constant(1 / dt) + dolfin.Constant(1 / Pe) * (
        dot(grad(mu), grad(phi_test))) + phi_test * dot(velocity, grad(phi))) * dx

    L1 = (mu * mu_test - (phi ** 3 - phi) * mu_test - dolfin.Constant(Cahn ** 2) * dot(grad(phi), grad(mu_test))) * dx

    F_phase = L0 + L1

    J_phase = dolfin.derivative(form=F_phase, u=u_phase, du=du_phase)

    # Problem
    problem_phase = dolfin.NonlinearVariationalProblem(F=F_phase, u=u_phase, bcs=bcs_phase, J=J_phase)
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
    return u_phase


### BOUNDARIES
def boundary_conditions_phase(space_ME: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
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

    # Boundary conditions for the phase
    bc_phi_left = dolfin.DirichletBC(space_ME.sub(0), dolfin.Constant(-1.0), boundaries, 1)  # phi = -1 on the left (1)
    bc_phi_right = dolfin.DirichletBC(space_ME.sub(0), dolfin.Constant(1.0), boundaries, 2)  # phi = +1 on the right (2)

    # Boundary conditions for mu
    bc_mu_left = dolfin.DirichletBC(space_ME.sub(1), dolfin.Constant(0.0), boundaries, 1)  # mu = 0 on left and right
    bc_mu_right = dolfin.DirichletBC(space_ME.sub(1), dolfin.Constant(0.0), boundaries, 2)
    bcs_phase = [bc_phi_left, bc_phi_right, bc_mu_left, bc_mu_right]

    return bcs_phase, domain
