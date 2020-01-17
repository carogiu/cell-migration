### Packages
# from dolfin import *
import dolfin
from ufl import dot, div, inner, grad, ds, dx


### Main functions
def space_flow(mesh):
    """
    Creates a function space for the flow
    First function is a vector (vx, vy)
    Second function is a scalar (p)
    :param mesh: mesh
    :return: Function space
    """
    element_P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    mixed_TH = element_P2 * element_P1
    w_flow = dolfin.FunctionSpace(mesh, mixed_TH)
    return w_flow


def problem_coupled(mesh, dim_x, dim_y, w_flow, phi, mu, vi, theta, factor, epsilon):
    """
    Solves the phase field problem, with the coupling, with inflow, no growth, no activity
    @param dim_x: dimension in the direction of x
    @param mesh: mesh
    @param w_flow: Function space
    @param phi: Function, phase
    @param mu: Function, chemical potential
    @param vi: Expression, inflow
    @param theta: float, friction ratio
    @param factor: float, numerical factor
    @param epsilon: float, length scale ratio
    @return: solution
    """
    bcs = boundary_conditions_flow(w_flow, vi, dim_x, dim_y)
    theta_p = theta_phi(theta, phi)
    normal = dolfin.FacetNormal(mesh)
    id_in = dolfin.Expression("x[0] < (- dim_x / 2 + tol) ? 1 : 0", degree=1,
                              dim_x=dim_x, tol=dolfin.DOLFIN_EPS)  # = 1 in the inflow on the left, 0 otherwise
    (velocity, pressure) = dolfin.TrialFunctions(w_flow)
    (v_test, p_test) = dolfin.TestFunctions(w_flow)
    a_flow = theta_p * dot(velocity, v_test) * dx - pressure * div(v_test) * dx - dot(grad(p_test), velocity) * dx + (
                id_in * p_test * dot(velocity, normal)) * ds
    L_flow = -factor * epsilon * phi * dot(v_test, grad(mu)) * dx
    u_flow = dolfin.Function(w_flow)

    # Solver
    problem_flow = dolfin.LinearVariationalProblem(a_flow, L_flow, u_flow, bcs)
    solver_flow = dolfin.LinearVariationalSolver(problem_flow)
    solver_flow.parameters["linear_solver"] = "lu"
    # solver_flow.parameters["preconditioner"] = "ilu"
    prm_flow = solver_flow.parameters["krylov_solver"]  # short form
    prm_flow["absolute_tolerance"] = 1E-7
    prm_flow["relative_tolerance"] = 1E-4
    prm_flow["maximum_iterations"] = 1000
    solver_flow.solve()

    # dolfin.solve(a_flow == L_flow, u_flow, bcs=bcs, solver_parameters={"linear_solver": "lu"},
    #             form_compiler_parameters={"optimize": True})
    return u_flow


### CREATE BOUNDARIES
def boundary_conditions_flow(w_flow, vi, dim_x, dim_y):
    """
    Creates the boundary conditions  : no slip condition, velocity inflow, pressure out
    :param w_flow: Function space
    :param vi: Expression, velocity inflow
    :return: array of boundary conditions
    """
    # no_slip = dolfin.Constant((0.0, 0.0))
    # bc_no_slip = dolfin.DirichletBC(w_flow.sub(0), no_slip, top_bottom)
    dom_left = BD_left(dim_x)
    dom_right = BD_right(dim_x)
    inflow = dolfin.Expression((vi, "0.0"), degree=2)
    bc_v_left = dolfin.DirichletBC(w_flow.sub(0), inflow, dom_left)
    bc_v_right = dolfin.DirichletBC(w_flow.sub(0), inflow, dom_right)
    pressure_out = dolfin.Constant(0.0)
    bc_p_right = dolfin.DirichletBC(w_flow.sub(1), pressure_out, dom_right)
    # bcs = [bc_no_slip, bc_v_left, bc_v_right, bc_p_right]
    bcs = [bc_v_left, bc_p_right, bc_v_right]

    return bcs


### UTILITARIAN FUNCTIONS

### BOUNDARIES : tells were the boundaries are (square)

class BD_right(dolfin.SubDomain):
    """
    :param dim_x: dimension in the direction of x
    """

    def __init__(self, dim_x, **kwargs):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        d_x = self.dim_x
        return x[0] > (d_x / 2 - dolfin.DOLFIN_EPS)


class BD_left(dolfin.SubDomain):
    """
    :param dim_x: dimension in the direction of x
    """

    def __init__(self, dim_x, **kwargs):
        self.dim_x = dim_x
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        d_x = self.dim_x
        return x[0] < - d_x / 2 + dolfin.DOLFIN_EPS


class BD_top_bottom(dolfin.SubDomain):
    """
    :param dim_y: dimension in the direction of y
    """

    def __init__(self, dim_y, **kwargs):
        self.dim_y = dim_y
        super().__init__(**kwargs)

    def inside(self, x, on_boundary):
        d_y = self.dim_y
        return x[1] > d_y - dolfin.DOLFIN_EPS or x[1] < dolfin.DOLFIN_EPS


### Transformation
def theta_phi(theta, phi):
    """
    Continuous dimensionless friction coefficient
    @param theta: float, friction ratio
    @param phi: Dolfin Function
    @return: Dolfin Function
    """
    theta_p = .5 * ((1 - phi) + (1 + phi) * theta)
    return theta_p


### TESTS Functions
def problem_stokes_flow(w_flow, vi):  # THIS IS WORKING
    """
    Creates the function U for the problem of Stokes flow (U = u, p)
    :param w_flow: Function space
    :param vi: Expression, inflow velocity
    :return: dolfin Function
    """
    bcs = boundary_conditions_flow(w_flow, vi)
    (velocity, pressure) = dolfin.TrialFunctions(w_flow)
    (v_test, p_test) = dolfin.TestFunctions(w_flow)
    f = dolfin.Constant((0.0, 0.0))
    a_flow = inner(grad(velocity), grad(v_test)) * dx + div(v_test) * pressure * dx + p_test * div(velocity) * dx
    L_flow = inner(f, v_test) * dx
    U_flow = dolfin.Function(w_flow)
    dolfin.solve(a_flow == L_flow, U_flow, bcs=bcs, solver_parameters={"linear_solver": "lu"},
                 form_compiler_parameters={"optimize": True})
    return U_flow


def flow_static_phase_no_mu(w_flow, vi, phi, theta):  # THIS IS WORKING
    """
    Partially solves the phase field with inflow vi, but no coupling term
    @param w_flow: Function space
    @param vi: Expression, inflow velocity
    @param phi: Function, phase
    @param theta: float, friction ratio
    @return:
    """
    theta_p = theta_phi(theta, phi)
    bcs = boundary_conditions_flow(w_flow, vi)
    (velocity, pressure) = dolfin.TrialFunctions(w_flow)
    (v_test, p_test) = dolfin.TestFunctions(w_flow)
    f = dolfin.Constant(0.0)
    a_flow = theta_p * dot(velocity, v_test) * dx - pressure * div(v_test) * dx + p_test * div(velocity) * dx
    L_flow = f * p_test * dx
    u_flow = dolfin.Function(w_flow)
    dolfin.solve(a_flow == L_flow, u_flow, bcs=bcs, solver_parameters={"linear_solver": "lu"},
                 form_compiler_parameters={"optimize": True})
    return u_flow


"""

def right(x):
    return x[0] > ( 1.0 - dolfin.DOLFIN_EPS)


def left(x):
    return x[0] < dolfin.DOLFIN_EPS


def top_bottom(x):
    return x[1] > 1.0 - dolfin.DOLFIN_EPS or x[1] < dolfin.DOLFIN_EPS 
"""
