### Packages
# from dolfin import *
import dolfin
from ufl import dx, dot, div, inner, grad


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


def problem_coupled(w_flow, phi, mu, vi, theta, factor, epsilon):
    """
    Solves the phase field problem, with the coupling, with inflow, no growth, no activity
    @param w_flow: Function space
    @param phi: Function, phase
    @param mu: Function, chemical potential
    @param vi: Expression, inflow
    @param theta: float, friction ratio
    @param factor: float, numerical factor
    @param epsilon: float, length scale ratio
    @return:
    """
    bcs = boundary_conditions_flow(w_flow, vi)
    theta_p = theta_phi(theta, phi)
    (velocity, pressure) = dolfin.TrialFunctions(w_flow)
    (v_test, p_test) = dolfin.TestFunctions(w_flow)
    a_flow = theta_p * dot(velocity, v_test) * dx - pressure * div(v_test) * dx + p_test * div(
        velocity) * dx
    L_flow = (factor / epsilon) * div(v_test) * phi * mu * dx
    u_flow = dolfin.Function(w_flow)
    dolfin.solve(a_flow == L_flow, u_flow, bcs=bcs, solver_parameters={"linear_solver": "lu"},
                 form_compiler_parameters={"optimize": True})
    return u_flow


### CREATE BOUNDARIES
def boundary_conditions_flow(w_flow, vi):
    """
    Creates the boundary conditions  : no slip condition, velocity inflow, pressure out
    :param w_flow: Function space
    :param vi: Expression, velocity inflow
    :return: array of boundary conditions
    """
    no_slip = dolfin.Constant((0.0, 0.0))
    bc0 = dolfin.DirichletBC(w_flow.sub(0), no_slip, top_bottom)
    inflow = dolfin.Expression((vi, "0.0"), degree=2)
    bc1 = dolfin.DirichletBC(w_flow.sub(0), inflow, left)
    pressure_out = dolfin.Constant(0.0)
    bc2 = dolfin.DirichletBC(w_flow.sub(1), pressure_out, right)
    # with the no-slip condition
    # bcs = [bc0, bc1, bc2]
    # without the no-slip condition
    bcs = [bc1, bc2]
    return bcs


### UTILITARIAN FUNCTIONS

### BOUNDARIES : tells were the boundaries are (square)
def right(x, on_boundary):
    """
    Return TRUE if x is in the right boundary
    @param x: array
    @param on_boundary: dolfin parameter
    @return: boolean
    """
    return x[0] > (1.0 - dolfin.DOLFIN_EPS)


def left(x, on_boundary):
    """
    Return TRUE if x is in the left boundary
    @param x: array
    @param on_boundary: dolfin parameter
    @return: boolean
    """
    return x[0] < dolfin.DOLFIN_EPS


def top_bottom(x, on_boundary):
    """
    Retrun TRUE if y is on the top or the bottom boundary
    @param x: array
    @param on_boundary: dolfin parameter
    @return: boolean
    """
    return x[1] > 1.0 - dolfin.DOLFIN_EPS or x[1] < dolfin.DOLFIN_EPS


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
