### Packages
import dolfin
from ufl import dot, grad
import ufl

### Imports
from model.model_domains import dom_and_bound


### Main functions
def space_flow(mesh: dolfin.cpp.generation.RectangleMesh) -> dolfin.function.functionspace.FunctionSpace:
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


def problem_coupled(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: int, dim_y: int,
                    w_flow: dolfin.function.functionspace.FunctionSpace, phi: ufl.indexed.Indexed,
                    mu: ufl.indexed.Indexed, vi: str, theta: float, Ca: float) -> dolfin.function.function.Function:
    """
    Solves the phase field problem, with the coupling, with inflow, no growth, no activity
    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param w_flow: Function space
    :param phi: Function, phase
    :param mu: Function, chemical potential
    :param vi: Expression, inflow
    :param theta: friction ratio
    :param Ca: Capillary number
    :return: solution
    """
    # boundary conditions
    bcs_flow, domain, boundaries = boundary_conditions_flow(w_flow=w_flow, vi=vi, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # continuous viscosity
    theta_p = theta_phi(theta=theta, phi=phi)

    # Functions
    du_flow = dolfin.TrialFunction(V=w_flow)
    v_test, p_test = dolfin.TestFunctions(V=w_flow)

    u_flow = dolfin.Function(w_flow)
    velocity, pressure = dolfin.split(u_flow)

    # Problem
    a_flow = theta_p * dot(velocity, v_test) * dx + dot(v_test, grad(pressure)) * dx - dot(velocity, grad(p_test)) * dx
    L_flow = p_test * ds(1) - (1 / Ca) * phi * dot(v_test, grad(mu)) * dx

    F_flow = a_flow - L_flow
    J_flow = dolfin.derivative(form=F_flow, u=u_flow, du=du_flow)

    # Non linear solver
    problem_flow = dolfin.NonlinearVariationalProblem(F=F_flow, u=u_flow, bcs=bcs_flow, J=J_flow)
    solver_flow = dolfin.NonlinearVariationalSolver(problem_flow)

    # Solver parameters
    prm = solver_flow.parameters
    prm["nonlinear_solver"] = "newton"
    prm["newton_solver"]["absolute_tolerance"] = 1E-7
    prm["newton_solver"]["relative_tolerance"] = 1E-4
    prm["newton_solver"]["maximum_iterations"] = 1000
    prm["newton_solver"]["relaxation_parameter"] = 1.0
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    prm["newton_solver"]["linear_solver"] = "mumps"
    # prm["newton_solver"]["preconditioner"] = "ilu"
    prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1E-7
    prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1E-4
    prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 1000
    prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
    prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = True
    solver_flow.solve()

    # v_div, _ = dolfin.split(u_flow)
    # if abs(div(v_div)) >= 10e-8:  # TODO : Correct calculation of div
    #    raise ValueError('Divergence should not be higher than a certain threshold - Check Physics solutions')

    """
    # Linear solver
    problem_flow = dolfin.LinearVariationalProblem(a=a_flow, L=L_flow, u=u_flow, bcs=bcs_flow,
                                                   form_compiler_parameters={"optimize": True, "cpp_optimize": True})
    solver_flow = dolfin.LinearVariationalSolver(problem_flow)
    solver_flow.parameters[
        "linear_solver"] = "mumps"  # LU did not work for big simulations because of memory capacity ? mumps?
    solver_flow.parameters["preconditioner"] = "ilu"
    prm_flow = solver_flow.parameters["krylov_solver"]
    prm_flow["absolute_tolerance"] = 1E-7
    prm_flow["relative_tolerance"] = 1E-4
    prm_flow["maximum_iterations"] = 1000
    solver_flow.solve()
    """
    return u_flow


### CREATE BOUNDARIES
def boundary_conditions_flow(w_flow: dolfin.function.functionspace.FunctionSpace, vi: str, dim_x: int, dim_y: int,
                             mesh: dolfin.cpp.generation.RectangleMesh) -> [list, dolfin.cpp.mesh.MeshFunctionSizet,
                                                                            dolfin.cpp.mesh.MeshFunctionSizet]:
    """
    Creates the boundary conditions  : no slip condition, velocity inflow, pressure out
    :param w_flow: Function space
    :param vi: Expression, velocity inflow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :return: array of boundary conditions
    """
    domain, boundaries = dom_and_bound(mesh=mesh, dim_x=dim_x, dim_y=dim_y)
    # inflow and outflow of fluid
    inflow = dolfin.Expression((vi, "0.0"), degree=2)
    bc_v_left = dolfin.DirichletBC(w_flow.sub(0), inflow, boundaries, 1)
    # pressure out
    pressure_out = dolfin.Constant(0.0)
    bc_p_right = dolfin.DirichletBC(w_flow.sub(1), pressure_out, boundaries, 2)
    # boundary conditions
    bcs_flow = [bc_v_left, bc_p_right]

    return bcs_flow, domain, boundaries


### UTILITARIAN FUNCTIONS


### Transformation
def theta_phi(theta: float, phi: ufl.indexed.Indexed):
    """
    Continuous dimensionless friction coefficient
    @param theta: float, friction ratio
    @param phi: Dolfin Function
    @return: Dolfin Function
    """
    theta_p = .5 * ((1 - phi) + (1 + phi) * theta)
    return theta_p


### TESTS Functions
"""
#   Creates the function U for the problem of Stokes flow (U = u, p)
def problem_stokes_flow(w_flow, vi):  # THIS IS WORKING
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

#     Partially solves the phase field with inflow vi, but no coupling term
def flow_static_phase_no_mu(w_flow, vi, phi, theta):  # THIS IS WORKING
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
