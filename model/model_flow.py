### Packages
import dolfin
from ufl import dot, grad, inner
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


# DARCY
def flow_passive_darcy(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: float, dim_y: float,
                       w_flow: dolfin.function.functionspace.FunctionSpace, phi: ufl.indexed.Indexed,
                       mu: ufl.indexed.Indexed, vi: int, theta: float, Ca: float) -> dolfin.function.function.Function:
    """
    Solves Darcy flow (with the phase field surface tension correction) with inflow, no activity, no growth
    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param w_flow: Function space
    :param phi: Function, phase
    :param mu: Function, chemical potential
    :param vi: int, inflow
    :param theta: friction ratio
    :param Ca: Capillary number
    :return: solution
    """
    # Boundary conditions
    bcs_flow, domain, boundaries = boundary_conditions_flow(w_flow=w_flow, vi=vi, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Continuous viscosity
    theta_p = theta_phi(theta=theta, phi=phi)

    # Functions
    du_flow = dolfin.TrialFunction(V=w_flow)  # Used only to calculate the Jacobian

    v_test, p_test = dolfin.TestFunctions(V=w_flow)
    u_flow = dolfin.Function(w_flow)  # We use Function and not TrialFunction because the problem is non linear
    velocity, pressure = dolfin.split(u_flow)

    # Problem
    a_flow = theta_p * dot(velocity, v_test) * dx + dot(v_test, grad(pressure)) * dx - dot(velocity, grad(p_test)) * dx
    L_flow = dolfin.Constant(vi) * p_test * ds(1) - dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu)) * dx

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

    return u_flow


def flow_active_darcy(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: float, dim_y: float,
                      w_flow: dolfin.function.functionspace.FunctionSpace, phi: ufl.indexed.Indexed,
                      mu: ufl.indexed.Indexed, vi: int, theta: float, alpha: float,
                      Ca: float) -> dolfin.function.function.Function:
    """
    Solves Darcy flow (with the phase field surface tension correction) with inflow, with activity, no growth
    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param w_flow: Function space
    :param phi: Function, phase
    :param mu: Function, chemical potential
    :param vi: int, inflow
    :param theta: friction ratio
    :param alpha: activity
    :param Ca: Capillary number
    :return: solution
    """
    # Boundary conditions
    bcs_flow, domain, boundaries = boundary_conditions_flow(w_flow=w_flow, vi=vi, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Continuous values
    theta_p = theta_phi(theta=theta, phi=phi)
    alpha_p = dolfin.Constant(alpha) * i_phi(phi=phi)

    # Functions
    du_flow = dolfin.TrialFunction(V=w_flow)
    v_test, p_test = dolfin.TestFunctions(V=w_flow)

    u_flow = dolfin.Function(w_flow)
    velocity, pressure = dolfin.split(u_flow)

    # Norm of the velocity
    norm_sq = dot(velocity, velocity)
    unit_vector_velocity = velocity * ((norm_sq + dolfin.DOLFIN_EPS) ** (-0.5))

    # Problem
    F_flow = theta_p * dot(velocity, v_test) * dx - alpha_p * dot(unit_vector_velocity, v_test) * dx \
        + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu)) * dx + dot(v_test, grad(pressure)) * dx \
        - dot(velocity, grad(p_test)) * dx - dolfin.Constant(vi) * p_test * ds(1)

    J_flow = dolfin.derivative(form=F_flow, u=u_flow, du=du_flow)

    # Non linear solver
    problem_flow = dolfin.NonlinearVariationalProblem(F=F_flow, u=u_flow, bcs=bcs_flow, J=J_flow)
    solver_flow = SolverClass(problem=problem_flow)

    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["linear_algebra_backend"] = "PETSc"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    solver_flow.solve()

    return u_flow


def flow_passive_div_darcy(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: float, dim_y: float,
                           w_flow: dolfin.function.functionspace.FunctionSpace, phi: ufl.indexed.Indexed,
                           mu: ufl.indexed.Indexed, theta: float, Ca: float,
                           k: float) -> dolfin.function.function.Function:
    """
    Solves Darcy flow (with the phase field surface tension correction) with no inflow, no activity, with growth
    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param w_flow: Function space
    :param phi: Function, phase
    :param mu: Function, chemical potential
    :param theta: friction ratio
    :param Ca: Capillary number
    :param k: division parameter
    :return: solution
    """
    # Boundary conditions
    bcs_flow, domain, boundaries = boundary_conditions_flow_division(w_flow=w_flow, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)

    # Continuous viscosity
    theta_p = theta_phi(theta=theta, phi=phi)
    k_p = dolfin.Constant(k) * i_phi(phi=phi)

    # Functions
    du_flow = dolfin.TrialFunction(V=w_flow)  # Used only to calculate the Jacobian

    v_test, p_test = dolfin.TestFunctions(V=w_flow)
    u_flow = dolfin.Function(w_flow)  # We use Function and not TrialFunction because the problem is non linear
    velocity, pressure = dolfin.split(u_flow)

    # Problem
    F_flow = (theta_p * dot(velocity, v_test) + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu))
              + dot(v_test, grad(pressure)) - dot(velocity, grad(p_test)) - k_p * p_test) * dx

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
    prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1E-7
    prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1E-4
    prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 1000
    prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
    prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = True
    solver_flow.solve()

    return u_flow


# TONER-TU
def flow_passive_toner_tu(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: float, dim_y: float,
                          w_flow: dolfin.function.functionspace.FunctionSpace, phi: ufl.indexed.Indexed,
                          mu: ufl.indexed.Indexed, vi: int, theta: float,
                          Ca: float,
                          # velocity_previous: ufl.indexed.Indexed,
                          # velocity_pprevious
                          ) -> dolfin.function.function.Function:
    """
    Solves Toner-Tu flow (with the phase field surface tension correction) with inflow, no activity, no growth
    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param w_flow: Function space
    :param phi: Function, phase
    :param mu: Function, chemical potential
    :param vi: int, inflow
    :param theta: friction ratio
    :param Ca: Capillary number
    #:param velocity_previous: solution of the velocity for the time n-1
    #:param velocity_pprevious: solution of the velocity for the time n-2
    :return: solution
    """
    # Boundary conditions
    bcs_flow, domain, boundaries = boundary_conditions_flow(w_flow=w_flow, vi=vi, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Continuous viscosity
    theta_p = theta_phi(theta=theta, phi=phi)

    ### In case we try to use the norm from the previous time step, somewhat corrected with the material derivative
    # dt = 5e-3 ### Put this in the arguments of the function
    # velocity_previous = dolfin.project(velocity_previous, w_flow.sub(0).collapse())
    # velocity_pprevious = dolfin.project(velocity_pprevious, w_flow.sub(0).collapse())
    # mat_derivative = dot(velocity_previous, grad(velocity_previous))  + (velocity_previous - velocity_pprevious) / dt
    # v_corr = velocity_previous + mat_derivative * dt
    # norm_sq = dot(v_corr, v_corr)

    # Functions
    du_flow = dolfin.TrialFunction(V=w_flow)  # Used only to calculate the Jacobian

    v_test, p_test = dolfin.TestFunctions(V=w_flow)
    u_flow = dolfin.Function(w_flow)  # We use Function and not TrialFunction because the problem is non linear
    velocity, pressure = dolfin.split(u_flow)
    norm_sq = dot(velocity, velocity)

    # Problem
    F_flow = (theta_p * norm_sq * dot(velocity, v_test) + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu))
              + dot(v_test, grad(pressure)) - dot(velocity, grad(p_test))) * dx - dolfin.Constant(vi) * p_test * ds(1)

    J_flow = dolfin.derivative(form=F_flow, u=u_flow, du=du_flow)

    # Non linear solver
    problem_flow = dolfin.NonlinearVariationalProblem(F=F_flow, u=u_flow, bcs=bcs_flow, J=J_flow)
    solver_flow = SolverClass(problem=problem_flow)

    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    solver_flow.solve()

    return u_flow


def flow_active_toner_tu(mesh: dolfin.cpp.generation.RectangleMesh, dim_x: float, dim_y: float,
                         w_flow: dolfin.function.functionspace.FunctionSpace, phi: ufl.indexed.Indexed,
                         mu: ufl.indexed.Indexed, vi: int, theta: float, alpha: float,
                         Ca: float) -> dolfin.function.function.Function:
    """
    Solves Toner-Tu flow (with the phase field surface tension correction) with inflow, with activity, no growth
    :param mesh: mesh
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param w_flow: Function space
    :param phi: Function, phase
    :param mu: Function, chemical potential
    :param vi: int, inflow
    :param theta: friction ratio
    :param alpha: activity
    :param Ca: Capillary number
    :return: solution
    """
    # Boundary conditions
    bcs_flow, domain, boundaries = boundary_conditions_flow(w_flow=w_flow, vi=vi, dim_x=dim_x, dim_y=dim_y, mesh=mesh)
    dx = dolfin.dx(subdomain_data=domain)
    ds = dolfin.ds(subdomain_data=boundaries)

    # Continuous values
    theta_p = theta_phi(theta=theta, phi=phi)
    alpha_p = dolfin.Constant(alpha) * i_phi(phi=phi)

    # Functions
    du_flow = dolfin.TrialFunction(V=w_flow)
    v_test, p_test = dolfin.TestFunctions(V=w_flow)

    u_flow = dolfin.Function(w_flow)
    velocity, pressure = dolfin.split(u_flow)
    norm_sq = dot(velocity, velocity)

    # Problem
    F_flow = theta_p * norm_sq * dot(velocity, v_test) * dx - alpha_p * dot(velocity, v_test) * dx \
        + dolfin.Constant(1 / Ca) * phi * dot(v_test, grad(mu)) * dx + dot(v_test, grad(pressure)) * dx \
        - dot(velocity, grad(p_test)) * dx - dolfin.Constant(vi) * p_test * ds(1)

    J_flow = dolfin.derivative(form=F_flow, u=u_flow, du=du_flow)

    # Non linear solver
    problem_flow = dolfin.NonlinearVariationalProblem(F=F_flow, u=u_flow, bcs=bcs_flow, J=J_flow)
    solver_flow = SolverClass(problem=problem_flow)

    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["linear_algebra_backend"] = "PETSc"
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True

    solver_flow.solve()

    return u_flow


# SOLVER PARAMETERS

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
                                                    "linear_solver": "mumps",  # or lu
                                                    # "preconditioner": "jacobi",  # if lu, comment that line
                                                    "report": True,
                                                    "error_on_nonconvergence": True,
                                                    "krylov_solver": {"absolute_tolerance": 1E-7,
                                                                      "relative_tolerance": 1E-4,
                                                                      "maximum_iterations": 10000,
                                                                      "monitor_convergence": True,
                                                                      "nonzero_initial_guess": False,
                                                                      }
                                                    }
                                  }

    def setup_solver(self):
        dolfin.PETScOptions.set("ksp_type", "gmres")
        dolfin.PETScOptions.set("ksp_monitor")
        dolfin.PETScOptions.set("pc_type", "ilu")
        self.parameters.update(SolverClass.NewtonDict)


### CREATE BOUNDARIES
def boundary_conditions_flow(w_flow: dolfin.function.functionspace.FunctionSpace, vi: int, dim_x: float, dim_y: float,
                             mesh: dolfin.cpp.generation.RectangleMesh) -> [list, dolfin.cpp.mesh.MeshFunctionSizet,
                                                                            dolfin.cpp.mesh.MeshFunctionSizet]:
    """
    Creates the boundary conditions :velocity inflow, pressure out, no-slip condition if no inflow
    :param w_flow: Function space
    :param vi: Expression, velocity inflow
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :return: array of boundary conditions
    """
    domain, boundaries = dom_and_bound(mesh=mesh, dim_x=dim_x, dim_y=dim_y)

    # Boundary conditions for the fluid (inflow and outflow)
    inflow = dolfin.Expression(("vi", "0.0"), degree=2, vi=vi)
    bc_v_left = dolfin.DirichletBC(w_flow.sub(0), inflow, boundaries, 1)

    # Boundary condition for the pressure
    pressure_out = dolfin.Constant(0.0)
    bc_p_right = dolfin.DirichletBC(w_flow.sub(1), pressure_out, boundaries, 2)

    if vi == 0:
        # No slip condition
        v_null = dolfin.Expression(("0.0", "0.0"), degree=2)
        bc_v_top = dolfin.DirichletBC(w_flow.sub(0), v_null, boundaries, 3)
        bc_v_bot = dolfin.DirichletBC(w_flow.sub(0), v_null, boundaries, 4)

        # Boundary conditions
        bcs_flow = [bc_v_left, bc_p_right, bc_v_bot, bc_v_top]

    else:
        # Boundary conditions
        bcs_flow = [bc_v_left, bc_p_right]

    return bcs_flow, domain, boundaries


def boundary_conditions_flow_division(w_flow: dolfin.function.functionspace.FunctionSpace, dim_x: float, dim_y: float,
                                      mesh: dolfin.cpp.generation.RectangleMesh) -> [list,
                                                                                     dolfin.cpp.mesh.MeshFunctionSizet,
                                                                                     dolfin.cpp.mesh.MeshFunctionSizet]:
    """
    Creates the boundary conditions :velocity inflow, pressure out, no-slip condition if no inflow
    :param w_flow: Function space
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param mesh: mesh
    :return: array of boundary conditions
    """
    domain, boundaries = dom_and_bound(mesh=mesh, dim_x=dim_x, dim_y=dim_y)

    # Boundary conditions for the fluid (inflow and outflow)
    inflow = dolfin.Expression(("0.0", "0.0"), degree=2)
    bc_v_left = dolfin.DirichletBC(w_flow.sub(0), inflow, boundaries, 1)

    # Boundary condition for the pressure
    pressure_out = dolfin.Constant(0.0)
    bc_p_right = dolfin.DirichletBC(w_flow.sub(1), pressure_out, boundaries, 2)

    # Boundary conditions
    bcs_flow = [bc_v_left, bc_p_right]

    return bcs_flow, domain, boundaries


### UTILITARIAN FUNCTIONS

### Transformation
def theta_phi(theta: float, phi: ufl.indexed.Indexed):
    """
    Continuous dimensionless friction coefficient (1 in the active fluid, theta in the passive fluid)
    @param theta: float, friction ratio
    @param phi: Dolfin Function
    @return: Dolfin Function
    """
    theta_p = dolfin.Constant(.5) * ((dolfin.Constant(1) - phi) + (dolfin.Constant(1) + phi) * dolfin.Constant(theta))
    return theta_p


def i_phi(phi: ufl.indexed.Indexed):
    """
    Phase parameter, is 1 in active fluid and 0 in passive fluid
    @param phi: Dolfin Function
    @return: Dolfin Function
    """
    id_phi = dolfin.Constant(.5) * (dolfin.Constant(1) - phi)
    return id_phi
