from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

#Uncomment before trying for thr first time
"""
# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()
"""

def mesh_from_dim(nx, ny):
    """
    Creates mesh of dimension nx, ny of type quadrilateral

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :return: mesh
    """
    return UnitSquareMesh.create(nx, ny, CellType.Type.quadrilateral)


def space_flow(mesh):
    """
    Creates a function space for the flow
    First function is a vector (vx, vy)
    Second function is a scalar (p)
    :param mesh: mesh
    :return: Function space
    """
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W_flow = FunctionSpace(mesh, TH)
    return W_flow

# tells were the boundaries are (square)
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary): return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS


def boundary_conditions_flow(W_flow):
    """
    Creates the boundary conditions  : no slip condition, velocity inflow, pressure out

    :param W_flow: Function space
    :return: table of boundary conditions
    """
    noslip = Constant((0.0, 0.0))
    bc0 = DirichletBC(W_flow.sub(0), noslip, top_bottom)
    inflow = Expression(("10", "0.0"), degree=2)
    bc1 = DirichletBC(W_flow.sub(0), inflow, left)
    pressure_out = Constant((0.0))
    bc2 = DirichletBC(W_flow.sub(1), pressure_out, right)
    bcs = [bc0, bc1, bc2]
    return bcs

def k_phi(phi):
    return .5*(1-phi)

def theta_phi(theta,phi):
    return .5*((1-phi)+(1+phi)*theta)

def problem_no_div(W_flow, beta):
    """
    Creates the function U for the problem of Darcy's flow (U = u, p)
    :param W_flow: Function space
    :param k: creation rate constant
    :return: Function
    """
    f = Constant((0.0))
    (u, p) = TrialFunctions(W_flow)
    (v, q) = TestFunctions(W_flow)
    a = beta * dot(u, v) * dx - p * div(v) * dx + q * div(u) * dx
    L = f * q * dx
    U = Function(W_flow)
    solve(a == L, U, bcs=bcs, solver_parameters={"linear_solver": "lu"}, form_compiler_parameters={"optimize": True})
    return U

def problem_stokes_flow(W_flow):
    """
    Creates the function U for the problem of Stokes flow (U = u, p)
    :param W_flow: Function space
    :return: Function
    """
    (u, p) = TrialFunctions(W_flow)
    (v, q) = TestFunctions(W_flow)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx
    U = Function(W_flow)
    solve(a == L, U, bcs=bcs, solver_parameters={"linear_solver": "lu"}, form_compiler_parameters={"optimize": True})
    return U

def array_exp(U):
    """
    From the function U, extract ux, uy and p and return them as nx x ny matrices
    :param U: Function
    :return: array
    """
    u, p = U.split()
    arr_p = p.compute_vertex_values(mesh)
    arr_p = np.reshape(arr_p, (nx + 1, ny + 1))[::-1]
    arr_u = u.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, nx + 1, ny + 1))[::-1]
    arr_ux = arr_u[1]
    arr_uy = arr_u[0]
    return arr_ux, arr_uy, arr_p

def visu(U):
    """
    To see ux, uy and p
    :param U: Function
    :return: None
    """
    arr_ux, arr_uy, arr_p = array_exp(U)

    plt.imshow(arr_ux, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('vx')
    plt.colorbar()
    plt.show()

    plt.imshow(arr_uy, cmap='jet')
    plt.title('vy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    plt.imshow(arr_p, cmap='jet')
    plt.title('p')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

#parameters
nx = ny = 100
beta = Constant((2.0))
#k = Constant((10.0))

#solving
mesh = mesh_from_dim(nx, ny)
W_flow = space_flow(mesh)
bcs = boundary_conditions_flow(W_flow)
U = problem_no_div(W_flow, beta)
#U = problem_stokes_flow(W_flow)

#seeing
visu(U)

# Initial problem

"""
# Load mesh
#mesh = UnitCubeMesh.create(16, 16, 16, CellType.Type.hexahedron)
nx=ny=100
#mesh = UnitSquareMesh(nx, ny)
mesh = UnitSquareMesh.create(nx, ny, CellType.Type.quadrilateral)

# Build function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# Next, we define the boundary conditions. ::

# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary): return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = Expression(("1", "0.0"), degree=2)
bc1 = DirichletBC(W.sub(0), inflow, left)

pressure_out = Constant((0.0))
bc2 = DirichletBC(W.sub(1), pressure_out, right)

# Collect boundary conditions
bcs = [bc0,bc1, bc2]


# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))
a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx


# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

# Create Krylov solver and AMG preconditioner
solver = KrylovSolver(krylov_method, "amg")

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)



# Solve
U = Function(W)
solver.solve(U.vector(), bb)
"""

