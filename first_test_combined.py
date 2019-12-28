from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Common mesh

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

def mesh_from_dim(nx, ny):
    """
    Creates mesh of dimension nx, ny of type quadrilateral

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :return: mesh
    """
    return UnitSquareMesh.create(nx, ny, CellType.Type.quadrilateral)

# Fluid set up

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

def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary): return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

def boundary_conditions_flow(W_flow, vi):
    """
    Creates the boundary conditions  : no slip condition, velocity inflow, pressure out

    :param W_flow: Function space
    :return: table of boundary conditions
    """
    noslip = Constant((0.0, 0.0))
    bc0 = DirichletBC(W_flow.sub(0), noslip, top_bottom)
    inflow = Expression((vi, "0.0"), degree=2)
    bc1 = DirichletBC(W_flow.sub(0), inflow, left)
    pressure_out = Constant((0.0))
    bc2 = DirichletBC(W_flow.sub(1), pressure_out, right)
    bcs = [bc0, bc1, bc2]
    return bcs

def theta_phi(theta,phi):
    theta_p = .5*((1.0-phi)+(1.0+phi)*theta)
    return theta_p

def problem_no_div(W_flow, vi, theta, phi, mu, factor):

    bcs = boundary_conditions_flow(W_flow, vi)
    f = Constant((0.0))
    theta_p = theta_phi(theta, phi)
    print(type(theta_p))
    (velocity, pressure) = TrialFunctions(W_flow)
    (v_test, p_test) = TestFunctions(W_flow)
    #a_flow = theta_p * dot(velocity, v_test) * dx - pressure * div(v_test) * dx + p_test * div(velocity) * dx - factor*div(v_test) * dx * phi * mu
    a_flow = theta_p * dot(velocity, v_test) * dx - pressure * div(v_test) * dx + p_test * div(velocity) * dx
    L_flow = f * p_test * dx
    U_flow = Function(W_flow)
    solve(a_flow == L_flow, U_flow, bcs=bcs, solver_parameters={"linear_solver": "lu"}, form_compiler_parameters={"optimize": True})
    return U_flow

# Phase set up

class InitialConditions(UserExpression):  # result is a dolfin Expression
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))  # need to seed random number
        super().__init__(**kwargs)

    def eval(self, values, x):
        # values[0] = 0.63 + 0.02*(0.5 - random.random())
        if abs(x[0] - .5) < epsilon/2:
            # random perturbation
            #values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2))) + np.random.randn(1) * 0.05 # phi(0)
            # sin perturbation
            values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2))) + np.sin(x[1]*30)*0.1

        else:
            values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2)))
        values[1] = 0.0  # mu(0)

    def value_shape(self):
        return (2,)  # dimension 2 (phi,mu)
    
class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a

    def F(self, b, x):
        assemble(self.L, tensor=b)  # computes the residual vector b

    def J(self, A, x):
        assemble(self.a, tensor=A)  # computes the Jacobian matrix A
        
def space_phase(mesh):
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1 * P1)
    return ME


def initiate_phase(ME):
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

def mu_calc(theta, mu, mu_0):
    return (1.0 - theta) * mu_0 + theta * mu

def potential(phi):
    phi = variable(phi)
    f = 100 * phi**2 * (1 - phi)**2
    dfdphi = diff(f, phi)
    return dfdphi

def potential_in_progress(phi):
    return phi ** 3 - phi

def problem_phase(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, theta, lmbda, dt):
    dfdphi = potential(phi)
    mu_mid = mu_calc(theta, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * dot(velocity, grad(phi_0)) * phi_test * dx + dt * dot(grad(mu_mid), grad(phi_test)) * dx
    L1 = mu * mu_test * dx - dfdphi * mu_test * dx - lmbda * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = derivative(L, u, du)
    return a, L, u

def problem_phase_in_progress(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, theta, epsilon, dt, M):
    #f = potential_in_progress(phi)
    f = potential(phi)
    mu_mid = mu_calc(theta, mu, mu_0)

    L0 = phi * phi_test * dx - phi_0 * phi_test * dx + dt * dot(velocity, grad(phi_0)) * phi_test * dx + epsilon * M * dt * dot(grad(mu_mid),
                                                                                                                                grad(phi_test)) * dx
    L1 = mu * mu_test * dx - f * mu_test * dx - epsilon ** 2 * dot(grad(phi), grad(mu_test)) * dx
    L = L0 + L1

    a = derivative(L, u, du)
    return a, L, u

# solver

def solve_phase(a, L, u):
    problem = CahnHilliardEquation(a, L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6
    solver.solve(problem, u.vector())
    return u

# arrays

def array_exp_phase(u):
    """
    :param u: Function of ME
    :return: array phi (nx x ny) and array mu (nx x ny) for easier plot
    """
    arr = u.compute_vertex_values(mesh)
    n = len(arr)
    mid = int(n / 2)
    arr_phi = arr[0:mid]
    arr_mu = arr[mid:n]
    arr_phi = np.reshape(arr_phi, (nx + 1, ny + 1))[::-1]
    arr_mu = np.reshape(arr_mu, (nx + 1, ny + 1))[::-1]
    return arr_phi, arr_mu

def array_exp_flow(U_flow):
    """
    From the function U, extract ux, uy and p and return them as nx x ny matrices
    :param U: Function
    :return: array
    """
    u, p = U_flow.split()
    arr_p = p.compute_vertex_values(mesh)
    arr_p = np.reshape(arr_p, (nx + 1, ny + 1))[::-1]
    arr_u = u.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, nx + 1, ny + 1))[::-1]
    arr_ux = arr_u[1]
    arr_uy = arr_u[0]
    return arr_ux, arr_uy, arr_p

def interface(arr_phi):
    n = len(arr_phi)
    interf = []
    for i in range(n):
        for j in range(n):
            if abs(arr_phi[i, j]) < .1:
                interf.append([i, j])
    interf = np.asarray(interf)
    return interf

def visu(U_flow, time):
    """
    To see ux, uy and p
    :param U_flow: Function
    :return: None
    """
    arr_ux, arr_uy, arr_p = array_exp_flow(U_flow)

    fig = plt.figure()
    plt.imshow(arr_ux, cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('vx')
    plt.colorbar()
    plt.title('vx for t='+time)
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(arr_uy, cmap='jet')
    plt.title('vy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('vy for t='+time)
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(arr_p, cmap='jet')
    plt.title('p')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('p for t='+time)
    plt.show()
    plt.close(fig)

def visu(arr_phi, time):
    fig = plt.figure()
    plt.imshow(arr_phi, cmap='jet')
    plt.colorbar()
    plt.plot(interface(arr_phi)[:, 1], interface(arr_phi)[:, 0], c='k')
    plt.title('phase for t='+time)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close(fig)

def see_all(phi_tot):
    n = phi_tot.shape[2]
    for i in range(n):
        if i % 2 == 0:
            arr_phi = phi_tot[:, :, i]
            time = str(int(i))
            visu(arr_phi, time)

def save(phi_tot, vx_tot, vy_tot, p_tot,u,U_flow,i):
    arr_phi, arr_mu = array_exp_phase(u)
    arr_ux, arr_uy, arr_p = array_exp_flow(U_flow)
    phi_tot[:, :, i] = arr_phi
    vx_tot[:, :, i] = arr_ux
    vy_tot[:, :, i] = arr_uy
    p_tot[:, :, i] = arr_p



# time evolution

def time_evolution_phase(ME, W_flow,vi, n, nx, ny, dt, theta, lmbda, factor, epsilon, M):
    #initiate the functions
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(ME)
    U_flow = problem_no_div(W_flow, vi, theta, phi, mu, factor)
    velocity, pressure = split(U_flow)
    #save the solutions
    phi_tot = np.zeros((nx + 1, ny + 1, n))
    vx_tot = np.zeros((nx + 1, ny + 1, n))
    vy_tot = np.zeros((nx + 1, ny + 1, n))
    p_tot = np.zeros((nx + 1, ny + 1, n))
    save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, 0)
    """
    arr_ux, arr_uy, arr_p = array_exp_flow(U_flow)
    arr_phi, arr_mu = array_exp_phase(u)
    phi_tot[:, :, 0] = arr_phi
    vx_tot[:,:,0] = arr_ux
    vy_tot[:,:,0] = arr_uy
    p_tot[:,:,0] = arr_p
    """

    for i in range(1, n):
        #a, L, u = problem_phase(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, theta, lmbda, dt)
        a, L, u = problem_phase_in_progress(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, theta, epsilon, dt, M)
        u0.vector()[:] = u.vector()
        u = solve_phase(a, L, u)
        U_flow = problem_no_div(W_flow, vi, theta, phi, mu, factor)
        velocity, pressure = split(U_flow)

        # save the solutions
        save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i)
        """
        arr_phi, arr_mu = array_exp_phase(u)
        phi_tot[:, :, i] = arr_phi
        arr_ux, arr_uy, arr_p = array_exp_flow(U_flow)
        vx_tot[:, :, i] = arr_ux
        vy_tot[:, :, i] = arr_uy
        p_tot[:, :, i] = arr_p
        """

    return phi_tot, vx_tot, vy_tot, p_tot


lmbda  = 1.0e-02
# physics parameters
# to see fingering, we need theta>1
theta = 2.0
epsilon = 0.1
factor = Constant((3/(2*epsilon*np.sqrt(2))))
M = 1
vi="1.0"

# computation parameters
nx = ny = 50
#dt = 5.0e-06
dt = 1.0e-07
#dt=1
n = 20


start_time = time.time()
# mesh
mesh = mesh_from_dim(nx, ny)
# Function space
ME = space_phase(mesh)
W_flow = space_flow(mesh)
# Boundary conditions
bcs = boundary_conditions_flow(W_flow,vi)


phi_tot, vx_tot, vy_tot, p_tot = time_evolution_phase(ME, W_flow,vi, n, nx, ny, dt, theta, lmbda, factor, epsilon, M)
see_all(phi_tot)

stop_time = time.time()

print('Took ' + str(int((stop_time - start_time) * 10) / 10) + 's to compute')





