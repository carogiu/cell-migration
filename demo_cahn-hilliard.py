import random
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

"""
make droplet and see if pressure is ok (Laplace)
"""


class InitialConditions(UserExpression):  # result is a dolfin Expression
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))  # need to seed random number
        super().__init__(**kwargs)

    def eval(self, values, x):
        # values[0] = 0.63 + 0.02*(0.5 - random.random())
        if abs(x[0] - .5) < epsilon:
            # random perturbation
            #values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2))) + np.random.randn(1) * 0.05 # phi(0)
            # sin perturbation
            values[0] = np.tanh(((x[0] - .5) * 2) / (epsilon * np.sqrt(2))) + np.sin(x[1]*10)*0.05

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


def mesh_from_dim(nx, ny):
    """
    Creates mesh of dimension nx, ny of type quadrilateral

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :return: mesh
    """
    return UnitSquareMesh.create(nx, ny, CellType.Type.quadrilateral)


def space_phase(mesh):
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1 * P1)
    return ME


def initiate_phase(ME):
    du = TrialFunction(ME)
    q, v = TestFunctions(ME)

    u = Function(ME)  # current solution u
    u0 = Function(ME)  # solution from previous converged step u0

    u_init = InitialConditions(degree=1)
    u.interpolate(u_init)
    u0.interpolate(u_init)

    phi, mu = split(u)
    phi_0, mu_0 = split(u0)

    return q, v, du, u, phi, mu, u0, phi_0, mu_0


def array_exp(u):
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


def interface(arr_phi):
    n = len(arr_phi)
    interf = []
    for i in range(n):
        for j in range(n):
            if abs(arr_phi[i, j]) < .01:
                interf.append([i, j])
    interf = np.asarray(interf)
    return interf


def potential(phi):
    phi = variable(phi)
    f = 100 * phi**2 * (1 - phi)**2
    dfdphi = diff(f, phi)
    return dfdphi

def potential_in_progress(phi):
    return phi ** 3 - phi


def mu_calc(theta, mu, mu_0):
    return (1.0 - theta) * mu_0 + theta * mu


def problem_phase(q, v, du, u, phi, mu, phi_0, mu_0, velocity, theta, lmbda, dt):
    dfdphi = potential(phi)
    mu_mid = mu_calc(theta, mu, mu_0)

    L0 = phi * q * dx - phi_0 * q * dx + dt * dot(velocity, grad(phi_0)) * q * dx + dt * dot(grad(mu_mid), grad(q)) * dx
    L1 = mu * v * dx - dfdphi * v * dx - lmbda * dot(grad(phi), grad(v)) * dx
    L = L0 + L1

    a = derivative(L, u, du)
    return a, L, u

def problem_phase_in_progress(q, v, du, u, phi, mu, phi_0, mu_0, velocity, theta, epsilon, dt, M):
    #f = potential_in_progress(phi)
    f = potential(phi)
    mu_mid = mu_calc(theta, mu, mu_0)

    L0 = phi * q * dx - phi_0 * q * dx + dt * dot(velocity, grad(phi_0)) * q * dx + epsilon * M * dt * dot(grad(mu_mid),
                                                                                                           grad(q)) * dx
    L1 = mu * v * dx - f * v * dx - epsilon ** 2 * dot(grad(phi), grad(v)) * dx
    L = L0 + L1

    a = derivative(L, u, du)
    return a, L, u

def solve_phase(a, L, u):
    problem = CahnHilliardEquation(a, L)
    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6
    solver.solve(problem, u.vector())
    return u


def time_evolution_phase(ME, n, nx, ny, dt, velocity, theta, lmbda, epsilon, M):
    q, v, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(ME)
    phi_tot = np.zeros((nx + 1, ny + 1, n))
    arr_phi, arr_mu = array_exp(u)
    phi_tot[:, :, 0] = arr_phi

    for i in range(1, n):
        #a, L, u = problem_phase(q, v, du, u, phi, mu, phi_0, mu_0, velocity, theta, lmbda, dt)
        a, L, u = problem_phase_in_progress(q, v, du, u, phi, mu, phi_0, mu_0, velocity, theta, epsilon, dt, M)
        u0.vector()[:] = u.vector()
        u = solve_phase(a, L, u)
        arr_phi, arr_mu = array_exp(u)
        phi_tot[:, :, i] = arr_phi

    return phi_tot


def visu(arr_phi, title):
    fig = plt.figure()
    plt.imshow(arr_phi, cmap='jet')
    plt.colorbar()
    # plt.plot(interface(arr_c)[:, 1], interface(arr_c)[:, 0], c='k')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close(fig)


def see_all(phi_tot):
    n = phi_tot.shape[2]
    for i in range(n):
        if i % 2 == 0:
            arr_phi = phi_tot[:, :, i]
            title = 't=' + str(int(dt * i * 10 ** 6)) + 'e-06s'
            visu(arr_phi, title)

lmbda  = 1.0e-02
# physics parameters
theta = 0.5
epsilon = 0.05
M = 1
velocity = Expression(("100", "0.0"), degree=2)

# computation parameters
nx = ny = 100
dt = 5.0e-06
n = 10

start_time = time.time()
mesh = mesh_from_dim(nx, ny)
ME = space_phase(mesh)
phi_tot = time_evolution_phase(ME, n, nx, ny, dt, velocity, theta, lmbda, epsilon, M)
stop_time = time.time()
print('Took ' + str(int((stop_time - start_time) * 10) / 10) + 's to compute')
see_all(phi_tot)

# Initial problem
"""
# Class representing the initial conditions

# Model parameters

lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06  # time step
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson


# Form compiler options #optimize the compilation (runs faster)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


# Create mesh and build function space
nx = ny = 200 #dimensions of the mesh
mesh = UnitSquareMesh.create(nx, ny, CellType.Type.quadrilateral)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #type (finite element) and dimension (1) of phi and mu
ME = FunctionSpace(mesh, P1 * P1) #function space, because u= phi * mu

# Define trial and test functions
du = TrialFunction(ME) #dimension 2 because ME has a dimension of 2
q, v = TestFunctions(ME) #don't forget the 's' for 'TestFunctions' if want several functions

# Define functions
u = Function(ME)  # current solution u = (c_{n+1}, \mu_{n+1})
u0 = Function(ME)  # solution from previous converged step u0 = (c_{n}, \mu_{n})

velocity = Expression(("10*(x[0]*x[0] - x[1]*x[1])", "0.0"), degree=2) #test of a velocity field

# Split mixed functions
dc, dmu = split(du)
c, mu = split(u)
c0, mu0 = split(u0)

# Creates initial conditions and interpolates
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

# since ``u`` and ``u0`` are finite element functions, they may not be
# able to represent a given function exactly, but the function can be
# approximated by interpolating it in a finite element space.

# Compute the chemical potential df/dc
c = variable(c) #declares that c is a variable that some function can be differentiated with respect to
f    = 100*c**2*(1-c)**2
dfdc = diff(f, c)


# It is convenient to introduce an expression for :math:`\mu_{n+\theta}`::

# mu_(n+theta)
mu_mid = (1.0-theta)*mu0 + theta*mu #Crank-Nicholson scheme

# Weak statement of the equations
#L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx #without convection
L0 = c*q*dx - c0*q*dx + dt*dot(velocity,grad(c0))*q*dx + dt*dot(grad(mu_mid), grad(q))*dx # with convection
L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
L = L0 + L1

# This is a statement of the time-discrete equations presented as part
# of the problem statement, using UFL syntax. The linear forms for the
# two equations can be summed into one form ``L``, and then the
# directional derivative of ``L`` can be computed to form the bilinear
# form which represents the Jacobian matrix::

# Compute directional derivative about u in the direction of du (Jacobian)
a = derivative(L, u, du)

# Create nonlinear problem and Newton solver
problem = CahnHilliardEquation(a, L)
solver = NewtonSolver() #type of solver #requires a :py:class:`NonlinearProblem <dolfin.cpp.NonlinearProblem>` object to solve a system of nonlinear equations
solver.parameters["linear_solver"] = "lu" #Lower-Upper decomposition (decomposition of the factors of a matrix as the product of a lower triangular matrix and an upper triangular matrix)
solver.parameters["convergence_criterion"] = "incremental" #type of convergence
solver.parameters["relative_tolerance"] = 1e-6



# Step in time
t = 0.0
T = 10*dt
while (t < T):
    t += dt
    u0.vector()[:] = u.vector() #the solution vector associated with ``u`` is copied to ``u0`` at the beginning of each time step
    solver.solve(problem, u.vector()) #the nonlinear problem is solved

"""
