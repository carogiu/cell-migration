import dolfin
from model.model_domains import dom_and_bound
from ufl import grad
import numpy as np

h = 0.05
K = 0.1

dim_x = 6
dim_y = 6

nx = int(dim_x / h)
ny = int(dim_y / h)

mesh = dolfin.RectangleMesh(dolfin.Point(-dim_x / 2, 0.0), dolfin.Point(dim_x / 2, dim_y), nx, ny)

element_P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
space_ME = dolfin.FunctionSpace(mesh, element_P1)

phi = dolfin.Expression("tanh((sqrt(x[0]*x[0]+(x[1]-3)*(x[1]-3))-2)/(K*sqrt(2))) && tanh((-sqrt(x[0]*x[0]+(x[1]-3)*(x[1]-3))-2)/(K*sqrt(2)))", degree=1, K=K)
mu = phi**3-phi-K**2*grad(grad(phi))

#r = sqrt(x[0]*x[0]+(x[1]-3)*(x[1]-3))

pressure = dolfin.TrialFunctions(space_ME)
p_test = dolfin.TestFunctions(space_ME)


domain, boundaries = dom_and_bound(mesh, dim_x, dim_y)
dx = dolfin.dx(subdomain_data=domain)

bc_p_l = dolfin.DirichletBC(space_ME, dolfin.Constant(0.0), boundaries, 1)
bc_p_r = dolfin.DirichletBC(space_ME, dolfin.Constant(0.0), boundaries, 2)
bc_p_tb = dolfin.DirichletBC(space_ME, dolfin.Constant(0.0), boundaries, 3)

bcs = [bc_p_l, bc_p_r, bc_p_tb]

a = grad(pressure)*p_test*dx
L = (-3/(K*np.sqrt(3)))*phi*p_test*grad(mu)*dx

u = dolfin.Function(space_ME)


problem = dolfin.LinearVariationalProblem(a, L, u, bcs)
solver = dolfin.LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "mumps"  # LU did not work for big simulations because of memory capacity
# solver_flow.parameters["preconditioner"] = "ilu"
prm_flow = solver.parameters["krylov_solver"]
prm_flow["absolute_tolerance"] = 1E-7
prm_flow["relative_tolerance"] = 1E-4
prm_flow["maximum_iterations"] = 1000
dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize"] = True
solver.solve()

