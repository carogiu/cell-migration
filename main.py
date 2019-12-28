### Packages
import dolfin
from dolfin import Constant
import numpy as np

from model.model_main import main_model

### Constants
dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize"] = True

# geometry + rep
nx = ny = 50
n = 30

# dimensionless parameters DON'T CHANGE
factor = 3 / (2 * np.sqrt(2))
mid = .5

# dimensionless parameters (can change)
theta = 2
epsilon = .2  # !!! CHANGE EPSILON IN INITIAL CONDITION AS WELL
lmbda = 1.0e-02
dt = 5.0e-06
M=1
#vi = "10*sin(x[1]*2*pi)"
vi ="1"

# test phase with a velocity
velocity = dolfin.Expression(("10* sin(x[1]) + 5 * x[0] * x[0]", "0.0"), degree=2)

# set as Dolfin Constants
factor = Constant(factor)
mid = Constant(mid)
theta = Constant(theta)
epsilon = Constant(epsilon)
lmbda = Constant(lmbda)
dt = Constant(dt)
M = Constant(M)

main_model(nx, ny, vi, theta, factor, epsilon, mid, dt, M, n)
