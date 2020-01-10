### Packages
from dolfin import Expression, Constant
import numpy as np


### Class

class ModelParam:
    """A configuration for the model.

    Attributes:

    nx, ny : int, size of the mesh
    n : int, number of time steps
    theta : float, relative viscosity (usually 0.5, 1 or 2)
    epsilon : float, ratio between interface and length scale
    lmbda : float, parameter for the example model
    dt : float, size of a time step
    mob : float, energy factor (mobility of the fluid)
    vi : Expression, inflow velocity
    velocity : Expression, inflow velocity (if the inflow velocity is not a constant)
    """

    def __init__(self, nx, ny, n, theta, epsilon, lmbda, dt, mob, vi,
                 velocity=Expression(("10* sin(x[1]) + 5 * x[0] * x[0]", "0.0"), degree=2)):
        # geometry + rep
        self.nx = nx
        self.ny = ny
        self.n = n
        # dimensionless parameters DON'T CHANGE
        self.factor = Constant(3 / (2 * np.sqrt(2)))
        self.mid = Constant(0.5)
        # dimensionless parameters (can change)
        self.theta = Constant(theta)
        self.epsilon = Constant(epsilon) # !!! CHANGE EPSILON IN INITIAL CONDITION AS WELL
        self.lmbda = Constant(lmbda)
        self.dt = Constant(dt)
        self.mob = Constant(mob)
        self.vi = vi
        # test phase with a velocity
        self. velocity = Expression(velocity, degree=2)

