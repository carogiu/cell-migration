### Packages
from dolfin import Expression, Constant
import numpy as np


### Class

class ModelParam:
    """A configuration for the model.

    Attributes:

    nx, ny : int, size of the mesh
    dim_x, dim_y : int, dimensions of the box
    n : int, number of time steps
    theta : float, relative viscosity (usually 0.5, 1 or 2)
    epsilon : float, ratio between interface and length scale
    dt : float, size of a time step
    mob : float, mobility ratio
    vi : Expression, inflow velocity
    """

    def __init__(self, nx, ny, n, dim_x, dim_y, theta, epsilon, dt, mob, vi):
        # geometry + rep
        self.nx = nx
        self.ny = ny
        self.n = n
        self.dim_x = dim_x
        self.dim_y = dim_y

        # dimensionless parameters DON'T CHANGE
        self.factor = Constant(3 / (2 * np.sqrt(2)))
        self.mid = Constant(0.5)

        # dimensionless parameters (can change)
        self.theta = Constant(theta)
        self.epsilon = Constant(epsilon)
        self.dt = Constant(dt)
        self.mob = Constant(mob)
        self.vi = vi
