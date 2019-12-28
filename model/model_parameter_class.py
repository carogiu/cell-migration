### Packages
from dolfin import Expression, Constant
import numpy as np


### Class

class ModelParam:
    """A configuration for the model.

    Attributes:
        TODO : precise attributes of parameters
        TODO : change names for clear ones (M ?)
    """

    def __init__(self, nx, ny, n, theta, epsilon, lmbda, dt, M, vi,
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
        self.M = Constant(M)
        self.vi = vi
        # test phase with a velocity
        self. velocity = Expression(velocity, degree=2)

