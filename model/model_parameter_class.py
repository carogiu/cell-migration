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
        self.nx = nx
        self.ny = ny
        self.n = n
        self.factor = Constant(3 / (2 * np.sqrt(2)))
        self.mid = Constant(0.5)
        self.theta = Constant(theta)
        self.epsilon = Constant(epsilon)
        self.lmbda = Constant(lmbda)
        self.dt = Constant(dt)
        self.M = Constant(M)
        self.vi = vi
        self. velocity = Expression(velocity, degree=2)




def create_menu(config: MenuConfig):
    title = config.title
    body = config.body
    # ...


config = MenuConfig
config.title = "My delicious menu"
config.body = "A description of the various items on the menu"
config.button_text = "Order now!"
# The instance attribute overrides the default class attribute.
config.cancellable = True

create_menu(config)

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
velocity = Expression(("10* sin(x[1]) + 5 * x[0] * x[0]", "0.0"), degree=2)

# set as Dolfin Constants


