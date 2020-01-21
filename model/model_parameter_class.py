### Packages
from dolfin import Constant
import numpy as np
import time
import os


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


def save_param(nx, ny, n, dim_x, dim_y, theta, epsilon, dt, mob, vi):
    """
    Saves the parameters in a text files + returns the name of the folder for other saves
    :param nx: size of the mesh
    :param ny: size of the mesh
    :param n: number of time steps
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param theta: viscosity ration
    :param epsilon: length scale ratio
    :param dt: time step
    :param mob: mobility ration
    :param vi: initial velocity
    :return: string, name of the folder where files should be saved
    """
    t = time.localtime()
    y, m, d = t.tm_year, t.tm_mon, t.tm_mday
    i = 1
    folder_name = str(d) + "-" + str(m) + "-" + str(y) + "#" + str(i)
    newpath = r'Figures/' + folder_name

    while os.path.exists(newpath):
        i = i + 1
        folder_name = str(d) + "-" + str(m) + "-" + str(y) + "#" + str(i)
        newpath = r'Figures/' + folder_name

    os.makedirs(newpath)
    file = open("Figures/" + folder_name + "/param.txt", "w")
    file.write("Parameters: \n nx=" + str(nx) + "\n ny=" + str(ny) + "\n n=" + str(n) + "\n dim_x=" + str(
        dim_x) + "\n dim_y=" + str(dim_y) + "\n theta=" + str(theta.values()[0]) + "\n epsilon=" + str(
        epsilon.values()[0]) + "\n dt=" + str(
        dt.values()[0]) + "\n mob=" + str(mob.values()[0]) + "\n vi=" + vi)
    file.close()
    return folder_name
