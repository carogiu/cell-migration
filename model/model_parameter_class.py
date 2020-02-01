### Packages
from dolfin import Constant
import numpy as np
import time
import os


### Class

class ModelParam:
    """A configuration for the model.

    Attributes:

    h:              float, smallest element of the grid
    dim_x, dim_y :  int, dimensions of the box
    nx, ny :        int, size of the mesh

    n :             int, number of time steps
    dt :            float, size of a time step

    theta :         float, friction ration
    Cahn:           float, Cahn number
    Pe:             float, Peclet number
    Ca_star:        float, capillary number for classic model without the phase field (in our case, 1)
    Ca :            float, Capillary number (with the phase field correction)

    h_0:            float, amplitude of the perturbation
    wave:           float, wave number of the perturbation

    mid:            float, time scheme parameter (0.5)
    vi :            Expression, inflow velocity (1)
    """

    def __init__(self, h, dim_x, dim_y, n, dt, theta, Cahn, Pe, h_0, wave):
        # Grid parameters
        self.h = h
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nx = int(dim_x / h)
        self.ny = int(dim_y / h)
        # Time parameters
        self.n = n
        self.dt = Constant(dt)
        # Model parameters
        self.theta = Constant(theta)
        self.Cahn = Constant(Cahn)
        self.Pe = Constant(Pe)
        self.Ca_star = 10
        self.Ca = Constant(2 * np.sqrt(2) * Cahn * self.Ca_star / 3)  #0.001?

        # Initial perturbation parameters
        self.h_0 = h_0
        self.wave = wave

        # Fixed parameters, don't change
        self.mid = Constant(0.5)
        self.vi = "1"


def save_param(h, dim_x, dim_y, nx, ny, n, dt, theta, Cahn, Pe, Ca, h_0, wave):
    """
    Saves the parameters in a text files + returns the name of the folder for other saves
    :param h : smallest element of the grid
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param nx: number of mesh element in direction x
    :param ny: number of mesh element in direction y
    :param n: number of time steps
    :param dt: time step
    :param theta: viscosity ration
    :param Cahn: Cahn number
    :param Pe: Peclet number
    :param Ca : Capillary number
    :param h_0: amplitude of the perturbation
    :param wave: wave number of the perturbation
    :return: string, name of the folder where files should be saved
    """
    t = time.localtime()
    y, m, d = t.tm_year, t.tm_mon, t.tm_mday
    i = 1
    folder_name = str(d) + "-" + str(m) + "-" + str(y) + "#" + str(i)
    new_path = r'results/Figures/' + folder_name

    while os.path.exists(new_path):
        i = i + 1
        folder_name = str(d) + "-" + str(m) + "-" + str(y) + "#" + str(i)
        new_path = r'results/Figures/' + folder_name

    os.makedirs(new_path)
    file = open("results/Figures/" + folder_name + "/param.txt", "w")
    file.write("Model 2 (slip, Ca, Pe, K) \n" + "Parameters: "
               + "\n h= " + str(h) + "\n dim_x= " + str(dim_x) + "\n dim_y= " + str(dim_y)
               + "\n nx= " + str(nx) + "\n ny= " + str(ny)
               + "\n n= " + str(n) + "\n dt= " + str(dt.values()[0]) + "\n theta= " + str(theta.values()[0])
               + "\n Cahn= " + str(Cahn.values()[0]) + "\n Pe= " + str(Pe.values()[0]) + "\n Ca= " + str(Ca.values()[0])
               + "\n h_0= " + str(h_0) + "\n wave= " + str(wave))
    file.close()
    return folder_name
