### Packages
import numpy as np
import time
import os


### Class

class ModelParam:
    """A configuration for the model.

    Attributes:

    h:              float, smallest element of the grid
    dim_x, dim_y:   float, dimensions of the box
    nx, ny:         int, size of the mesh

    n:              int, number of time steps
    dt:             float, size of a time step

    theta:          float, friction ratio
    alpha:          float, activity
    vi:             int, inflow velocity
    Cahn:           float, Cahn number
    Pe:             float, Peclet number
    Ca_star:        float, capillary number for classic model without the phase field (in our case, 1)
    Ca:             float, Capillary number (with the phase field correction)

    starting_point: where the interface is at the beginning

    h_0:            float, amplitude of the perturbation
    k_wave:         float, wave number of the perturbation

    """

    def __init__(self, h: float, dim_x: float, dim_y: float, n: int, dt: float, theta: float, alpha: float, vi: int,
                 Cahn: float, Pe: int, starting_point: float, h_0: float, k_wave: float, folder_name: str) -> None:
        # Grid parameters
        self.h = h
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.nx = int(dim_x / h)
        self.ny = int(dim_y / h)

        # Time parameters
        self.n = n
        self.dt = dt

        # Model parameters
        self.theta = theta
        self.alpha = alpha
        self.vi = vi
        self.Cahn = Cahn
        self.Pe = Pe
        self.Ca_star = 1
        self.Ca = 2 * np.sqrt(2) * Cahn * self.Ca_star / 3
        self.starting_point = starting_point

        # Initial perturbation parameters
        self.h_0 = h_0
        self.k_wave = k_wave

        # Saving parameter
        self.folder_name = folder_name


def save_param(h: float, dim_x: float, dim_y: float, nx: int, ny: int, n: int, dt: float, theta: float, alpha: float,
               vi: int, Cahn: float, Pe: float, Ca: float, starting_point: float, h_0: float, k_wave: float) -> str:
    """
    Saves the parameters in a text file + returns the name of the folder for other saves
    :param h: smallest element of the grid
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param nx: number of mesh element in direction x
    :param ny: number of mesh element in direction y
    :param n: number of time steps
    :param dt: time step
    :param theta: viscosity ratio
    :param alpha: activity
    :param vi: inflow velocity
    :param Cahn: Cahn number
    :param Pe: Peclet number
    :param Ca: Capillary number
    :param starting_point: where the interface is at the beginning
    :param h_0: amplitude of the perturbation
    :param k_wave: wave number of the perturbation
    :return: string, name of the folder where files should be saved
    """

    if vi == 0:  # No inflow: no preferred wave length, sigma<0
        q = 0
        sigma = -(k_wave ** 3 - 1) / theta

    else:
        if alpha < 1:  # Low activity regime
            if theta >= 1 - alpha:  # Fingers can grow
                q = np.sqrt((theta - 1 + alpha) / 3)
                sigma = q * (theta - 1 + alpha) / (theta + np.sqrt(1 - alpha)) - q ** 3 / (theta + np.sqrt(1 - alpha))
            else:  # Fingers cannot grow: no preferred wave length, sigma<0
                q = 0
                sigma = (alpha - 1 + theta - k_wave ** 2) * k_wave / (theta + np.sqrt(1 - alpha))
        else:  # High activity regime, no solution
            q = 0
            sigma = 0

    # Create new folder to save the parameters
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
    # os.makedirs(new_path + '/Checks')
    os.makedirs(new_path + '/Analysis')

    # Save the parameters
    file = open("results/Figures/" + folder_name + "/param.txt", "w")
    file.write("Model 3 (slip, Ca, Pe, K, activity) \n" + "Parameters: "
               + "\n h= " + str(h) + "\n dim_x= " + str(dim_x) + "\n dim_y= " + str(dim_y)
               + "\n nx= " + str(nx) + "\n ny= " + str(ny)
               + "\n n= " + str(n) + "\n dt= " + str(dt)
               + "\n theta= " + str(theta)
               + "\n Cahn= " + str(Cahn) + "\n Pe= " + str(Pe) + "\n Ca= " + str(Ca)
               + "\n starting_point= " + str(starting_point)
               + "\n h_0= " + str(h_0) + "\n k_wave= " + str(k_wave)
               + "\n sigma= " + str(sigma) + "\n q= " + str(q)
               + "\n alpha= " + str(alpha) + "\n vi= " + str(vi))
    file.close()
    return folder_name
