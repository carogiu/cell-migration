### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv
import dolfin
from pylab import *

### Imports
from model.model_flow import theta_phi


def all_peaks(arr_interface: np.ndarray) -> np.ndarray:
    """
    To find the peaks of the interface
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :return: array, contains the coordinates of the peaks sorted by ordinates (y axis)
    """
    peaks_t, _ = find_peaks(arr_interface[:, 0])
    peaks_b, _ = find_peaks(-arr_interface[:, 0])
    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()]
    return peaks


def save_peaks(folder_name: str, arr_interface: np.ndarray, h_0: float) -> None:
    """
    Finds the peaks of the interface and then saves the distance between two peaks if it is bigger than the initial
    instability
    :param folder_name: str, name of the folder were to save the values
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param h_0: float, size of the initial instability
    :return:
    """
    peaks = all_peaks(arr_interface=arr_interface)
    summits = peaks[:, 0]
    n = len(summits)
    instability = np.zeros(n)  # - 1)
    for i in range(n - 1):
        d = abs(summits[i] - summits[i + 1])
        if d >= h_0:
            instability[i] = d
    file_name = "results/Figures/" + folder_name + "/peaks.csv"
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(instability)
    return


def check_div_v(velocity: dolfin.function.function.Function, mesh: dolfin.cpp.generation.RectangleMesh, nx: int,
                ny: int, dim_x: int, dim_y: int, time: int, folder_name: str) -> None:
    """
    Save div(v)
    :param velocity: dolfin function for the velocity
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param time: time of the simulation
    :param folder_name: name of the folder where to save the files
    :return:
    """

    fig = plt.figure()
    p = dolfin.plot(dolfin.div(velocity))
    p.set_clim(-.5, .5)
    fig.colorbar(p, boundaries=[-.5, .5], cmap='jet')
    plt.title('Divergence for t=' + str(time))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Divergence_" + str(time) + '.png')
    plt.close(fig)

    """
    tot_div = np.zeros(1)

    for collection in p.collections:
        for path in collection.get_paths():
            #np.append(tot_div, np.asarray(path.to_polygons()))
            print(path.to_polygons()[0].shape)
    #print(tot_div)
    """
    return


def check_hydro(velocity: dolfin.function.function.Function, pressure: dolfin.function.function.Function,
                u: dolfin.function.function.Function, theta: float, Ca: float,
                mesh: dolfin.cpp.generation.RectangleMesh, nx: int, ny: int, dim_x: int, dim_y: int, time: int,
                folder_name: str) -> None:
    """
    Check the hydrodynamic relation far from the interface
    :param velocity: dolfin function for the velocity
    :param pressure: dolfin function for the pressure
    :param u: dolfin function for the phase and mu
    :param theta: viscosity ratio
    :param Ca: Capillary number
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param time: time of the simulation
    :param folder_name: name of the folder where to save the files
    :return:
    """

    phi, mu = u.split()
    theta_p = theta_phi(theta, phi)
    hydro = dolfin.grad(pressure) + (1 / Ca) * phi * dolfin.grad(mu) + theta_p * velocity

    fig = plt.figure()
    plot_hydro_x = dolfin.plot(hydro[0])
    plot_hydro_x.set_clim(-.1, .1)
    fig.colorbar(plot_hydro_x, boundaries=[-.1, .1], cmap='jet')
    plt.title('Hydro_x for t=' + str(time))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Hydro_x_" + str(time) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plot_hydro_y = dolfin.plot(hydro[1])
    plot_hydro_y.set_clim(-.01, .01)
    fig.colorbar(plot_hydro_y, boundaries=[-.01, .01], cmap='jet')
    plt.title('Hydro_y for t=' + str(time))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Hydro_y_" + str(time) + '.png')
    plt.close(fig)
    return
