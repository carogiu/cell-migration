### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv
import dolfin

# from model.model_flow import theta_phi
from model.model_save_evolution import array_exp_velocity, arr_exp_pressure


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
    Finds the peaks of the interface and then saves the distance between two peaks if it is bigger than the initial instability
    :param folder_name: str, name of the folder were to save the values
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param h_0: float, size of the initial instability
    :return:
    """
    peaks = all_peaks(arr_interface=arr_interface)
    summits = peaks[:, 0]
    n = len(summits)
    instability = np.zeros(n - 1)
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
    h_x = dim_x / nx
    h_y = dim_y / ny
    arr_ux, arr_uy = array_exp_velocity(velocity=velocity, mesh=mesh, nx=nx, ny=ny)
    arr_div = np.zeros((nx, ny))
    for i in range(nx - 1):
        for j in range(ny - 1):
            divergence = (arr_ux[i + 1, j] - arr_ux[i, j]) / h_x + (arr_uy[i, j + 1] - arr_uy[i, j]) / h_y
            arr_div[i, j] = divergence

    fig = plt.figure()
    plt.imshow(arr_div, cmap='jet', extent=[-dim_x / 2, dim_x / 2, 0, dim_y], vmin=-2, vmax=2)
    plt.colorbar()
    plt.title('Divergence for t=' + str(time))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Divergence_" + str(time) + '.png')
    plt.close(fig)
    return


def check_hydro(velocity: dolfin.function.function.Function, pressure: dolfin.function.function.Function,
                theta: dolfin.Constant, mesh: dolfin.cpp.generation.RectangleMesh, nx: int, ny: int, dim_x: int,
                dim_y: int, time: int, folder_name: str) -> None:
    """
    Check the hydrodynamic relation far from the interface
    :param velocity: dolfin function for the velocity
    :param pressure: dolfin function for the pressure
    :param theta: viscosity ration
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param time: time of the simulation
    :param folder_name: name of the folder where to save the files
    :return:
    """
    h_x = dim_x / nx

    # convert into arrays
    arr_p = arr_exp_pressure(pressure=pressure, mesh=mesh, nx=nx, ny=ny)
    arr_ux, _ = array_exp_velocity(velocity=velocity, mesh=mesh, nx=nx, ny=ny)
    arr_ux = arr_ux[:ny + 1, :nx]

    arr_hydro = np.zeros((ny + 1, ny))

    for i in range(nx - 1):
        grad_p = (arr_p[:, i + 1] - arr_p[:, i]) / h_x
        if i < nx / 2:
            arr_hydro[:, i] = grad_p + arr_ux[:, i]
        else:
            arr_hydro[:, i] = grad_p + theta * arr_ux[:, i]
    fig = plt.figure()
    plt.imshow(arr_hydro, cmap='jet', extent=[-dim_x / 2, dim_x / 2, 0, dim_y], vmin=-2, vmax=2)
    plt.colorbar()
    plt.title('Hydro for t=' + str(time))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Hydro_" + str(time) + '.png')
    plt.close(fig)
    return
