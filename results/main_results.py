### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv
import dolfin
from pylab import *

### Imports
from model.model_flow import theta_phi


def save_interface_and_peaks(arr_interface: np.ndarray, folder_name: str, time_simu: int, dt: float, h_0: float,
                             starting_point: float) -> None:
    """

    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param folder_name: str, name of the folder were to save the values
    :param time_simu: int, time of the simulation
    :param dt: float, time step
    :param h_0: float, initial amplitude of the instability
    :param starting_point: float, where the simulation starts
    :return:
    """
    file_name = "results/Figures/" + folder_name + "/interface.csv"
    file_name_final = "results/Figures/" + folder_name + "/interface_final.csv"
    position_th = time_simu * dt + starting_point
    a, b = arr_interface.shape  # b=2

    # Save the interface coordinates
    len_int = int(a * b)
    interface_one_line = arr_interface.reshape(len_int)
    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(interface_one_line)

    # Save the peaks coordinates
    peaks_t, _ = find_peaks(arr_interface[:, 0])
    peaks_b, _ = find_peaks(-arr_interface[:, 0])
    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()][:, 0]
    amplitude = peaks - position_th
    amplitude = amplitude[abs(amplitude) >= h_0 / 4]
    with open(file_name_final, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(amplitude)
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


# To save one array as a heat-map
def save_fig(arr: np.ndarray, name: str, time: int, dim_x: int, dim_y: int, nx: int, ny: int, theta: float,
             folder_name: str) -> None or np.ndarray:
    """
    To save one array as a heat-map
    :param arr: array
    :param name: string
    :param time: int
    :param dim_x: dimension in x
    :param dim_y: dimension in y
    :param nx : grid dimension
    :param ny: grid dimension
    :param theta: viscosity ratio
    :param folder_name: string
    :return:
    """
    fig = plt.figure()
    if name == 'Phase':
        v_min, v_max = -1.1, 1.1
        color_map = 'seismic'
        arr_interface = interface(arr_phi=arr, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
        # plt.plot(arr_interface[:, 0], arr_interface[:, 1], ls=':', c='k')
        # peaks_t, _ = find_peaks(arr_interface[:, 0])
        # peaks_b, _ = find_peaks(-arr_interface[:, 0])
        # plt.plot(arr_interface[peaks_t, 0], arr_interface[peaks_t, 1], "x", c='g')
        # plt.plot(arr_interface[peaks_b, 0], arr_interface[peaks_b, 1], "x", c='g')
    elif name == 'Vy':
        v_min, v_max = -.5, +.5
        color_map = 'jet'
    elif name == 'Vx':
        v_min, v_max = .5, +1.5
        color_map = 'jet'
    else:  # name == 'Pressure'
        v_min, v_max = 0, dim_x / 2 * (theta + 1)
        color_map = 'jet'

    plt.imshow(arr, cmap=color_map, extent=[-dim_x / 2, dim_x / 2, 0, dim_y], vmin=v_min, vmax=v_max)
    plt.colorbar()
    plt.title(name + ' for t=' + str(time))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/" + name + '_' + str(time) + '.png')
    plt.close(fig)

    if name == 'Phase':
        return arr_interface
    else:
        return


# Extracts the interface from the phase
def interface(arr_phi: np.ndarray, nx: int, ny: int, dim_x: int, dim_y: int) -> np.ndarray:
    """
    Find the interface from the phase, where phi=0
    :param arr_phi: values of phi for a given time
    :param nx: size of the mesh in the direction of x
    :param ny: size of the mesh in the direction of y
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :return: array, 2 x ny , with the coordinates of the interface
    """
    arr_interface = np.zeros((ny + 1, 2))
    for j in range(ny + 1):
        i = np.argmin(abs(arr_phi[ny - j, :]))
        arr_interface[ny - j, :] = [i, j]
    arr_interface = np.asarray(arr_interface)
    arr_interface[:, 0] = (arr_interface[:, 0] - ((nx + 1) / 2)) * dim_x / (nx + 1)
    arr_interface[:, 1] = arr_interface[:, 1] * dim_y / (ny + 1)

    return arr_interface


#### Dolfin function to explicit array
def array_exp_flow(u_flow: dolfin.function.function.Function, mesh: dolfin.cpp.generation.RectangleMesh, nx: int,
                   ny: int) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    From the function u_flow, extract ux, uy and p and return them as (nx x ny) arrays. Need mesh to know the dimensions
    :param mesh: dolfin mesh
    :param u_flow: dolfin Function
    :param nx: int, grid dimension
    :param ny: int, grid dimension
    :return: arrays, contain the values of vx, vy and p for a given time
    """
    velocity, pressure = u_flow.split()
    arr_p = pressure.compute_vertex_values(mesh)
    arr_p = np.reshape(arr_p, (ny + 1, nx + 1))
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, ny + 1, nx + 1))
    arr_ux = arr_u[0]
    arr_uy = arr_u[1]

    return arr_ux, arr_uy, arr_p


def array_exp_velocity(velocity: dolfin.function.function.Function, mesh: dolfin.cpp.generation.RectangleMesh, nx: int,
                       ny: int) -> [np.ndarray, np.ndarray]:
    """
    Extract the values of the velocity and return them into two arrays
    :param velocity: dolfin function for the velocity
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :return:
    """
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, ny + 1, nx + 1))
    arr_ux = arr_u[0]
    arr_uy = arr_u[1]
    return arr_ux, arr_uy


def arr_exp_pressure(pressure, mesh: dolfin.cpp.generation.RectangleMesh, nx: int, ny: int) -> np.ndarray:
    """
    Extract the values of the pressure and return them into an array
    :param pressure: dolfin function for the pressure
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :return: array with the values of the pressure
    """
    arr_p = pressure.compute_vertex_values(mesh)
    arr_p = np.reshape(arr_p, (ny + 1, nx + 1))
    return arr_p


def array_exp_phase(u: dolfin.function.function.Function, mesh: dolfin.cpp.generation.RectangleMesh, nx: int,
                    ny: int) -> [np.ndarray, np.ndarray]:
    """
    From the function u, extracts the phase phi and the chemical potential mu and saves them as arrays
    :param u: Function of ME
    :param mesh: dolfin mesh
    :param nx: int, grid dimension
    :param ny: int, grid dimension
    :return: array phi (nx x ny) and array mu (nx x ny) for easier plot
    """
    arr = u.compute_vertex_values(mesh)
    n = len(arr)
    mid = int(n / 2)
    arr_phi = arr[0:mid]
    arr_mu = arr[mid:n]
    arr_phi = np.reshape(arr_phi, (ny + 1, nx + 1))
    arr_mu = np.reshape(arr_mu, (ny + 1, nx + 1))
    return arr_phi, arr_mu


# Not used anymore

"""
# Older version to save the peaks, not used anymore
def save_peaks(folder_name: str, arr_interface: np.ndarray, h_0: float) -> None:
    # Finds the peaks of the interface and then saves the distance between two peaks if it is bigger than the initial
    # instability
    # :param folder_name: str, name of the folder were to save the values
    # :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    # :param h_0: float, size of the initial instability
    # :return:

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


def all_peaks(arr_interface: np.ndarray) -> np.ndarray:
    # To find the peaks of the interface
    # :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    # :return: array, contains the coordinates of the peaks sorted by ordinates (y axis)

    peaks_t, _ = find_peaks(arr_interface[:, 0])
    peaks_b, _ = find_peaks(-arr_interface[:, 0])
    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()]
    return peaks
"""

# Older version to save the figures
"""
def main_save_fig(u: dolfin.function.function.Function, u_flow: dolfin.function.function.Function, i: int,
                  mesh: dolfin.cpp.generation.RectangleMesh, nx: int, ny: int, dim_x: int, dim_y: int, theta: float,
                  folder_name: str) -> np.ndarray:
    # To save the phase, vx, vy and the pressure as figures in the appropriate folder, for the time i
    # :param u: dolfin function for the phase
    # :param u_flow: dolfin function for the flow
    # :param i: time of the save
    # :param mesh: mesh
    # :param nx: grid dimension
    # :param ny: grid dimension
    # :param dim_x: dimension in x
    # :param dim_y: dimension in y
    # :param theta: viscosity ratio
    # :param folder_name: string
    # :return: array of the interface

    arr_phi, _ = array_exp_phase(u=u, mesh=mesh, nx=nx, ny=ny)
    arr_ux, arr_uy, arr_p = array_exp_flow(u_flow=u_flow, mesh=mesh, nx=nx, ny=ny)

    arr_interface = save_fig(arr=arr_phi, name='Phase', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                             folder_name=folder_name)
    # save_fig(arr=arr_ux, name='Vx', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
    #          folder_name=folder_name)
    # save_fig(arr=arr_uy, name='Vy', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
    #          folder_name=folder_name)
    # save_fig(arr=arr_p, name='Pressure', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
    #          folder_name=folder_name)
    return arr_interface


def main_save_fig_interm(u: dolfin.function.function.Function, velocity: dolfin.function.function.Function,
                         pressure: dolfin.function.function.Function, i: int, mesh: dolfin.cpp.generation.RectangleMesh,
                         nx: int, ny: int, dim_x: int, dim_y: int, theta: float, folder_name: str) -> np.ndarray:
    # To save the phase, vx, vy and the pressure as figures in the appropriate folder, for the time i
    # :param u: dolfin function
    # :param velocity: dolfin function
    # :param pressure: dolfin function
    # :param i: time of the save
    # :param mesh: mesh
    # :param nx: grid dimension
    # :param ny: grid dimension
    # :param dim_x: dimension in x
    # :param dim_y: dimension in y
    # :param theta: viscosity ratio
    # :param folder_name: string
    # :return: array of the interface

    arr_phi, _ = array_exp_phase(u=u, mesh=mesh, nx=nx, ny=ny)
    arr_p = arr_exp_pressure(pressure=pressure, mesh=mesh, nx=nx, ny=ny)
    arr_ux, arr_uy = array_exp_velocity(velocity=velocity, mesh=mesh, nx=nx, ny=ny)

    arr_interface = save_fig(arr=arr_phi, name='Phase', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                             folder_name=folder_name)
    save_fig(arr=arr_ux, name='Vx', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
             folder_name=folder_name)
    save_fig(arr=arr_uy, name='Vy', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
             folder_name=folder_name)
    save_fig(arr=arr_p, name='Pressure', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
             folder_name=folder_name)
    return arr_interface

"""
