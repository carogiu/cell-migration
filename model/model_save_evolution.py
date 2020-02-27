### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import dolfin
from pylab import *


### Main save
def main_save_fig(u: dolfin.function.function.Function, u_flow: dolfin.function.function.Function, i: int,
                  mesh: dolfin.cpp.generation.RectangleMesh, nx: int, ny: int, dim_x: int, dim_y: int, theta: float,
                  folder_name: str) -> np.ndarray:
    """
    To save the phase, vx, vy and the pressure as figures in the appropriate folder, for the time i
    :param u: dolfin function for the phase
    :param u_flow: dolfin function for the flow
    :param i: time of the save
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_x: dimension in x
    :param dim_y: dimension in y
    :param theta: viscosity ratio
    :param folder_name: string
    :return: array of the interface
    """
    arr_phi, _ = array_exp_phase(u=u, mesh=mesh, nx=nx, ny=ny)
    arr_ux, arr_uy, arr_p = array_exp_flow(u_flow=u_flow, mesh=mesh, nx=nx, ny=ny)
    velocity, pressure = u_flow.split()
    phi, _ = u.split()
    """
    # Test pressure
    fig = plt.figure()
    pr = dolfin.plot(object=pressure, cmap='jet')
    pr.set_clim(0.0, dim_x / 2 * (theta + 1))
    fig.colorbar(pr, boundaries=[0, dim_x / 2 * (theta + 1)], cmap='jet')
    plt.title('Pressure for t=' + str(i))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Pressure2_" + str(i) + '.png')
    plt.close(fig)

    
    # Test vx
    fig = plt.figure()
    plot_vx = dolfin.plot(object=velocity[0], cmap='jet')
    plot_vx.set_clim(0.0, 2)
    fig.colorbar(plot_vx, boundaries=[0, 2], cmap='jet')
    plt.title('Vx for t=' + str(i))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Vx2_" + str(i) + '.png')
    plt.close(fig)

    # Test vy
    fig = plt.figure()
    plot_vy = dolfin.plot(object=velocity[1], cmap='jet')
    plot_vy.set_clim(-1, 1)
    fig.colorbar(plot_vy, boundaries=[-1, 1], cmap='jet')
    plt.title('Vy for t=' + str(i))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Vy2_" + str(i) + '.png')
    plt.close(fig)

    # Test phi
    fig = plt.figure()
    plot_phi = dolfin.plot(object=phi, cmap='seismic')
    plot_phi.set_clim(-1.1, 1.1)
    fig.colorbar(plot_phi, boundaries=[-1.1, 1.1], cmap='seismic')
    plt.title('Phi for t=' + str(i))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Phi2_" + str(i) + '.png')
    plt.close(fig)
    """
    arr_interface = save_fig(arr=arr_phi, name='Phase', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                             folder_name=folder_name)
    # save_fig(arr=arr_ux, name='Vx', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta, folder_name=folder_name)
    # save_fig(arr=arr_uy, name='Vy', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta, folder_name=folder_name)
    # save_fig(arr=arr_p, name='Pressure', time=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta, folder_name=folder_name)
    return arr_interface


def main_save_fig_interm(u: dolfin.function.function.Function, velocity: dolfin.function.function.Function,
                         pressure: dolfin.function.function.Function, i: int, mesh: dolfin.cpp.generation.RectangleMesh,
                         nx: int, ny: int, dim_x: int, dim_y: int, theta: float, folder_name: str) -> np.ndarray:
    """
    To save the phase, vx, vy and the pressure as figures in the appropriate folder, for the time i
    :param u: dolfin function
    :param velocity: dolfin function
    :param pressure: dolfin function
    :param i: time of the save
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_x: dimension in x
    :param dim_y: dimension in y
    :param theta: viscosity ratio
    :param folder_name: string
    :return: array of the interface
    """
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


# to save one array as a heat-map
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
    if name == 'Vy':
        v_min, v_max = -.5, +.5
        color_map = 'jet'
    if name == 'Vx':
        v_min, v_max = .5, +1.5
        color_map = 'jet'
    if name == 'Pressure':
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

### Partial saves (not used anymore)
# def save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh, nx, ny):
#     """
#     For a given U_flow for time i, extract the corresponding ux, uy and p and saves them at the position i in the
#     corresponding arrays
#     @param vx_tot: array, contains of vx for a given time step i
#     @param vy_tot: array, contains of vy for a given time step i
#     @param p_tot: array, contains of p for a given time step i
#     @param U_flow: dolfin Function
#     @param i: int, number of the time step
#     @param mesh: dolfin mesh
#     @param nx: int, grid dimension
#     @param ny: int, grid dimension
#     @return: array
#     """
#     arr_ux, arr_uy, arr_p = array_exp_flow(U_flow, mesh, nx, ny)
#     vx_tot[:, :, i] = arr_ux
#     vy_tot[:, :, i] = arr_uy
#     p_tot[:, :, i] = arr_p
#
#     return vx_tot, vy_tot, p_tot
#
#
# def save_phi(phi_tot, u, i, mesh, nx, ny):
#     """
#     For a given u for time i, extract the phase and saves it at position i in phi_tot
#     @param phi_tot: array, contains all the values of phi for all the intermediate times
#     @param u: dolfin Function
#     @param i: int, number of the time step
#     @param mesh: dolphin mesh
#     @param nx: int, grid dimension
#     @param ny: int, grid dimension
#     @return: array
#     """
#     arr_phi, _ = array_exp_phase(u, mesh, nx, ny)
#     phi_tot[:, :, i] = arr_phi
#
#     return phi_tot
#
#
# def main_save_old(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i, mesh, nx, ny):
#     """
#     For a time i, extracts ux, uy, p from U_flow and the phase from u and saves them at the position i in the
#     corresponding arrays
#     @param phi_tot: array, contains all the values of phi for all the intermediate times
#     @param vx_tot: array, contains all the values of vx for all the intermediate times
#     @param vy_tot: array, contains all the values of vy for all the intermediate times
#     @param p_tot: array, contains all the values of p for all the intermediate times
#     @param u: dolfin Function
#     @param U_flow: dolfin Function
#     @param i: int, number of the time step
#     @param mesh: dolfin mesh
#     @param nx: int, grid dimension
#     @param ny: int, grid dimension
#     @return: arrays
#     """
#     phi_tot = save_phi(phi_tot, u, i, mesh, nx, ny)
#     vx_tot, vy_tot, p_tot = save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh, nx, ny)
#
#     return phi_tot, vx_tot, vy_tot, p_tot
#
#
# def interm_save_old(phi_tot, vx_tot, vy_tot, p_tot, u, velocity, pressure, i, mesh, nx, ny):
#     """
#     For a time i, extracts ux, uy, p from U_flow and the phase from u and saves them at the position i in the
#     corresponding arrays
#     @param phi_tot: array, contains all the values of phi for all the intermediate times
#     @param vx_tot: array, contains all the values of vx for all the intermediate times
#     @param vy_tot: array, contains all the values of vy for all the intermediate times
#     @param p_tot: array, contains all the values of p for all the intermediate times
#     @param u: dolfin Function
#     @param pressure: dolfin Function
#     @param velocity: dolfin Function
#     @param i: int, number of the time step
#     @param mesh: dolfin mesh
#     @param nx: int, grid dimension
#     @param ny: int, grid dimension
#     @return: arrays
#     """
#     phi_tot = save_phi(phi_tot, u, i, mesh, nx, ny)
#     arr_p = pressure.compute_vertex_values(mesh).reshape(nx + 1, ny + 1)
#     arr_u = velocity.compute_vertex_values(mesh)
#     arr_u = np.reshape(arr_u, (2, nx + 1, ny + 1))
#     arr_ux = arr_u[0][::-1]
#     arr_uy = arr_u[1][::-1]
#     vx_tot[:, :, i] = arr_ux
#     vy_tot[:, :, i] = arr_uy
#     p_tot[:, :, i] = arr_p
#
#     return phi_tot, vx_tot, vy_tot, p_tot
#
