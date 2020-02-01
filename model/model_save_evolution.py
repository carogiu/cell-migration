### Packages
import numpy as np
import matplotlib.pyplot as plt
import dolfin


### Main save
def main_save_fig(u, u_flow, i, mesh, nx, ny, dim_x, dim_y, folder_name):
    """
    To save the phase, vx, vy and the pressure as figures in the appropriate folder, for the time i
    :param u: dolfin function
    :param u_flow: dolfin function
    :param i: time of the save
    :param mesh: mesh
    :param nx: grid dimension
    :param ny: grid dimension
    :param dim_x: dimension in x
    :param dim_y: dimension in y
    :param folder_name: string
    :return:
    """
    arr_phi, _ = array_exp_phase(u, mesh, nx, ny)
    arr_ux, arr_uy, arr_p = array_exp_flow(u_flow, mesh, nx, ny)
    save_fig(arr_phi, 'Phase', i, dim_x, dim_y, folder_name)
    save_fig(arr_ux, 'Vx', i, dim_x, dim_y, folder_name)
    save_fig(arr_uy, 'Vy', i, dim_x, dim_y, folder_name)
    save_fig(arr_p, 'Pressure', i, dim_x, dim_y, folder_name)
    return


def main_save_fig_interm(u, velocity, pressure, i, mesh, nx, ny, dim_x, dim_y, folder_name):
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
    :param folder_name: string
    :return:
    """
    arr_phi, _ = array_exp_phase(u, mesh, nx, ny)
    arr_p = pressure.compute_vertex_values(mesh).reshape(ny + 1, nx + 1)
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, ny + 1, nx + 1))
    arr_ux = arr_u[0]
    arr_uy = arr_u[1]
    save_fig(arr_phi, 'Phase', i, dim_x, dim_y, folder_name)
    save_fig(arr_ux, 'Vx', i, dim_x, dim_y, folder_name)
    save_fig(arr_uy, 'Vy', i, dim_x, dim_y, folder_name)
    save_fig(arr_p, 'Pressure', i, dim_x, dim_y, folder_name)
    return


# to save one array as a heat-map
def save_fig(arr, name, time, dim_x, dim_y, folder_name):
    """
    To save one array as a heat-map
    :param arr: array
    :param name: string
    :param time: int
    :param dim_x: dimension in x
    :param dim_y: dimension in y
    :param folder_name: string
    :return:
    """
    fig = plt.figure()
    if name == 'Phase':
        vmin, vmax = -1.1, 1.1
        map = 'seismic'
    if name == 'Vy':
        vmin, vmax = -2.5, +2.5
        map = 'jet'
    if name == 'Vx':
        vmin, vmax = -4, +6
        map = 'jet'
    if name == 'Pressure':
        vmin, vmax = None, None
        map = 'jet'

    plt.imshow(arr, cmap=map, extent=[-dim_x / 2, dim_x / 2, 0, dim_y], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(name + ' for t=' + str(time))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/" + name + '_' + str(time) + '.png')
    plt.close(fig)

    return


#### Dolfin function to explicit array
def array_exp_flow(U_flow, mesh, nx, ny):
    """
    From the function U_flow, extract ux, uy and p and return them as (nx x ny) arrays. Need mesh to know the dimensions
    :param mesh: dolfin mesh
    :param U_flow: dolfin Function
    :param nx: int, grid dimension
    :param ny: int, grid dimension
    :return: arrays, contain the values of vx, vy and p for a given time
    """
    velocity, pressure = U_flow.split()
    arr_p = pressure.compute_vertex_values(mesh)
    arr_p = np.reshape(arr_p, (ny + 1, nx + 1))
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, ny + 1, nx + 1))
    arr_ux = arr_u[0]
    arr_uy = arr_u[1]

    return arr_ux, arr_uy, arr_p


def array_exp_phase(u, mesh, nx, ny):
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


def interface(arr_phi):
    """
    Find the interface from the phase (interface : phi=0)
    @param arr_phi: array, values of phi for a given time
    @return: array (should be 1 x ny but needs improvements)
    """
    n = len(arr_phi)
    arr_interface = []
    for i in range(n):
        for j in range(n):
            if abs(arr_phi[i, j]) < .2:
                arr_interface.append([i, j])
    arr_interface = np.asarray(arr_interface)

    return arr_interface


### Partial saves (not used anymore)
def save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh, nx, ny):
    """
    For a given U_flow for time i, extract the corresponding ux, uy and p and saves them at the position i in the
    corresponding arrays
    @param vx_tot: array, contains of vx for a given time step i
    @param vy_tot: array, contains of vy for a given time step i
    @param p_tot: array, contains of p for a given time step i
    @param U_flow: dolfin Function
    @param i: int, number of the time step
    @param mesh: dolfin mesh
    @param nx: int, grid dimension
    @param ny: int, grid dimension
    @return: array
    """
    arr_ux, arr_uy, arr_p = array_exp_flow(U_flow, mesh, nx, ny)
    vx_tot[:, :, i] = arr_ux
    vy_tot[:, :, i] = arr_uy
    p_tot[:, :, i] = arr_p

    return vx_tot, vy_tot, p_tot


def save_phi(phi_tot, u, i, mesh, nx, ny):
    """
    For a given u for time i, extract the phase and saves it at position i in phi_tot
    @param phi_tot: array, contains all the values of phi for all the intermediate times
    @param u: dolfin Function
    @param i: int, number of the time step
    @param mesh: dolphin mesh
    @param nx: int, grid dimension
    @param ny: int, grid dimension
    @return: array
    """
    arr_phi, _ = array_exp_phase(u, mesh, nx, ny)
    phi_tot[:, :, i] = arr_phi

    return phi_tot


def main_save_old(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i, mesh, nx, ny):
    """
    For a time i, extracts ux, uy, p from U_flow and the phase from u and saves them at the position i in the
    corresponding arrays
    @param phi_tot: array, contains all the values of phi for all the intermediate times
    @param vx_tot: array, contains all the values of vx for all the intermediate times
    @param vy_tot: array, contains all the values of vy for all the intermediate times
    @param p_tot: array, contains all the values of p for all the intermediate times
    @param u: dolfin Function
    @param U_flow: dolfin Function
    @param i: int, number of the time step
    @param mesh: dolfin mesh
    @param nx: int, grid dimension
    @param ny: int, grid dimension
    @return: arrays
    """
    phi_tot = save_phi(phi_tot, u, i, mesh, nx, ny)
    vx_tot, vy_tot, p_tot = save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh, nx, ny)

    return phi_tot, vx_tot, vy_tot, p_tot


def interm_save_old(phi_tot, vx_tot, vy_tot, p_tot, u, velocity, pressure, i, mesh, nx, ny):
    """
    For a time i, extracts ux, uy, p from U_flow and the phase from u and saves them at the position i in the
    corresponding arrays
    @param phi_tot: array, contains all the values of phi for all the intermediate times
    @param vx_tot: array, contains all the values of vx for all the intermediate times
    @param vy_tot: array, contains all the values of vy for all the intermediate times
    @param p_tot: array, contains all the values of p for all the intermediate times
    @param u: dolfin Function
    @param pressure: dolfin Function
    @param velocity: dolfin Function
    @param i: int, number of the time step
    @param mesh: dolfin mesh
    @param nx: int, grid dimension
    @param ny: int, grid dimension
    @return: arrays
    """
    phi_tot = save_phi(phi_tot, u, i, mesh, nx, ny)
    arr_p = pressure.compute_vertex_values(mesh).reshape(nx + 1, ny + 1)
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, nx + 1, ny + 1))
    arr_ux = arr_u[0][::-1]
    arr_uy = arr_u[1][::-1]
    vx_tot[:, :, i] = arr_ux
    vy_tot[:, :, i] = arr_uy
    p_tot[:, :, i] = arr_p

    return phi_tot, vx_tot, vy_tot, p_tot
