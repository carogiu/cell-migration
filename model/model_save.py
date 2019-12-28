### Packages
import numpy as np

### Main save

def main_save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i, mesh):
    """
    TODO : comment ?
    """
    phi_tot = save_phi(phi_tot, u, i, mesh)
    vx_tot, vy_tot, p_tot = save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh)

    return phi_tot, vx_tot, vy_tot, p_tot


### Partial saves

#### Flow

def array_exp_flow(U_flow, mesh):
    """
    From the function U, extract ux, uy and p and return them as nx x ny matrices
    :param mesh: mesh
    :param U: Function
    :return: array
    """
    velocity, pressure = U_flow.split()
    nx = ny = int(np.sqrt(mesh.num_cells()))
    arr_p = pressure.compute_vertex_values(mesh)
    arr_p = np.reshape(arr_p, (nx + 1, ny + 1))[::-1]
    arr_u = velocity.compute_vertex_values(mesh)
    arr_u = np.reshape(arr_u, (2, nx + 1, ny + 1))
    arr_ux = arr_u[0][::-1]
    arr_uy = arr_u[1][::-1]

    return arr_ux, arr_uy, arr_p


def save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh):
    """
    TODO : comment?
    """
    arr_ux, arr_uy, arr_p = array_exp_flow(U_flow, mesh)
    vx_tot[:, :, i] = arr_ux
    vy_tot[:, :, i] = arr_uy
    p_tot[:, :, i] = arr_p

    return vx_tot, vy_tot, p_tot


#### Phase

def array_exp_phase(u, mesh):
    """
    :param u: Function of ME
    :return: array phi (nx x ny) and array mu (nx x ny) for easier plot
    """
    arr = u.compute_vertex_values(mesh)
    nx = ny = int(np.sqrt(mesh.num_cells()))
    n = len(arr)
    mid = int(n / 2)
    arr_phi = arr[0:mid]
    arr_mu = arr[mid:n]
    arr_phi = np.reshape(arr_phi, (nx + 1, ny + 1))[::-1]
    arr_mu = np.reshape(arr_mu, (nx + 1, ny + 1))[::-1]

    return arr_phi, arr_mu


def save_phi(phi_tot, u, i, mesh):
    """
    TODO : comment
    """
    arr_phi, _ = array_exp_phase(u, mesh)
    phi_tot[:, :, i] = arr_phi

    return phi_tot
