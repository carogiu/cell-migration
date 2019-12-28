### Packages
import matplotlib.pyplot as plt
import numpy as np


### Main functions

def main_visu(arr_tot, name):
    """
    TODO : comment
    """
    n = arr_tot.shape[2]
    for i in range(n):
        if i % 2 == 0:
            arr = arr_tot[:, :, i]
            time = str(int(i))
            visu(arr, name, time)

    return


### Base functions

def visu(arr, name, time):
    """
    TODO : comment
    """
    fig = plt.figure()
    plt.imshow(arr, cmap='jet')
    plt.colorbar()
    plt.title(name + ' for t=' + time)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close(fig)

    return


def visu_flow(U_flow, mesh, time):
    """
    To see ux, uy and p
    :param time: time of the visualisation (string)
    :param mesh: mesh
    :param U_flow: Function
    :return: None
    """
    arr_ux, arr_uy, arr_p = array_exp_flow(U_flow, mesh)
    nx = ny = int(np.sqrt(mesh.num_cells()))
    fig = plt.figure()
    plt.imshow(arr_ux, cmap='jet', extent=[0, 1, 0, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('vx')
    plt.colorbar()
    plt.title('vx for t=' + time)
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(arr_uy, cmap='jet', extent=[0, 1, 0, 1])
    plt.title('vy')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('vy for t=' + time)
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(arr_p, cmap='jet', extent=[0, 1, 0, 1])
    plt.title('p')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('p for t=' + time)
    plt.show()
    plt.close(fig)

    return


def visu_phase(arr_phi, time):
    """
    TODO : comment or delete ?
    """
    fig = plt.figure()
    plt.imshow(arr_phi, cmap='jet')
    plt.colorbar()
    plt.plot(interface(arr_phi)[:, 1], interface(arr_phi)[:, 0], c='k')
    plt.title('phase for t=' + time)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close(fig)

    return


def see_all_phi(phi_tot):
    """
    TODO : comment or delete ?
    """
    n = phi_tot.shape[2]
    for i in range(n):
        if i % 2 == 0:
            arr_phi = phi_tot[:, :, i]
            time = str(int(i))
            visu_phase(arr_phi, time)


def interface(arr_phi):
    """
    TODO : comment or delete ?
    """
    n = len(arr_phi)
    interf = []
    for i in range(n):
        for j in range(n):
            if abs(arr_phi[i, j]) < .05:
                interf.append([i, j])
    interf = np.asarray(interf)

    return interf