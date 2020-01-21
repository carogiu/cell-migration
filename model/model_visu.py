### Packages
import matplotlib.pyplot as plt
import numpy as np


### Main functions

def main_visu(arr_tot, name, dim_x, dim_y):
    """
    Shows an array of name 'name' for all the times
    @param arr_tot: array, has all the intermediate arrays for the times i
    @param name: string
    @return: figures
    """
    n = arr_tot.shape[2]
    for i in range(n):
        if i % 1 == 0:  # can change here if want 1/2, 1/3 figures etc
            arr = arr_tot[:, :, i]
            time = str(int(i))
            visu(arr, name, time, dim_x, dim_y)

    return


### Base functions

def visu(arr, name, time, dim_x, dim_y):
    """
    Show the heatmap of an array, name and time appear in the title
    @param arr: array, values for a given time 'time'
    @param name: string, name of the value plotted
    @param time: string, time of the visualisation
    @return: figure
    """
    fig = plt.figure()
    plt.imshow(arr, cmap='jet', extent=[-dim_x / 2, dim_x / 2, 0, dim_y])
    plt.colorbar()
    plt.title(name + ' for t=' + time)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Figures/' + name + '_' + time + '.png')
    plt.show()
    plt.close(fig)

    return


### NOT USED ANYMORE

def visu_flow(arr_ux, arr_uy, arr_p, time, ):
    """
    To see ux, uy and p
    :param time: string, time of the visualisation
    :return: figure
    :param arr_ux: array, values of vx for the time 'time'
    :param arr_uy: array, values of vy for the time 'time'
    :param arr_p: array, values of p for the time 'time'
    """
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
    Shows the phase at time 'time' as a heatmap
    @param arr_phi: array, values of phi for the time 'time'
    @param time: string, time of the visualisation
    @return: figure
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
    Show the phase as a heatmap when time is an even number
    @param phi_tot: array, contains all the values of phi for all the intermediate times
    @return: figure
    """
    n = phi_tot.shape[2]
    for i in range(n):
        if i % 2 == 0:
            arr_phi = phi_tot[:, :, i]
            time = str(int(i))
            visu_phase(arr_phi, time)
    return


def interface(arr_phi):
    """
    Find the interface from the phase (interface : phi=0)
    TODO: To be improved, not very precise
    @param arr_phi: array, values of phi for a given time
    @return: array (should be 1 x ny but needs improvements)
    """
    n = len(arr_phi)
    interf = []
    for i in range(n):
        for j in range(n):
            if abs(arr_phi[i, j]) < .05:
                interf.append([i, j])
    interf = np.asarray(interf)

    return interf
