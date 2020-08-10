### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import ndimage
import csv
import dolfin
from pylab import *

### Imports
from model.model_flow import theta_phi


def save_interface_and_peaks(arr_interface: np.ndarray, folder_name: str, time_simu: int, dt: float, h_0: float,
                             starting_point: float, k_wave: float, h: float) -> None:
    """

    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param folder_name: str, name of the folder where to save the values
    :param time_simu: int, time of the simulation
    :param dt: float, time step
    :param h_0: float, initial amplitude of the instability
    :param starting_point: float, where the simulation starts
    :param k_wave: float, wave number of the perturbation
    :param h: smallest element of the grid
    :return:
    """
    file_name = "results/Figures/" + folder_name + "/interface.csv"
    file_name_final = "results/Figures/" + folder_name + "/interface_final.csv"
    position_th = time_simu * dt + starting_point
    a, b = arr_interface.shape  # b=2

    # Save the interface coordinates
    len_int = int(a * b)
    interface_one_line = arr_interface.reshape(len_int)
    interface_with_time = np.asarray([time_simu * dt])
    interface_with_time = np.concatenate((interface_with_time, interface_one_line))

    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(interface_with_time)

    # Save the peaks coordinates
    if k_wave == 0:
        k_wave = 0.0000001

    peaks_t, _ = find_peaks(arr_interface[:, 0], distance=(np.pi / (k_wave * h)), height=position_th + 0.8 * h_0)
    peaks_b, _ = find_peaks(-arr_interface[:, 0], distance=(np.pi / (k_wave * h)), height=0.8 * h_0 - position_th)

    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()][:, 0]
    amplitude = peaks - position_th
    amplitude = amplitude[abs(amplitude) >= h_0 * 0.8]
    amplitude_with_time = np.asarray([time_simu * dt])
    amplitude_with_time = np.concatenate((amplitude_with_time, amplitude))
    with open(file_name_final, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(amplitude_with_time)
    return


# To save one array as a heat-map
def save_fig(arr: np.ndarray, name: str, time_simu: int, dim_x: float, dim_y: float, theta: float,
             arr_interface: np.ndarray, folder_name: str, dt: float) -> None or np.ndarray:
    """
    To save one array as a color map
    :param arr: array
    :param name: string
    :param time_simu: int
    :param dim_x: dimension in x
    :param dim_y: dimension in y
    :param theta: viscosity ratio
    :param arr_interface: contains the coordinates of the interface (ny x 2)
    :param folder_name: string
    :param dt: time step
    :return:
    """
    color_map = 'seismic'
    v_min, v_max = -1, 1

    fig = plt.figure()
    if name == 'Phase':
        v_min, v_max = -1.1, 1.1
        color_map = 'seismic'
    elif name == 'Vy':
        v_min, v_max = -2, +2
        color_map = 'jet'
    elif name == 'Vx':
        v_min, v_max = -1, 3  # -2, 4
        color_map = 'jet'
    elif name == 'Pressure':
        v_min, v_max = 0, (theta + 1) * dim_x / 2 - (theta - 1) * (-1)  # (theta+1)*dim_x/2 - (theta-1)*start
        color_map = 'jet'
    elif name == 'Chemical_potential':
        v_min, v_max = -0.1, 0.1  # -0.5, 0.5
        color_map = 'jet'

    plt.imshow(arr, cmap=color_map, extent=[-dim_x / 2, dim_x / 2, -dim_y / 2, dim_y / 2], vmin=v_min, vmax=v_max)
    plt.colorbar()
    plt.plot(arr_interface[:, 0], arr_interface[:, 1], c='green', linewidth=0.5)
    plt.title(name + '\n t=' + str(int(time_simu * dt * 1000000) / 1000000))
    plt.xlabel('x')
    plt.ylabel('y')
    file_name = 'results/Figures/' + folder_name + "/" + name + '_' + str(time_simu) + '.png'
    plt.savefig(fname=file_name, dpi=300)
    plt.close(fig)

    return


def save_quiver(arr_ux: np.ndarray, arr_uy: np.ndarray, time_simu: int, dim_x: float, dim_y: float, nx: int, ny: int,
                folder_name: str, arr_interface: np.ndarray, dt: float):
    """
    Saves the velocity as a quiver plot.The colour of the arrow indicates the value of the norm of the velocity
    :param arr_ux: values of vx
    :param arr_uy: values of vx
    :param time_simu: time of the simulation
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :param nx: size of the mesh in the direction of x
    :param ny: size of the mesh in the direction of y
    :param folder_name: name of the folder where to save the values
    :param arr_interface: contains the coordinates of the interface (ny x 2)
    :param dt: time step

    :return:
    """
    X = np.linspace(-dim_x / 2, dim_x / 2, nx + 1)
    Y = np.linspace(-dim_y / 2, dim_y / 2, ny + 1)
    X, Y = np.meshgrid(X, Y)

    n_tot = X.shape[0] * X.shape[1]

    M = np.hypot(arr_ux, arr_uy)

    X = X.reshape(n_tot)
    Y = Y.reshape(n_tot)
    M = M.reshape(n_tot)
    Ux = arr_ux.reshape(n_tot)
    Uy = arr_uy.reshape(n_tot)

    norm_ = matplotlib.colors.Normalize(vmin=0.8, vmax=1.4)  # alpha <1 -> (0.25,1.75), alpha >=1 (-1,3)

    # choose a colormap
    cm_ = matplotlib.cm.jet

    # create a ScalarMappable and initialize a data structure
    sm = matplotlib.cm.ScalarMappable(cmap=cm_, norm=norm_)
    sm.set_array([])

    arrow_step = 15

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.plot(arr_interface[:, 0], arr_interface[:, 1], c='green', linewidth=0.5)
    ax.quiver(X[::arrow_step], Y[::arrow_step], Ux[::arrow_step], Uy[::arrow_step], color=cm_(norm_(M[::arrow_step])),
              pivot='tail', headwidth=3, scale=15)
    ax.set(xlabel='x', ylabel='y')
    ax.set_ylim([-2, 3])
    ax.set_xlim([-2, 2])
    plt.colorbar(sm)
    plt.title('Velocity field \n t=' + str(int(time_simu * dt * 1000000) / 1000000))
    file_name = 'results/Figures/' + folder_name + "/Velocity_field_" + str(time_simu) + '.png'
    plt.savefig(fname=file_name, dpi=500)
    plt.close(fig)

    return


def Delta_criterion(u_flow: dolfin.function.function.Function, mesh: dolfin.cpp.generation.RectangleMesh, nx: int,
                    ny: int, dim_x: float, dim_y: float, time_simu: int, arr_interface: np.ndarray,
                    folder_name: str, dt: float) -> None:
    """
    From the velocity field, compute the Delta criterion to find the vortex. Saves the size of the vortexes in a csv
    file. Saves the Delta criterion as a figure (can be black and white or with intermediate colors).
    J = Jacobian of the velocity
    Q = tr(J)^2 - 4det(J) = (vx_x + vy_y)**2 - 4(vx_x*vy_y - vx_y*vy_x)
    Need Q<0 to have vortex
    :param u_flow: dolfin Function with the flow
    :param mesh: dolfin Mesh
    :param nx: size of the mesh in the direction of x
    :param ny: size of the mesh in the direction of y
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :param time_simu: time of the simulation
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param folder_name: name of the folder where to save the values
    :param dt: time step
    :return:
    """
    # Delta criterion
    element_int = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    w_int = dolfin.FunctionSpace(mesh, element_int)
    velocity, _ = u_flow.split()

    Delta = (dolfin.Dx(velocity[0], 0) + dolfin.Dx(velocity[1], 1)) ** 2 - 4 * (
            dolfin.Dx(velocity[0], 0) * dolfin.Dx(velocity[1], 1)
            - dolfin.Dx(velocity[0], 1) * dolfin.Dx(velocity[1], 0))
    Delta = dolfin.project(Delta, w_int)

    arr_Delta = arr_exp_pressure(pressure=Delta, mesh=mesh, nx=nx, ny=ny)
    a, b = arr_Delta.shape

    # Binary map
    G = np.zeros((a, b))
    G[arr_Delta >= 0] = 0
    G[arr_Delta < 0] = 1

    label_im, nb_labels = ndimage.label(G, structure=np.ones((3, 3)))

    """
    # Remove straight lines
    for i in range(1, nb_labels):
        mask = (label_im == i)
        mask_bis = np.sum(mask, axis=1)
        mask_ter = np.sum(mask, axis=0)
        if np.max(mask_bis) < 2:
            label_im[mask] = 0
        if np.max(mask_ter) < 2:
            label_im[mask] = 0

    # Recount the labels
    label_im[label_im > 0] = 1
    label_im, nb_labels = ndimage.label(label_im, structure=np.ones((3, 3)))
    
    # Remove 1D lines between domains
    for i in range(1, nb_labels):
        mask = (label_im == i)
        mask_changed = (label_im == i)
        mask_bis = np.sum(mask, axis=1)
        mask_changed[mask_bis == 1, :] = False
        mask_diff = mask != mask_changed
        label_im[mask_diff] = 0

    # Recount the labels
    label_im[label_im > 0] = 1
    label_im, nb_labels = ndimage.label(label_im)
    
    # Try to remove straight patterns

    for i in range(1, nb_labels):
        mask = (label_im == i)
        a, b = mask.shape
        mask_changed = (label_im == i)
        for j in range(1, a - 1):
            for k in range(1, b - 1):
                if mask_changed[j, k]:
                    if (not mask_changed[j - 1, k]) and (not mask_changed[j + 1, k]):
                        mask_changed[j, k] = False
                    if (not mask_changed[j, k - 1]) and (not mask_changed[j, k + 1]):
                        mask_changed[j, k] = False
        mask_diff = mask != mask_changed
        label_im[mask_diff] = 0

    # Recount the labels
    label_im[label_im > 0] = 1
    label_im, nb_labels = ndimage.label(label_im)

    # Remove too small areas
    sizes = ndimage.sum(G, label_im, range(nb_labels + 1))
    mask_size = sizes < 5
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    # Recount the labels
    label_im[label_im > 0] = 1
    label_im, nb_labels = ndimage.label(label_im, structure=np.ones((3, 3)))
    """

    # Save the size of the domains
    sizes = ndimage.sum(G, label_im, range(nb_labels + 1))
    sizes = sizes[sizes != np.max(sizes)]
    size_with_time = np.asarray([time_simu])
    size_with_time = np.concatenate((size_with_time, sizes))

    file_name = "results/Figures/" + folder_name + "/size_vortex.csv"
    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(size_with_time)

    # All vortex with same label
    # label_im[label_im > 0] = 1

    # Plot

    spec = cm.get_cmap('nipy_spectral', 256)
    new_colors = spec(np.linspace(0, 1, 256))
    new_colors[0, :] = [0, 0, 0, 0]
    # newcmp = matplotlib.colors.ListedColormap(new_colors)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.imshow(arr_Delta, cmap='gnuplot2', extent=[-dim_x / 2, dim_x / 2, dim_y / 2, dim_y / 2], vmin=-10, vmax=0)
    plt.colorbar()
    # plt.imshow(G, cmap='binary', extent=[-dim_x / 2, dim_x / 2, -dim_y/2, dim_y/2])
    # plt.imshow(label_im, cmap=newcmp, extent=[-dim_x / 2, dim_x / 2, -dim_y/2, dim_y/2])
    # plt.contour(label_im, [0.1], linewidths=0.1, colors='r', extent=[-dim_x / 2, dim_x / 2, dim_y/2, -dim_y/2])
    plt.plot(arr_interface[:, 0], arr_interface[:, 1], c='green', linewidth=0.5)
    ax.set(xlabel='x', ylabel='y', xlim=(-dim_x / 2, dim_x / 2), ylim=(-dim_y / 2, dim_y / 2))
    plt.title('Delta criterion for t=' + str(int(time_simu * dt * 1000000) / 1000000))
    file_name = 'results/Figures/' + folder_name + '/Delta_vortex_' + str(time_simu) + '.png'
    plt.savefig(fname=file_name, dpi=500)
    plt.close(fig)

    return


def velocity_orientation(arr_ux: np.ndarray, arr_uy: np.ndarray, dim_x: float, dim_y: float,
                         time_simu: int, arr_interface: np.ndarray, folder_name: str, dt: float):
    """
    From the velocity, compute the angle and save the mean and the std of the angles in the active fluid. Draws the
    contour of areas in the active part where the fluid is going backwards (vx<0) (saves it in a csv file). Saves a
    figure with the angles, the contour lines and a color wheel for the angles.
    :param arr_ux: values of vx
    :param arr_uy: values of vx
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :param time_simu: time of the simulation
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param folder_name: name of the folder where to save the values
    :param dt: time step
    :return:
    """
    orientation = np.arccos(np.divide(arr_ux, np.hypot(arr_ux, arr_uy)))
    orientation[arr_uy <= 0] = -orientation[arr_uy <= 0]  # to have signed angles

    """
    # Where vx<0
    G = np.zeros((ny + 1, nx + 1))
    G[arr_ux >= 0] = 0
    G[arr_ux < 0] = 1

    label_im, nb_labels = ndimage.label(G, structure=np.ones((3, 3)))
    sizes = ndimage.sum(G, label_im, range(nb_labels + 1))
    sizes = sizes[sizes != np.max(sizes)]
    size_back = np.sum(sizes)

    # Area of the active part
    active = np.zeros((ny + 1, nx + 1))
    x_array = np.linspace(-dim_x / 2, dim_x / 2, nx + 1)
    for j in range(ny + 1):
        for i in range(nx + 1):
            if x_array[i] <= arr_interface[j, 0]:
                active[j, i] = 1
    size_active = np.sum(active)

    ratio = size_back / size_active

    area_with_time = np.asarray([time_simu, ratio])

    file_name = "results/Figures/" + folder_name + "/size_backward.csv"
    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(area_with_time)
    """

    # angles = []
    # x_array = np.linspace(-dim_x / 2, dim_x / 2, nx + 1)
    # for j in range(ny + 1):
    #     for i in range(nx + 1):
    #         if x_array[i] <= arr_interface[j, 0]:
    #             angles.append(orientation[j, i])

    # angles = np.asarray(angles)
    # # values = np.mean(angles), np.std(angles)
    # angles_with_time = np.asarray([time_simu])
    # # angles_with_time = np.concatenate((angles_with_time, np.asarray(values)))
    # angles_with_time = np.concatenate((angles_with_time, angles))

    # Save in csv file
    # file_name = "results/Figures/" + folder_name + "/angles2.csv"
    # with open(file_name, 'a') as file:
    #     writer = csv.writer(file, delimiter=' ')
    #     writer.writerow(angles_with_time)

    # To plot

    # For the color wheel
    x_val = np.arange(-np.pi, np.pi, 0.001)
    y_val = np.ones_like(x_val)
    norm_ = matplotlib.colors.Normalize(-np.pi, np.pi)

    fig = plt.figure()

    # Angles
    ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=4, rowspan=3)
    ax1.set_aspect('equal')
    plt.imshow(orientation, cmap='hsv', extent=[-dim_x / 2, dim_x / 2, -dim_y / 2, dim_y / 2], vmin=-np.pi, vmax=np.pi)
    plt.plot(arr_interface[:, 0], arr_interface[:, 1], c='green', linewidth=0.5)
    # plt.contour(label_im, [0.1], linewidths=0.1, colors='black', extent=[-dim_x / 2, dim_x / 2, -dim_y / 2, dim_y / 2])
    ax1.set(xlabel='x', ylabel='y', xlim=(-dim_x / 2, dim_x / 2), ylim=(-dim_y / 2, dim_y / 2))
    plt.title('Angle for t=' + str(int(time_simu * dt * 1000000) / 1000000))

    # Color wheel
    ax2 = plt.subplot2grid((3, 5), (0, 4), polar=True)
    ax2.scatter(x_val, y_val, c=x_val, s=50, cmap='hsv', norm=norm_, linewidths=0)
    ax2.set_yticks([])

    file_name = 'results/Figures/' + folder_name + '/Angles_' + str(time_simu) + '.png'
    plt.savefig(fname=file_name, dpi=500)
    plt.close(fig)

    return


def save_norm(arr_ux: np.ndarray, arr_uy: np.ndarray, nx: int, ny: int, dim_x: float, dim_y: float,
              time_simu: int, arr_interface: np.ndarray, folder_name: str, dt: float):
    """
    Computes the norm of the velocity at each point of the mesh, saves the norms in a csv file and as a figure
    :param arr_ux: values of vx
    :param arr_uy: values of vx
    :param nx: size of the mesh in the direction of x
    :param ny: size of the mesh in the direction of y
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :param time_simu: time of the simulation
    :param arr_interface: contains the coordinates of the interface (ny x 2)
    :param folder_name: name of the folder where to save the values
    :param dt: time step
    :return:
    """
    norm_v = np.sqrt(np.power(arr_ux, 2) + np.power(arr_uy, 2))

    norms = []
    x_array = np.linspace(-dim_x / 2, dim_x / 2, nx + 1)
    for j in range(ny + 1):
        for i in range(nx + 1):
            if x_array[i] <= arr_interface[j, 0]:
                norms.append(norm_v[j, i])

    norms = np.asarray(norms)
    norms_with_time = np.asarray([time_simu])
    norms_with_time = np.concatenate((norms_with_time, norms))

    # Save in csv file
    file_name = "results/Figures/" + folder_name + "/norms.csv"
    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(norms_with_time)

    fig = plt.figure()
    plt.imshow(norm_v, cmap='bwr', extent=[-dim_x / 2, dim_x / 2, -dim_y / 2, dim_y / 2], vmin=0, vmax=5)
    plt.colorbar()
    plt.plot(arr_interface[:, 0], arr_interface[:, 1], c='green', linewidth=0.5)
    plt.title('Norm of velocity for t=' + str(int(time_simu * dt * 1000000) / 1000000))
    plt.xlabel('x')
    plt.ylabel('y')
    file_name = 'results/Figures/' + folder_name + '/Norm_' + str(time_simu) + '.png'
    plt.savefig(fname=file_name, dpi=300)
    plt.close(fig)

    return


def save_streamlines(arr_ux: np.ndarray, arr_uy: np.ndarray, nx: int, ny: int, dim_x: float, dim_y: float,
                     time_simu: int, arr_interface: np.ndarray, folder_name: str, dt: float):
    """
    Computes some streamlines
    :param arr_ux: values of vx
    :param arr_uy: values of vx
    :param nx: size of the mesh in the direction of x
    :param ny: size of the mesh in the direction of y
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :param time_simu: time of the simulation
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param folder_name: name of the folder where to save the values
    :param dt: time step
    :return:
    """

    X = np.linspace(-dim_x / 2, dim_x / 2, nx + 1)
    Y = np.linspace(-dim_y / 2, dim_y / 2, ny + 1)

    speed = np.sqrt(np.power(arr_ux, 2) + np.power(arr_uy, 2))

    # Seeding points
    N = ny + 1
    start_x = np.ones(N) * (-dim_x / 2)
    start_y = np.linspace(-dim_y / 2, dim_y / 2, N)
    start = np.zeros((2 * N, 2))
    start[:N, 0] = start_x
    start[:N, 1] = start_y
    start[N:, 0] = arr_interface[:, 0]
    start[N:, 1] = arr_interface[:, 1]

    cm_ = 'jet'
    norm_ = matplotlib.colors.Normalize(vmin=0.7, vmax=1.3)
    sm = matplotlib.cm.ScalarMappable(cmap=cm_, norm=norm_)
    sm.set_array([])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.streamplot(X, Y, arr_ux, arr_uy, start_points=start, linewidth=0.5, arrowsize=0.2, color=speed, cmap=cm_,
                  norm=norm_)
    plt.plot(arr_interface[:, 0], arr_interface[:, 1], c='r', linewidth=0.5)
    plt.colorbar(sm)
    ax.set(xlabel='x', ylabel='y', xlim=(-dim_x / 2, dim_x / 2), ylim=(-dim_y / 2, dim_y / 2))
    plt.title('Streamlines for t=' + str(int(time_simu * dt * 1000000) / 1000000))
    file_name = 'results/Figures/' + folder_name + "/Streamlines_" + str(time_simu) + '.png'
    plt.savefig(fname=file_name, dpi=500)
    plt.close(fig)

    return


def interface_width(arr_phi: np.ndarray, folder_name: str, nx: int, ny: int, dim_x: float, dim_y: float, time_simu: int,
                    dt: float):
    """
    Computes the width of the interface (where phi goes from -0.9 to 0.9). Saves the results in a csv file
    :param arr_phi: values of phi for a given time
    :param folder_name: str, name of the folder where to save the values
    :param nx: size of the mesh in the direction of x
    :param ny: size of the mesh in the direction of y
    :param dim_x: dimension of the mesh in the direction of x
    :param dim_y: dimension of the mesh in the direction of y
    :param time_simu: int, time of the simulation
    :param dt: float, time step
    :return:
    """
    file_name = "results/Figures/" + folder_name + "/interface_width_2.csv"

    # Save where phi = 0.9 and -0.9
    arr_sup = arr_phi - 0.9 * np.ones((ny + 1, nx + 1))
    arr_inf = arr_phi + 0.9 * np.ones((ny + 1, nx + 1))
    lim_sup = np.zeros((ny + 1, 2))
    lim_inf = np.zeros((ny + 1, 2))
    for j in range(ny + 1):
        i_1 = np.argmin(abs(arr_sup[ny - j, :]))
        i_2 = np.argmin(abs(arr_inf[ny - j, :]))
        lim_sup[ny - j, :] = [i_1, j]
        lim_inf[ny - j, :] = [i_2, j]
    lim_sup = np.asarray(lim_sup)
    lim_inf = np.asarray(lim_inf)

    # Convert indexes to coordinates
    lim_sup[:, 0] = (lim_sup[:, 0] - ((nx + 1) / 2)) * dim_x / (nx + 1)
    lim_sup[:, 1] = (lim_sup[:, 1] - ((ny + 1) / 2)) * dim_y / (ny + 1)

    lim_inf[:, 0] = (lim_inf[:, 0] - ((nx + 1) / 2)) * dim_x / (nx + 1)
    lim_inf[:, 1] = (lim_inf[:, 1] - ((ny + 1) / 2)) * dim_y / (ny + 1)

    # Find the width of the interface, the mean width and its STD
    width = lim_sup[:, 0] - lim_inf[:, 0]

    for i in range(ny + 1):
        for j in range(max(0, i - 20), min(ny + 1, i + 20)):
            d = np.sqrt((lim_sup[i, 0] - lim_inf[j, 0]) ** 2 + (lim_sup[i, 1] - lim_inf[j, 1]) ** 2)
            if d < width[i]:
                width[i] = d

    width_with_time = np.asarray([time_simu * dt])
    width_with_time = np.concatenate((width_with_time, [np.mean(width), np.std(width)]))

    # Save in a csv file
    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(width_with_time)
    return


# Extracts the interface from the phase
def interface(arr_phi: np.ndarray, nx: int, ny: int, dim_x: float, dim_y: float) -> np.ndarray:
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
    arr_interface[:, 1] = (arr_interface[:, 1] - ((ny + 1) / 2)) * dim_y / (ny + 1)

    return arr_interface


#### Change a Dolfin function into an explicit array
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
    Extract the values of the pressure and return them into an array (works for any 1D function)
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


def phase_LSA(nx, ny, dim_x, start, vi, i, dt, Cahn, d_phi):
    x = np.linspace(-dim_x / 2, dim_x / 2, nx + 1)
    phi_lin_one = np.tanh((x - start - vi * i * dt) / (Cahn * np.sqrt(2)))
    for j in range(ny + 1):
        d_phi[j, :] += phi_lin_one
    return d_phi


### Hydrodynamic tests
# Not used anymore, should check if still works

def check_div_v(velocity: dolfin.function.function.Function, dim_x: float, dim_y: float, time_simu: int,
                folder_name: str) -> None:
    """
    Save div(v)
    :param velocity: dolfin function for the velocity
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param time_simu: time of the simulation
    :param folder_name: name of the folder where to save the files
    :return:
    """
    fig = plt.figure()
    p = dolfin.plot(dolfin.div(velocity))
    p.set_clim(-.5, .5)
    fig.colorbar(p, boundaries=[-.5, .5], cmap='jet')
    plt.title('Divergence for t=' + str(time_simu))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(-dim_y / 2, dim_y / 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Divergence_" + str(time_simu) + '.png')
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
                u: dolfin.function.function.Function, theta: float, Ca: float, dim_x: float, dim_y: float,
                time_simu: int, folder_name: str) -> None:
    """
    Check the hydrodynamic relation far from the interface
    :param velocity: dolfin function for the velocity
    :param pressure: dolfin function for the pressure
    :param u: dolfin function for the phase and mu
    :param theta: viscosity ratio
    :param Ca: Capillary number
    :param dim_x: dimension in the direction of x
    :param dim_y: dimension in the direction of y
    :param time_simu: time of the simulation
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
    plt.title('Hydro_x for t=' + str(time_simu))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Hydro_x_" + str(time_simu) + '.png')
    plt.close(fig)

    fig = plt.figure()
    plot_hydro_y = dolfin.plot(hydro[1])
    plot_hydro_y.set_clim(-.01, .01)
    fig.colorbar(plot_hydro_y, boundaries=[-.01, .01], cmap='jet')
    plt.title('Hydro_y for t=' + str(time_simu))
    ax = axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(-dim_x / 2, dim_x / 2)
    ax.set_ylim(0, dim_y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('results/Figures/' + folder_name + "/Checks/Hydro_y_" + str(time_simu) + '.png')
    plt.close(fig)
    return
