### Packages
import dolfin
from os import listdir
from os.path import isfile, join

# Imports
from results.main_results import save_interface_and_peaks, array_exp_phase, array_exp_flow, interface, save_fig, \
    save_quiver
from model.model_common import mesh_from_dim
from model.model_phase import space_phase
from model.model_flow import space_flow


def retrieve_param(folder_name: str):
    """
    Extract the parameters from the txt file and return them
    :param folder_name: Name of the folder of the test
    :return: list of parameters
    """
    file_param = "results/Figures/" + folder_name + "/param.txt"
    file = open(file_param, 'r')

    i = 0
    # Saving all the parameters in the good variable
    for line in file:
        if i == 2:
            h = float(line[3:])
        if i == 3:
            dim_x = float(line[7:])
        if i == 4:
            dim_y = float(line[7:])
        if i == 5:
            nx = int(line[4:])
        if i == 6:
            ny = int(line[4:])
        if i == 7:
            n = int(line[3:])
        if i == 8:
            dt = float(line[4:])
        if i == 9:
            theta = float(line[7:])
        if i == 10:
            Cahn = float(line[6:])
        if i == 11:
            Pe = float(line[4:])
        if i == 12:
            Ca = float(line[4:])
        if i == 13:
            starting_point = float(line[16:])
        if i == 14:
            h_0 = float(line[5:])
        if i == 15:
            k_wave = float(line[8:])
        if i == 16:
            sigma = float(line[7:])
        if i == 17:
            q = float(line[3:])
        i += 1
    file.close()
    return dim_x, dim_y, nx, ny, dt, theta, starting_point, h_0, k_wave, h


def extract_files(folder_name: str):
    """
    Extract the names of the solutions files and return them separately for the phase and the flow
    :param folder_name: Name of the folder of the test
    :return: list of str, with the names of the files
    """
    mypath = "results/Figures/" + folder_name + "/Solutions/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()
    t = int((len(onlyfiles) - 1) / 2)

    files_phase = onlyfiles[0:t + 1]
    files_flow = onlyfiles[t + 1:]
    return files_phase, files_flow


def extract_for_phase(folder_name: str):
    """
    From the solution files, save all the figures for the phase , extract the coordinates of the interface and the
    coordinates of the peaks
    :param folder_name: Name of the folder of the test
    :return:
    """
    files_phase, _ = extract_files(folder_name=folder_name)
    dim_x, dim_y, nx, ny, dt, theta, starting_point, h_0, k_wave, h = retrieve_param(folder_name=folder_name)
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    space_ME = space_phase(mesh=mesh)

    for file in files_phase:
        file_path = "results/Figures/" + folder_name + "/Solutions/" + file
        i = int(file[11:-3])
        u = dolfin.Function(space_ME)
        input_file = dolfin.HDF5File(mesh.mpi_comm(), file_path, 'r')
        input_file.read(u, 'Solution Phi_and_mu')
        input_file.close()
        arr_phi, arr_mu = array_exp_phase(u=u, mesh=mesh, nx=nx, ny=ny)
        arr_interface = interface(arr_phi=arr_phi, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
        save_interface_and_peaks(arr_interface=arr_interface, folder_name=folder_name, time_simu=i, dt=dt, h_0=h_0,
                                 starting_point=starting_point, k_wave=k_wave, h=h)
        save_fig(arr=arr_phi, name='Phase', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                 folder_name=folder_name)
    return print('Phase extracted')


def extract_for_flow(folder_name: str):
    """
    From the solution files, save all the figures for the flow
    :param folder_name: Name of the folder of the test
    :return:
    """
    _, files_flow = extract_files(folder_name=folder_name)
    dim_x, dim_y, nx, ny, dt, theta, starting_point, h_0, k_wave, h = retrieve_param(folder_name=folder_name)
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    w_flow = space_flow(mesh=mesh)
    for file in files_flow:
        file_path = "results/Figures/" + folder_name + "/Solutions/" + file
        i = int(file[8:-3])
        u_flow = dolfin.Function(w_flow)
        input_file = dolfin.HDF5File(mesh.mpi_comm(), file_path, 'r')
        input_file.read(u_flow, 'Solution V_and_P')
        input_file.close()
        arr_ux, arr_uy, arr_p = array_exp_flow(u_flow=u_flow, mesh=mesh, nx=nx, ny=ny)
        save_fig(arr=arr_ux, name='Vx', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                 folder_name=folder_name)
        save_fig(arr=arr_uy, name='Vy', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                 folder_name=folder_name)
        save_fig(arr=arr_p, name='Pressure', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                 folder_name=folder_name)
        save_quiver(arr_ux=arr_ux, arr_uy=arr_uy, time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny,
                    folder_name=folder_name, starting_point=starting_point, dt=dt)
    return print('Flow extracted')


print('Which folder? ')
x = input()
print('Extracting')
folder_name = str(x)
extract_for_phase(folder_name=folder_name)
extract_for_flow(folder_name=folder_name)
