### Packages
import dolfin
from os import listdir
from os.path import isfile, join

# Imports
from results.main_results import save_interface_and_peaks, array_exp_phase, array_exp_flow, interface, save_fig, \
    save_quiver, arr_exp_pressure, interface_width, Delta_criterion, save_streamlines, velocity_orientation, save_norm, \
    array_exp_velocity

from model.model_common import mesh_from_dim
from model.model_phase import space_phase
from model.model_flow import space_flow

from model.unique_solver import space_common


def retrieve_param(folder_name: str):
    """
    Extract the parameters from the txt file and return them
    :param folder_name: Name of the folder of the test
    :return: list of parameters
    """
    file_param = "results/Figures/" + folder_name + "/param.txt"
    file = open(file_param, 'r')
    dim_x, dim_y, nx, ny, dt, theta, starting_point, h_0, k_wave, h = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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
        if not ('._' in file_path):
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
            save_fig(arr=arr_mu, name='Chemical_potential', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny,
                     theta=theta,
                     folder_name=folder_name)
            interface_width(arr_phi=arr_phi, folder_name=folder_name, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y,
                            time_simu=i, dt=dt)
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
    space_ME = space_phase(mesh=mesh)

    for file in files_flow:
        file_path = "results/Figures/" + folder_name + "/Solutions/" + file
        if not ('._' in file_path):
            if not ('Phi' in file_path):
                i = int(file[8:-3])
                u_flow = dolfin.Function(w_flow)
                input_file = dolfin.HDF5File(mesh.mpi_comm(), file_path, 'r')
                input_file.read(u_flow, 'Solution V_and_P')
                input_file.close()
                arr_ux, arr_uy, arr_p = array_exp_flow(u_flow=u_flow, mesh=mesh, nx=nx, ny=ny)

                # Get the interface to display it as well
                file_path_phase = "results/Figures/" + folder_name + "/Solutions/Phi_and_mu_" + str(i) + ".h5"
                u_phase = dolfin.Function(space_ME)
                input_file_phase = dolfin.HDF5File(mesh.mpi_comm(), file_path_phase, 'r')
                input_file_phase.read(u_phase, 'Solution Phi_and_mu')
                input_file_phase.close()
                arr_phi, arr_mu = array_exp_phase(u=u_phase, mesh=mesh, nx=nx, ny=ny)
                arr_interface = interface(arr_phi=arr_phi, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)

                save_fig(arr=arr_ux, name='Vx', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                         folder_name=folder_name)
                save_fig(arr=arr_uy, name='Vy', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                         folder_name=folder_name)
                save_fig(arr=arr_p, name='Pressure', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                         folder_name=folder_name)
                # save_quiver(arr_ux=arr_ux, arr_uy=arr_uy, time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny,
                #             folder_name=folder_name, starting_point=starting_point, dt=dt)
                # Delta_criterion(u_flow=u_flow, mesh=mesh, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y, time_simu=i,
                #                 arr_interface=arr_interface, folder_name=folder_name)
                # velocity_orientation(arr_ux=arr_ux, arr_uy=arr_uy, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y,
                #                       time_simu=i,arr_interface=arr_interface, folder_name=folder_name)
                # save_norm(arr_ux=arr_ux, arr_uy=arr_uy, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y, time_simu=i,
                #           arr_interface=arr_interface, folder_name=folder_name)
                # save_streamlines(arr_ux=arr_ux, arr_uy=arr_uy, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y, time_simu=i,
                #                  arr_interface=arr_interface, folder_name=folder_name)

    return print('Flow extracted')


def extract(folder_name: str):
    # List of files
    mypath = "results/Figures/" + folder_name + "/Solutions/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles.sort()

    # Initiate
    dim_x, dim_y, nx, ny, dt, theta, starting_point, h_0, k_wave, h = retrieve_param(folder_name=folder_name)
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    mixed_space = space_common(mesh=mesh)

    for file in onlyfiles:
        file_path = "results/Figures/" + folder_name + "/Solutions/" + file
        if not ('._' in file_path):
            i = int(file[10:-3])
            u = dolfin.Function(mixed_space)
            input_file = dolfin.HDF5File(mesh.mpi_comm(), file_path, 'r')
            input_file.read(u, 'Solution Functions')
            input_file.close()

            # From dolfin functions to arrays
            velocity, pressure, phi, mu = u.split()
            arr_p = arr_exp_pressure(pressure, mesh=mesh, nx=nx, ny=ny)
            arr_phi = arr_exp_pressure(phi, mesh=mesh, nx=nx, ny=ny)
            arr_mu = arr_exp_pressure(mu, mesh=mesh, nx=nx, ny=ny)
            arr_ux, arr_uy = array_exp_velocity(velocity, mesh=mesh, nx=nx, ny=ny)

            # Phase field
            arr_interface = interface(arr_phi=arr_phi, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
            save_interface_and_peaks(arr_interface=arr_interface, folder_name=folder_name, time_simu=i, dt=dt,
                                     h_0=h_0, starting_point=starting_point, k_wave=k_wave, h=h)
            save_fig(arr=arr_phi, name='Phase', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                     folder_name=folder_name)
            save_fig(arr=arr_mu, name='Chemical_potential', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny,
                     theta=theta, folder_name=folder_name)
            interface_width(arr_phi=arr_phi, folder_name=folder_name, nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y,
                            time_simu=i, dt=dt)

            # Flow
            save_fig(arr=arr_ux, name='Vx', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                     folder_name=folder_name)
            save_fig(arr=arr_uy, name='Vy', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                     folder_name=folder_name)
            save_fig(arr=arr_p, name='Pressure', time_simu=i, dim_x=dim_x, dim_y=dim_y, nx=nx, ny=ny, theta=theta,
                     folder_name=folder_name)
    return


simulations = ['8-7-2020#10']#, '8-7-2020#2', '8-7-2020#3', '8-7-2020#4', '8-7-2020#5', '8-7-2020#6', '8-7-2020#7', '8-7-2020#8']
for x in simulations:
    print('Extracting')
    folder_name = str(x)
    # extract_for_phase(folder_name=folder_name)
    # extract_for_flow(folder_name=folder_name)
    extract(folder_name=folder_name)
