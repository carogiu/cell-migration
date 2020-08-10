### Packages
import dolfin
import time

### Imports
from model.model_common import mesh_from_dim, \
    time_evolution_passive_darcy_2_solvers, time_evolution_active_darcy_2_solvers, \
    time_evolution_passive_div_darcy_2_solvers, \
    time_evolution_passive_toner_tu_2_solvers, time_evolution_active_tuner_tu_2_solvers
from model.model_phase import space_phase
from model.model_flow import space_flow
from model.unique_solver import space_common, time_evolution_general, time_evolution_toner_tu

### Optimization parameters
dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["cpp_optimize"] = True


### Main function

def main_model(config):
    """
    Run complete model from global main parameters and observe results.
    """
    # Retrieve parameters

    # Grid parameters
    nx, ny = config.nx, config.ny
    dim_x, dim_y = config.dim_x, config.dim_y

    # Time parameters
    n, dt = config.n, config.dt

    # Model parameters
    theta = config.theta
    alpha = config.alpha
    vi = config.vi
    k = config.k
    Cahn = config.Cahn
    Pe = config.Pe
    Ca = config.Ca
    starting_point = config.starting_point
    model = config.model_type

    # Initial perturbation parameters
    h_0, k_wave = config.h_0, config.k_wave

    # Saving parameter
    folder_name = config.folder_name

    # Make sure that we simulated growth properly
    if k != 0:
        if model != 'darcy':
            print('Model must be Darcy for growth!')
            print('Simulation stopped')
            return
        if vi != 0:
            print('Inflow must be zero for growth!')
            print('Simulation stopped')
            return
        if alpha != 0:
            print('Activity must be zero for growth!')
            print('Simulation stopped')
            return

    # Create Mesh
    mesh = mesh_from_dim(nx=nx, ny=ny, dim_x=dim_x, dim_y=dim_y)
    # space_ME = space_phase(mesh=mesh)
    # w_flow = space_flow(mesh=mesh)
    mixed_space = space_common(mesh=mesh)

    print('Expected computation time = ' + str(nx * ny * n * 5.5E-4 / 60) + ' minutes')  # 5e-4 on Mac 2e-4 on big Linux
    t1 = time.time()

    # Compute the model

    # One solver
    # The 2 flow equations and the 2 phase field equations are all solved together

    if model == 'darcy':
        print('Simulating Darcy')
        time_evolution_general(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, mixed_space=mixed_space, theta=theta,
                               Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave, vi=vi,
                               alpha=alpha, k=k, folder_name=folder_name)
    else:
        print('Simulating Toner-Tu')
        time_evolution_toner_tu(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, mixed_space=mixed_space, theta=theta,
                                Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave, vi=vi,
                                alpha=alpha, folder_name=folder_name)

    # Two solvers
    # In that case, the phase field is solved first with the solution of the velocity from the previous time step,
    # and the flow is then solved with the new values of the phase field.
    # Less exact, but faster and slight difference in the growth rate
    """
    if model == 'darcy':
        print('Simulating Darcy')
        if alpha == 0:
            print('Simulation without activity')
            if k == 0:
                print('No division')
                time_evolution_passive_darcy_2_solvers(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n,
                                                       space_ME=space_ME, w_flow=w_flow, theta=theta, Cahn=Cahn, Pe=Pe,
                                                       Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave,
                                                       vi=vi, folder_name=folder_name)
            else:
                print('Division')
                time_evolution_passive_div_darcy_2_solvers(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n,
                                                           space_ME=space_ME, w_flow=w_flow, theta=theta, Cahn=Cahn,
                                                           Pe=Pe, Ca=Ca, k=k, starting_point=starting_point, h_0=h_0,
                                                           k_wave=k_wave, folder_name=folder_name)
        else:
            print('Simulation with activity')
            time_evolution_active_darcy_2_solvers(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n, space_ME=space_ME,
                                                  w_flow=w_flow, theta=theta, alpha=alpha, Cahn=Cahn, Pe=Pe, Ca=Ca,
                                                  starting_point=starting_point, h_0=h_0, k_wave=k_wave, vi=vi,
                                                  folder_name=folder_name)
    else:
        print('Simulating Toner-Tu')
        if alpha == 0:
            print('Simulation without activity')
            time_evolution_passive_toner_tu_2_solvers(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n,
                                                      space_ME=space_ME, w_flow=w_flow, theta=theta, Cahn=Cahn, Pe=Pe,
                                                      Ca=Ca, starting_point=starting_point, h_0=h_0, k_wave=k_wave,
                                                      vi=vi, folder_name=folder_name)
        else:
            print('Simulation with activity')
            time_evolution_active_tuner_tu_2_solvers(mesh=mesh, dim_x=dim_x, dim_y=dim_y, dt=dt, n=n,
                                                     space_ME=space_ME, w_flow=w_flow, theta=theta, alpha=alpha,
                                                     Cahn=Cahn, Pe=Pe, Ca=Ca, starting_point=starting_point, h_0=h_0,
                                                     k_wave=k_wave, vi=vi, folder_name=folder_name)
    """
    t2 = time.time()
    print('Total computation time = ' + str((t2 - t1) / 60) + ' minutes')

    return
