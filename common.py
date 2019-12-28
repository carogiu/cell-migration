from flow import *
from phase import *

def mesh_from_dim(nx, ny):
    """
    Creates mesh of dimension nx, ny of type quadrilateral

    :param nx: number of cells in x direction
    :param ny: number of cells in y direction
    :return: mesh
    """
    mesh = UnitSquareMesh.create(nx, ny, CellType.Type.quadrilateral)
    return mesh

def save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i, mesh):
    save_phi(phi_tot, u, i, mesh)
    save_flow(vx_tot, vy_tot, p_tot, U_flow, i, mesh)

def visu(arr, name, time):
    fig = plt.figure()
    plt.imshow(arr, cmap='jet')
    plt.colorbar()
    plt.title(name + ' for t=' + time)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close(fig)

def see_all(arr_tot, name):
    n = arr_tot.shape[2]
    for i in range(n):
        if i % 2 == 0:
            arr = arr_tot[:, :, i]
            time = str(int(i))
            visu(arr, name, time)

def time_evolution(ME, W_flow, vi, theta, factor, epsilon, mid, dt, M, n, mesh):
    nx = ny = int(np.sqrt(mesh.num_cells()))
    phi_test, mu_test, du, u, phi, mu, u0, phi_0, mu_0 = initiate_phase(ME)
    U_flow = problem_coupled(W_flow, phi, mu, vi, theta, factor, epsilon)
    velocity, pressure = split(U_flow)
    # save the solutions
    phi_tot = np.zeros((nx + 1, ny + 1, n))
    vx_tot = np.zeros((nx + 1, ny + 1, n))
    vy_tot = np.zeros((nx + 1, ny + 1, n))
    p_tot = np.zeros((nx + 1, ny + 1, n))
    save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, 0, mesh)
    for i in range(1, n):
        a, L, u = problem_phase_with_epsilon(phi_test, mu_test, du, u, phi, mu, phi_0, mu_0, velocity, mid, dt, M, epsilon)
        u0.vector()[:] = u.vector()
        u = solve_phase(a, L, u)
        U_flow = problem_coupled(W_flow, phi, mu, vi, theta, factor, epsilon)
        velocity, pressure = split(U_flow)
        save(phi_tot, vx_tot, vy_tot, p_tot, u, U_flow, i, mesh)

    return phi_tot, vx_tot, vy_tot, p_tot