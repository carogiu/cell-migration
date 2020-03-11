### Packages
import dolfin
from pylab import *


### Main save
def save_HDF5(function, function_name: str, mesh, time_simu: int, folder_name: str):
    name_hdF5File = "results/Figures/" + folder_name + "/Solutions/" + function_name + "_" + str(time_simu) + ".h5"

    name_solution = 'Solution ' + function_name
    output_file = dolfin.HDF5File(mesh.mpi_comm(), name_hdF5File, "w")
    output_file.write(u=function, name=name_solution)
    output_file.close()
    return
