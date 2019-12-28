### Packages
import matplotlib.pyplot as plt


### Main function

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
