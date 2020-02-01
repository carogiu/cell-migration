### Packages
import numpy as np
from scipy.signal import find_peaks

def main_distance_save(folder_name, arr_interface_tot, n):
    dist = distance_time(arr_interface_tot,n)
    path = "results/Figures/" + folder_name + "/distance.csv"
    np.savetxt(path, dist, delimiter=",")
    return


def distance_time(arr_interface_tot,n):
    dist = np.zeros(n)
    for i in range(n):
        arr_interface = arr_interface_tot[:, :, i]
        d = distance_avg(arr_interface)
        dist[i] = d
    return dist


def distance_avg(arr_interface):
    peaks_t, _ = find_peaks(arr_interface[:, 0])
    peaks_b, _ = find_peaks(-arr_interface[:, 0])
    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()]
    num_peaks = peaks.shape[0]
    d = 0
    for p in range(num_peaks - 1):
        d = d + abs(peaks[p, 0] - peaks[p + 1, 0])
    d = d / (num_peaks - 1)
    return d
