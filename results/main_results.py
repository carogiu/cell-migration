### Packages
import numpy as np
from scipy.signal import find_peaks
import csv


def all_peaks(arr_interface):
    """
    To find the peaks of the interface
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :return: array, contains the coordinates of the peaks sorted by ordinates (y axis)
    """
    peaks_t, _ = find_peaks(arr_interface[:, 0])
    peaks_b, _ = find_peaks(-arr_interface[:, 0])
    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()]
    return peaks


def save_peaks(folder_name, arr_interface, h_0):
    """
    Finds the peaks of the interface and then saves the distance between two peaks if it is bigger than the initial instability
    :param folder_name: str, name of the folder were to save the values
    :param arr_interface: array, contains the coordinates of the interface (ny x 2)
    :param h_0: float, size of the initial instability
    :return:
    """
    peaks = all_peaks(arr_interface)
    summits = peaks[:, 0]
    n = len(summits)
    instability = np.zeros(n - 1)
    for i in range(n - 1):
        d = abs(summits[i] - summits[i + 1])
        if d >= h_0 * 2 * 0.9:
            instability[i] = d  # - h_0 * 2
    file_name = "results/Figures/" + folder_name + "/peaks.csv"
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(instability)
    return
