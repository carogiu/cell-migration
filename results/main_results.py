### Packages
import numpy as np
from scipy.signal import find_peaks
import csv


def all_peaks(arr_interface):
    peaks_t, _ = find_peaks(arr_interface[:, 0])
    peaks_b, _ = find_peaks(-arr_interface[:, 0])
    peaks = np.concatenate((arr_interface[peaks_t, :], arr_interface[peaks_b, :]), axis=0)
    peaks = peaks[peaks[:, 1].argsort()]
    return peaks


def save_peaks(folder_name, arr_interface, h_0):
    peaks = all_peaks(arr_interface)
    summits = peaks[:, 0]
    n = len(summits)
    instability = np.zeros(n - 1)
    for i in range(n - 1):
        d = abs(summits[i] - summits[i + 1])
        if d >= h_0 * 2 * 0.9:
            instability[i] = d #- h_0 * 2
    file_name = "results/Figures/" + folder_name + "/peaks.csv"
    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(instability)
    return
