import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy.signal import butter,filtfilt, find_peaks

# Sampling frequency
f_sampling = 100
# Nyquist frequency
f_nyquist = 0.5 * f_sampling
# Cutoff frequency we want for low-pass filter
f_cutoff = 2
f_cutoff_ratio = f_cutoff / f_nyquist
order = 2

def read_data(file_name):
    data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            if not('' in row):
                data.append(row)
    return data

# Uses scipy.butter to apply low-pass filter
def low_pass_filter(data):
    # f_cutoff_ratio gives ratio of the <desired frequency = 2Hz> : <Nyquist frequency = 0.5*100Hz>
    b, a = butter(order, f_cutoff_ratio, btype='low', analog=False)
    return filtfilt(b, a, data)

# Get magnitude of row by using Pythagorean on x,y,z (1,2,3) IMU data
def get_magnitude(row):
    return ( (row[1]**2) + (row[2]**2) + (row[3]**2) )**(0.5)

# Check that every 2 valleys have peak between them
def check_peaks_valleys(peaks_indices, valleys_indices):
    if peaks_indices[0] < valleys_indices[0]:
        peaks_indices = peaks_indices[1:]

    num_not_between = 0
    for i in range(len(valleys_indices) - 1):
        if i >= len(peaks_indices):
            break
        valley_i = valleys_indices[i]
        valley_i_1 = valleys_indices[i+1]
        peak_i = peaks_indices[i]
        if not(valley_i < peak_i and peak_i < valley_i_1):
            # print("Valleys:", valley_i, valley_i_1)
            # print("Peak:", peak_i)
            num_not_between += 1
            # Delete
            if (peak_i <= valley_i):
                continue
            if (peak_i >= valley_i_1):
                continue
    return num_not_between

# ---- Read in data ----
acc_filename = "./data_imu_loc/route1/Accelerometer.csv"
gyro_filename = "./data_imu_loc/route1/Gyroscope.csv"
acc_data = read_data(acc_filename)
gyro_data = read_data(gyro_filename)

# Change ms to s
acc_data = [[row[0]/1000.0, row[1], row[2], row[3]] for row in acc_data]
gyro_data = [[row[0]/1000.0, row[1], row[2], row[3]] for row in gyro_data]

acc_times = [row[0] for row in acc_data]
gyro_times = [row[0] for row in gyro_data]


# ---- Filter & count steps from acc_data----
acc_data_mags = [get_magnitude(row) for row in acc_data]
mag_avg = np.mean(acc_data_mags)
acc_data_mags = [(mag - mag_avg) for mag in acc_data_mags]

# # Plot acc magnitudes
# plt.plot(acc_data_mags[:10000])
# plt.show()

# threshold gives allowed vertical distance between a peak and its surrounding samples
filtered_data = low_pass_filter(acc_data_mags)
(peaks_indices, props) = find_peaks(filtered_data, threshold=0.0001)
(valleys_indices, props) = find_peaks(-1*filtered_data, threshold=0.0001)
peaks = [filtered_data[i] for i in peaks_indices]
valleys = [filtered_data[i] for i in valleys_indices]
num_steps = len(peaks)
num_valleys = len(valleys)

print(mag_avg)
print(num_discrepancies)

print(num_steps, "peaks")
print(num_valleys, "valleys")
