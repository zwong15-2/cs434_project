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

def plot_data(xs):
    plt.plot(range(len(xs)), xs)
    plt.show()

# NOT USED: gets lower outliers of dataset using IQR
def find_outliers(xs):
    q1, q3 = np.percentile(xs,[25,75])
    iqr = q3 - q1
    #print("IQR:", iqr)
    acceptable_range = q1 - (1.5*iqr)
    #print("Acceptable range:", acceptable_range)
    outliers = []
    for x in xs:
        if x < acceptable_range:
            outliers.append(x)
    return outliers

# Get magnitude of row by using Pythagorean on x,y,z (1,2,3) IMU data
def get_magnitude(row):
    return ( (row[1]**2) + (row[2]**2) + (row[3]**2) )**(0.5)

# ---- Read in data ----
acc_filename = "./data_imu_loc/route1/Accelerometer.csv"
gyro_filename = "./data_imu_loc/route1/Gyroscope.csv"
acc_data = read_data(acc_filename)
gyro_data = read_data(gyro_filename)

# ---- Filter & count steps from acc_data----
acc_data_mags = [get_magnitude(row) for row in acc_data]
filtered_data = low_pass_filter(acc_data_mags)
# threshold gives allowed vertical distance between a peak and its surrounding samples
(peaks_indices, props) = find_peaks(filtered_data, threshold=0.0001)
peaks = [filtered_data[i] for i in peaks_indices]
num_steps = len(peaks)

print(num_steps, "steps")
