import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt, find_peaks

# Sampling frequency
f_sampling = 100
# Nyquist frequency
f_nyquist = 0.5 * f_sampling
# Cutoff frequency we want for low-pass filter
f_cutoff = 2
f_cutoff_ratio = f_cutoff / f_nyquist
order = 2
delta_t = 0.01

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
def magnitude(row):
    return ( (row[1]**2) + (row[2]**2) + (row[3]**2) )**(0.5)

# Used to plot peak indices against a range of acc data (3000,5000)
def plot_range(step_indices, filtered_data):
    filtered_step_indices = list(filter(lambda x: x < 5000 and x >= 3000, step_indices))
    filtered_peaks = [filtered_data[i] for i in filtered_step_indices]
    plt.plot(range(3000,5000),acc_data_mags[3000:5000])
    plt.plot(filtered_step_indices, filtered_peaks, 'rs')
    plt.show()

# Find out which leg has the stronger steps by looking at even / odd peaks
def get_pocket_leg_info(step_indices, filtered_data):
    even_step_indices = step_indices[0::2]
    even_peaks = [filtered_data[i] for i in even_step_indices]
    odd_step_indices = step_indices[1::2]
    odd_peaks = [filtered_data[i] for i in odd_step_indices]

    pl_step_indices = even_step_indices
    non_pl_step_indices = odd_step_indices
    pl_peaks = even_peaks
    non_pl_peaks = odd_peaks
    if np.mean(odd_peaks) > np.mean(even_peaks):
        pl_step_indices = odd_step_indices
        non_pl_step_indices = even_step_indices
        pl_peaks = odd_peaks
        non_pl_peaks = even_peaks

    return (pl_step_indices, non_pl_step_indices, pl_peaks, non_pl_peaks)


# Using accelerometer data (first row of CSV), get initial orientation
# (assumes projection of phone’s local Y axis onto hor. plane is towards the global Y axis)
def get_init_orientation(acc_unnormalized):
    # Get v1 from v2, using assumption that proj. of phone’s local Y axis onto hor. plane is towards North
    v2 = acc_unnormalized / la.norm(acc_unnormalized)
    proj_loc_y_onto_v2 = ((np.dot([0, 1, 0], v2)/(la.norm(v2)**2)) * v2)
    proj_loc_y_onto_hor = [0, 1, 0] - proj_loc_y_onto_v2
    v1 = proj_loc_y_onto_hor / la.norm(proj_loc_y_onto_hor)
    [ax, ay, az] = v2
    [mx, my, mz] = v1
    a = my*az - ay*mz
    b = -(mx*az - ax*mz)
    c = mx*ay - ax*my
    # Solve system of equations: v1*row1 = 0; v2*row2 = 0; det(R) = 1
    A = np.array([v1, v2, [a, b, c]])
    row1 = la.solve(A, [0,0,1])
    R = np.array([row1, v1, v2])
    return R

def get_gyro_readings_between_times(first_step_time, second_step_time):
    return filter(lambda row: row[0] >= first_step_time and row[0] <= second_step_time, gyro_data)

def axis_angle_to_matrix(axis, theta):
    normalized_axis = axis / la.norm(axis)
    [ux, uy, uz] = normalized_axis
    # Source: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = np.array([[np.cos(theta)+((ux**2)*(1-np.cos(theta))), (ux*uy*(1-np.cos(theta)))-(uz*np.sin(theta)), (ux*uz*(1-np.cos(theta)))+(uy*np.sin(theta))],
                  [(ux*uy*(1-np.cos(theta)))+(uz*np.sin(theta)), np.cos(theta)+((uy**2)*(1-np.cos(theta))), (uy*uz*(1-np.cos(theta)))-(ux*np.sin(theta))],
                  [(ux*uz*(1-np.cos(theta)))-(uy*np.sin(theta)), (uy*uz*(1-np.cos(theta)))+(ux*np.sin(theta)), np.cos(theta)+((uz**2)*(1-np.cos(theta)))]])
    return R

def integrate_gyros(gyro_data, R_init):
    delta_Ri = R_init
    for l in gyro_data:
        delta_theta = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2) * delta_t
        # Project instant rotation axis l into global frame
        proj_l = l
        new_delta_Ri = axis_angle_to_matrix(proj_l, delta_theta)
        # R(i+1) = delta_Ri * Ri
        delta_Ri = delta_Ri @ new_delta_Ri
    return delta_Ri

def matrix_to_axis(A):
    a = A[0,0]
    b = A[0,1]
    c = A[0,2]
    d = A[1,0]
    e = A[1,1]
    f = A[1,2]
    g = A[2,0]
    h = A[2,1]
    i = A[2,2]
    axis = np.array([h-f, c-g, d-b])
    return axis / la.norm(axis)

# ---- Read in data ----
acc_filename = "./data_imu_loc/route1/Accelerometer.csv"
gyro_filename = "./data_imu_loc/route1/Gyroscope.csv"
acc_data = read_data(acc_filename)
gyro_data = read_data(gyro_filename)

# # Change ms to s?
# acc_data = [[row[0]/1000.0, row[1], row[2], row[3]] for row in acc_data]
# gyro_data = [[row[0]/1000.0, row[1], row[2], row[3]] for row in gyro_data]

init_acc_data = acc_data[0][1:]
R0 = get_init_orientation(init_acc_data)
acc_times = [row[0] for row in acc_data]
gyro_times = [row[0] for row in gyro_data]


# ---- Filter & count steps from acc_data----
acc_data_mags = [magnitude(row) for row in acc_data]
mag_avg = np.mean(acc_data_mags)
acc_data_mags = [(mag - mag_avg) for mag in acc_data_mags]


# Threshold gives allowed vertical distance between a peak and its surrounding samples
filtered_data = low_pass_filter(acc_data_mags)
(step_indices, props) = find_peaks(filtered_data, threshold=0.0001)
peaks = [filtered_data[i] for i in step_indices]
num_steps = len(peaks)
step_times = [acc_data[i][0] for i in step_indices]

# # Find which of the leg's steps (even / odd) has greater average mag
# # This is the pocket leg
# (pl_step_indices, non_pl_step_indices, pl_peaks, non_pl_peaks) = get_pocket_leg_info(step_indices, filtered_data)
# plot_range(pl_step_indices, filtered_data)

locs = []
locs.append(np.array([0,0,0]))
delta_R = R0
# For each 2 successive steps for the leg,
for i in range(len(step_times) - 1):
    # Get all gyro readings between those 2 steps
    gyros = get_gyro_readings_between_times(step_times[i], step_times[i+1])
    # Integrate those gyro readings and multiply them w/ orientation
    gyros = [gyro[1:] for gyro in gyros]
    delta_R = integrate_gyros(gyros, delta_R)
    # Express that as axis-angle
    axis = matrix_to_axis(delta_R)
    # Get orthogonal vector -> this is walking direction for that step
    # (for now I'm just gonna use the axis)
    walking_dir = np.array(axis)
    walking_dir = walking_dir / la.norm(walking_dir)
    # Multiply that walking direction w/ step length
    # (for now I'm just gonna assume step length of 0.5m)
    step_length = 0.5
    disp_of_step = step_length * walking_dir
    locs.append(locs[i] + disp_of_step)

locs = np.array(locs)
plt.plot(locs[:,0], locs[:,1])
plt.show()

print(mag_avg)
