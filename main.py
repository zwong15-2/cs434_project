import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt, find_peaks, lfilter

# Sampling frequency
f_sampling = 100
# Nyquist frequency
f_nyquist = 0.5 * f_sampling
# Cutoff frequency we want for low-pass filter
f_cutoff = 2.5
f_cutoff_ratio = f_cutoff / f_nyquist
order = 5
delta_t = 0.01

peak_threshold=0.00001
peak_prominence=1.0
# Peaks only every 0.25s or more
peak_distance=0.3/0.01

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

def high_pass_filter(data):
    b, a = butter(order, f_cutoff_ratio, btype='high', analog=False)
    return filtfilt(b, a, data)

# Get magnitude of row by using Pythagorean on x,y,z (1,2,3) IMU data
def magnitude(row):
    return ( (row[1]**2) + (row[2]**2) + (row[3]**2) )**(0.5)

# Find out which leg has the stronger steps by looking at even / odd peaks
def get_pocket_leg_info(step_indices, filtered_data):
    even_step_indices = step_indices[0::2]
    even_peaks = [filtered_data[i] for i in even_step_indices]
    odd_step_indices = step_indices[1::2]
    odd_peaks = [filtered_data[i] for i in odd_step_indices]

    pl_step_indices, non_pl_step_indices = even_step_indices, odd_step_indices
    pl_peaks, non_pl_peaks = even_peaks, odd_peaks
    if np.mean(odd_peaks) > np.mean(even_peaks):
        pl_step_indices, non_pl_step_indices = odd_step_indices, even_step_indices
        pl_peaks, non_pl_peaks = odd_peaks, even_peaks

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
    gyros = list(filter(lambda row: row[0] > first_step_time and row[0] <= second_step_time, gyro_data))
    return [gyro[1:] for gyro in gyros]

def get_gyro_delta_Rs_between_times(first_step_time, second_step_time, all_delta_Rs):
    start_index = ts_to_index(first_step_time)
    end_index = ts_to_index(second_step_time)
    return all_delta_Rs[start_index : end_index]

def axis_angle_to_matrix(axis, theta):
    normalized_axis = axis / la.norm(axis)
    [ux, uy, uz] = normalized_axis
    c = np.cos(theta)
    s = np.sin(theta)
    # Source: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = np.array([[c+((ux**2)*(1-c)),    (ux*uy*(1-c))-(uz*s), (ux*uz*(1-c))+(uy*s)],
                  [(ux*uy*(1-c))+(uz*s), c+((uy**2)*(1-c)),    (uy*uz*(1-c))-(ux*s)],
                  [(ux*uz*(1-c))-(uy*s), (uy*uz*(1-c))+(ux*s), c+((uz**2)*(1-c))]])
    return R

# Integrate gyro data (within a step)
def integrate_gyros(gyro_data, R):
    delta_R = R
    # Looking at gyro data for each timestamp within the step
    for l in gyro_data:
        delta_theta = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2) * delta_t
        l_proj = delta_R @ l
        delta_R_gyro = axis_angle_to_matrix(l_proj, delta_theta)
        delta_R = delta_R_gyro @ delta_R
    return delta_R

# Multiply all gyro_delta_Rs together and return
def integrate_gyro_delta_Rs(gyro_delta_Rs):
    delta_R_for_step = np.eye(3)
    for gyro_delta_R in gyro_delta_Rs:
        delta_R_for_step = gyro_delta_R @ delta_R_for_step
    return delta_R_for_step

# Integrate all gyro readings, put these matrices in a (massive) array, return
def integrate_all_gyros(gyro_data, R):
    Ri = R0
    all_Rs = []
    all_delta_Rs = []
    for l in gyro_data:
        delta_theta = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2) * delta_t
        proj_l = Ri @ l
        delta_Ri = axis_angle_to_matrix(proj_l, delta_theta)
        Ri = delta_Ri @ Ri
        all_Rs.append(Ri)
        all_delta_Rs.append(delta_Ri)
    return all_Rs, all_delta_Rs

# Convert rotation matrix to axis & angle
def matrix_to_axis_angle(A):
    a,b,c = A[0,0],A[0,1],A[0,2]
    d,e,f = A[1,0],A[1,1],A[1,2]
    g,h,i = A[2,0],A[2,1],A[2,2]
    axis = np.array([h-f, c-g, d-b])
    angle = np.arcsin(la.norm(axis) / 2)
    axis = axis / la.norm(axis)
    return (axis,angle)

def get_step_length_from_angle(angle):
    return 0.762*angle

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# Source: https://www.kite.com/python/answers/how-to-rotate-a-3d-vector-about-an-axis-in-python
def get_perpendicular(v):
    # Rotate v around [0,0,-1] by 90 degrees
    rot_radians = np.radians(90)
    rot_axis = np.array([0, 0, 1])

    rot_vector = rot_radians * rot_axis
    rotation = R.from_rotvec(rot_vector)
    rotated_v = rotation.apply(v)
    return rotated_v

def project_onto_hor(v):
    grav = np.array([0,0,-1])
    proj_onto_grav = (np.dot(v, grav)/(la.norm(grav))) * grav
    proj_onto_hor = v - proj_onto_grav
    proj_onto_hor = proj_onto_hor / la.norm(proj_onto_hor)
    return proj_onto_hor

def get_acc_data_global(acc_data, all_Rs):
    acc_data_global = []
    for i in range(len(all_Rs)):
        time = acc_data[i][0]
        current_R = all_Rs[i]
        current_acc = np.array(acc_data[i][1:])
        current_acc_g = current_R @ current_acc
        acc_data_global.append([time, current_acc_g[0], current_acc_g[1], current_acc_g[2]])
    return acc_data_global

def ts_to_index(timestamp):
    return int(timestamp/10)
def index_to_ts(index):
    return float(index*10)

def find_outliers(data):
    anomalies = []
    # Set upper and lower limit to 3 standard deviation
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * 3
    lower_limit  = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

def get_nth_max_val(n, data):
    outliers = find_outliers(data)
    # print(outliers)
    data_no_outliers = list(filter(lambda d: not(d in outliers), data))
    data_no_outliers.sort()
    nth_max = data_no_outliers[len(data_no_outliers) - n]
    print(nth_max)
    return nth_max

def get_upper_limit(data):
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * 3
    lower_limit  = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    return upper_limit

def find_step_indices(data):
    filtered_data = low_pass_filter(data)

    (step_indices_temp, props) = find_peaks(filtered_data)
    peaks_temp = [filtered_data[i] for i in step_indices_temp]
    fifth_max = get_nth_max_val(5, peaks_temp)

    upper_limit = get_upper_limit(filtered_data)

    print(0.6*fifth_max, upper_limit)

    (step_indices, props) = find_peaks(filtered_data,
                                       threshold=peak_threshold,
                                       prominence=peak_prominence,
                                       distance=peak_distance,
                                       height=[0.6*fifth_max, upper_limit])
    return (step_indices, filtered_data)

# Plotting all 4 routes side-by-side
plt.figure(figsize=(16,9))
plot_index = 221
for route in ['1', '2', '3', '4']:

    acc_filename = './data_imu_loc/route' + route + '/Accelerometer.csv'
    gyro_filename = './data_imu_loc/route' + route + '/Gyroscope.csv'

    # ---- Read in data ----
    acc_data = read_data(acc_filename)
    gyro_data = read_data(gyro_filename)

    # Trim down data so they are the same # samples
    min_length = min(len(acc_data), len(gyro_data))
    acc_data = acc_data[:min_length]
    gyro_data = gyro_data[:min_length]

    # Round times to nearest 0.01 s (regard as constant 100Hz)
    acc_data = [[truncate(row[0], 1), row[1], row[2], row[3]] for row in acc_data]
    gyro_data = [[truncate(row[0], 1), row[1], row[2], row[3]] for row in gyro_data]
    acc_times = [row[0] for row in acc_data]
    gyro_times = [row[0] for row in gyro_data]

    init_acc_data = acc_data[0][1:]
    R0 = get_init_orientation(init_acc_data)

    gyro_data_no_times = [row[1:] for row in gyro_data]
    all_Rs, all_delta_Rs = integrate_all_gyros(gyro_data_no_times, R0)

    acc_data_global = get_acc_data_global(acc_data, all_Rs)
    acc_data_global_zs = [row[3] for row in acc_data_global]
    # z_avg = np.mean(acc_data_global_zs)
    # acc_data_global_zs = [(z - z_avg) for z in acc_data_global_zs]

    # ---- Filter & count steps from acc_data----
    (step_indices, filtered_data) = find_step_indices(acc_data_global_zs)
    step_times = [acc_times[i] for i in step_indices]
    peaks = [filtered_data[i] for i in step_indices]

    # plt.figure(figsize=(16,9))
    # plt.plot(filtered_data, 'g')
    # plt.plot(step_indices, peaks, '.')
    # plt.title('Path ' + route)
    # plt.show()

    # ---- Track walking direction & next locations step-wise ----
    locs = []
    locs.append(np.array([0,0,0]))
    even = True
    for i in range(len(step_times) - 1):
        gyro_delta_Rs = get_gyro_delta_Rs_between_times(step_times[i], step_times[i+1], all_delta_Rs)
        delta_R_for_step = integrate_gyro_delta_Rs(gyro_delta_Rs)

        (axis,angle) = matrix_to_axis_angle(delta_R_for_step)
        axis = project_onto_hor(axis)
        axis = get_perpendicular(axis)

        if not(even):
            axis = np.array([-1*axis[0], -1*axis[1], axis[2]])

        walking_dir = axis
        step_length = 0.5
        disp_of_step = step_length * walking_dir
        locs.append(locs[i] + disp_of_step)

        even = not(even)

    # Plot all 4 routes side-by-side (as 4 subplots)

    locs = np.array(locs)
    plt.axis('equal')
    plt.subplot(plot_index)
    plt.plot(locs[:,0], locs[:,1])
    plt.title('Route ' + route)
    plot_index += 1

plt.show()
