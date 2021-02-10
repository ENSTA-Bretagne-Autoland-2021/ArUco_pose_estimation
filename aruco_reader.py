import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from scipy import signal


def filter(X, n, f):
    b, a = signal.butter(n, f)
    return signal.filtfilt(b, a, X)

# def reject_outliers(time, data, m=2):
#     idx = abs(data - np.mean(data)) < m * np.std(data)
#     return time[idx], data[idx]

# def reject_outliers(time, data, m=2):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/mdev if mdev else 0.
#     data_range = np.arange(len(data))
#     idx_list = data_range[s<m]
#     return time[idx_list], data[s<m]

def reject_outliers(time, data, m=2):
    y = signal.medfilt(data, kernel_size=9)
    d = np.abs(data - y)
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    data_range = np.arange(len(data))
    idx_list = data_range[s<m]
    return time[idx_list], data[s<m]

X = []

if __name__ == "__main__":

    cap = cv2.VideoCapture('zoom_in.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open('matrix_calibration.npy', 'rb') as f:
        ret = np.load(f)
        matrix_coefficients = np.load(f)
        distortion_coefficients = np.load(f)
        rvecs = np.load(f)
        tvecs = np.load(f)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Specify marker size as 4x4, 5x5, 6x6
    parameters = aruco.DetectorParameters_create()  # Marker detection parameters
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    current_time = 0.0

    while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        try:    
            if np.all(ids is not None):  # If there are markers found by detector
                for i in range(0, len(ids)):  # Iterate in markers
                    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.2, matrix_coefficients,
                                                                            distortion_coefficients)
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    aruco.drawDetectedMarkers(frame, corners, borderColor=[255, 200, 0])  # Draw A square around the markers
                    aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw axis

                    c_x = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4 # X coordinate of marker's center
                    c_y = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4 # Y coordinate of marker's center
                    cv2.putText(frame, "id"+str(ids[i, 0]), (int(c_x), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,225,250), 2)

                    if ids[i] == 1:
                        X.append(np.hstack((np.array([current_time]), rvec[0,0], tvec[0,0])))

        except:
            if ids is None or len(ids) == 0:
                print("******************************************************")
                print("*************** Marker Detection Failed **************")
                print("******************************************************")

        cv2.imshow('frame', frame)
        current_time += 1/fps

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    X = np.asarray(X)

    # Position
    plt.figure()
    x, y, z = filter(X[:, 4], 3, 0.05), filter(X[:, 5], 3, 0.05), filter(X[:, 6], 3, 0.05)
    plt.plot(X[:, 0], x, color="red", label="X")
    plt.plot(X[:, 0], y, color="green", label="Y")
    plt.plot(X[:, 0], z, color="blue", label="Z")
    plt.title("Position du marqueur")
    plt.xlabel(r"Temps (en $s$)")
    plt.ylabel(r"Position (en $m$)")
    plt.grid(True)
    plt.legend(loc="best")

    # Orientation
    plt.figure()
    t_phi, phi = reject_outliers(X[:, 0], X[:, 1], m=2)
    t_theta, theta = reject_outliers(X[:, 0], X[:, 2], m=2)
    t_psi, psi = reject_outliers(X[:, 0], X[:, 3], m=2)
    plt.plot(t_phi, filter(phi, 3, 0.005), color="crimson", label=r"$\phi$")
    plt.plot(t_theta, filter(theta, 3, 0.005), color="teal", label=r"$\theta$")
    plt.plot(t_psi, filter(psi, 3, 0.005), color="purple", label=r"$\psi$")
    plt.title("Orientation du marqueur")
    plt.xlabel(r"Temps (en $s$)")
    plt.ylabel(r"Orientation (en $rad$)")
    plt.grid(True)
    plt.legend(loc="best")

    plt.show()