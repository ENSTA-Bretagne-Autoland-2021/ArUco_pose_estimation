import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from scipy import signal


def filter(X):
    b, a = signal.butter(3, 0.05)
    return signal.filtfilt(b, a, X)

X = []

if __name__ == "__main__":

    cap = cv2.VideoCapture('rotation.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open('matrix_calibration.npy', 'rb') as f:
        ret = np.load(f)
        matrix_coefficients = np.load(f)
        distortion_coefficients = np.load(f)
        rvecs = np.load(f)
        tvecs = np.load(f)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # Specify marker size as 4x4, 5x5, 6x6
    parameters = aruco.DetectorParameters_create()  # Marker detection parameters

    current_time = 0.0

    while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        try:    
            if np.all(ids is not None):  # If there are markers found by detector
                for i in range(0, len(ids)):  # Iterate in markers
                    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
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
    x, y, z = filter(X[:, 4]), filter(X[:, 5]), filter(X[:, 6])
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
    phi, theta, psi = filter(np.abs(X[:, 1])), filter(np.abs(X[:, 2]+np.pi/2)), filter(np.abs(X[:, 3]))
    plt.plot(X[:, 0], phi, color="crimson", label=r"$\phi$")
    plt.plot(X[:, 0], theta, color="teal", label=r"$\theta$")
    plt.plot(X[:, 0], psi, color="purple", label=r"$\psi$")
    plt.title("Orientation du marqueur")
    plt.xlabel(r"Temps (en $s$)")
    plt.ylabel(r"Orientation (en $rad$)")
    plt.grid(True)
    plt.legend(loc="best")

    plt.show()