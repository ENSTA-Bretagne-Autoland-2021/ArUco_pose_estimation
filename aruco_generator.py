import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

fig = plt.figure()
nx = 4
ny = 3
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")

plt.savefig("data/markers.pdf")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
img = aruco.drawMarker(aruco_dict,1, 10000)
plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
plt.savefig("data/marker.pdf")
plt.show()