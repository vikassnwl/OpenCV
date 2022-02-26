import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 600, 250)
cv2.createTrackbar('Bmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Bmax', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Gmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Gmax', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Rmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Rmax', 'Trackbars', 255, 255, nothing)

while True:
    image = cv2.imread('inputs-outputs/inputs/sample1.jpeg')
    h, w = image.shape[:2]
    aspect = 800 / w
    dim = (800, int(h * aspect))
    image_800 = cv2.resize(image, dim)
    bmin = cv2.getTrackbarPos('Bmin', 'Trackbars')
    bmax = cv2.getTrackbarPos('Bmax', 'Trackbars')
    gmin = cv2.getTrackbarPos('Gmin', 'Trackbars')
    gmax = cv2.getTrackbarPos('Gmax', 'Trackbars')
    rmin = cv2.getTrackbarPos('Rmin', 'Trackbars')
    rmax = cv2.getTrackbarPos('Rmax', 'Trackbars')
    print(bmin, bmax, gmin, gmax, rmin, rmax)
    lower = np.array([bmin, gmin, rmin])
    upper = np.array([bmax, gmax, rmax])
    mask = cv2.inRange(image_800, lower, upper)
    final_result = cv2.bitwise_and(image_800, image_800, mask=mask)
    cv2.imshow('Final Output', final_result)
    cv2.imshow('Mask', mask)
    cv2.imshow('Output', image_800)
    cv2.waitKey(1)
