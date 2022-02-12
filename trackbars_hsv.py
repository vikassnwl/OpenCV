import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 600, 250)
cv2.createTrackbar('Huemin', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('Huemax', 'Trackbars', 180, 180, nothing)
cv2.createTrackbar('Satmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Satmax', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Valmin', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Valmax', 'Trackbars', 255, 255, nothing)

while True:
    image = cv2.imread('blur/sample14.jpg')
    h, w = image.shape[:2]
    aspect = 800 / w
    dim = (800, int(h * aspect))
    image_800 = cv2.resize(image, dim)
    hmin = cv2.getTrackbarPos('Huemin', 'Trackbars')
    hmax = cv2.getTrackbarPos('Huemax', 'Trackbars')
    smin = cv2.getTrackbarPos('Satmin', 'Trackbars')
    smax = cv2.getTrackbarPos('Satmax', 'Trackbars')
    vmin = cv2.getTrackbarPos('Valmin', 'Trackbars')
    vmax = cv2.getTrackbarPos('Valmax', 'Trackbars')
    print(hmin, hmax, smin, smax, vmin, vmax)
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    mask = cv2.inRange(cv2.cvtColor(image_800, cv2.COLOR_BGR2HSV), lower,
                       upper)
    final_result = cv2.bitwise_and(image_800, image_800, mask=mask)
    cv2.imshow('Final Output', final_result)
    cv2.imshow('Mask', mask)
    cv2.imshow('Output', image_800)
    cv2.waitKey(1)
