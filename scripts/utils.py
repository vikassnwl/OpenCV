import cv2
import numpy as np
import matplotlib.pyplot as plt


def plott(img1, img2=None):
    if img2 is None:
        # plt.figure(figsize=(10, 10))
        if len(img1.shape) == 3:
            if img1.shape[-1] == 3:
                plt.imshow(img1[..., ::-1])
            else:
                plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
        else:
            plt.imshow(img1, cmap="gray")
    else:
        # plt.figure(figsize=(20, 20))
        if (
            len(img1.shape) == 3
            and len(img2.shape) == 3
            and img1.shape[-1] == 3
            and img2.shape[-1] == 3
        ):
            plt.subplot(121)
            plt.imshow(img1[..., ::-1])
            plt.subplot(122)
            plt.imshow(img2[..., ::-1])
        elif len(img1.shape) == 2 and len(img2.shape) == 2:
            plt.subplot(121)
            plt.imshow(img1, cmap="gray")
            plt.subplot(122)
            plt.imshow(img2, cmap="gray")
        elif len(img1.shape) == 3 and img1.shape[-1] == 3 and len(img2.shape) == 2:
            plt.subplot(121)
            plt.imshow(img1[..., ::-1])
            plt.subplot(122)
            plt.imshow(img2, cmap="gray")
        elif len(img1.shape) == 2 and len(img2.shape) == 3 and img2.shape[-1] == 3:
            plt.subplot(121)
            plt.imshow(img1, cmap="gray")
            plt.subplot(122)
            plt.imshow(img2[..., ::-1])
        elif (
            len(img1.shape) == 3
            and len(img2.shape) == 3
            and img1.shape[-1] == 4
            and img2.shape[-1] == 4
        ):
            plt.subplot(121)
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
            plt.subplot(122)
            plt.imshow(np.concatenate((img2[..., 2::-1], img2[..., 3:]), axis=2))
        elif (
            len(img1.shape) == 3
            and img1.shape[-1] == 3
            and len(img2.shape) == 3
            and img2.shape[-1] == 4
        ):
            plt.subplot(121)
            plt.imshow(img1[..., ::-1])
            plt.subplot(122)
            plt.imshow(np.concatenate((img2[..., 2::-1], img2[..., 3:]), axis=2))
        elif (
            len(img1.shape) == 3
            and img1.shape[-1] == 4
            and len(img2.shape) == 3
            and img2.shape[-1] == 3
        ):
            plt.subplot(121)
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
            plt.subplot(122)
            plt.imshow(img2[..., ::-1])
        elif len(img1.shape) == 3 and img1.shape[-1] == 4 and len(img2.shape) == 2:
            plt.subplot(121)
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
            plt.subplot(122)
            plt.imshow(img2, cmap="gray")
        elif len(img1.shape) == 2 and len(img2.shape) == 3 and img2.shape[-1] == 4:
            plt.subplot(121)
            plt.imshow(img1, cmap="gray")
            plt.subplot(122)
            plt.imshow(np.concatenate((img2[..., 2::-1], img2[..., 3:]), axis=2))


def resize(img, dim):
    h, w = img.shape[:2]
    W, H = dim
    if W == 0:
        aspect_ratio = H / h
        dim = (round(w * aspect_ratio), H)
    elif H == 0:
        aspect_ratio = W / w
        dim = (W, round(h * aspect_ratio))
    return cv2.resize(img, dim)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def blur(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))


def canny(img, lt=100, ut=200, apertureSize=3):
    return cv2.Canny(img, lt, ut, apertureSize=apertureSize)


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def thresh(img, lt=128, ut=255, inv=True):
    return cv2.threshold(gray(img), lt, ut, cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY)[1]


def findContours(img, lt=128, ut=255, inv=True):
    return cv2.findContours(thresh(img, lt, ut, inv), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def drawContours(img, cntrs, idx=-1, thickness=2, color=(0, 255, 0)):
    if idx != -1:
        cntrs = [cntrs[idx]]
    return cv2.drawContours(img.copy(), cntrs, idx, color, thickness)


def dilate(img, ksize=3, iters=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iters)


def erode(img, ksize=3, iters=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iters)


def morphologyEx(img, ksize=3, morph=0):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN if morph==1 else cv2.MORPH_CLOSE, kernel)