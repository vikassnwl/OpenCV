import cv2
import numpy as np
import matplotlib.pyplot as plt
from more_itertools import sort_together


def num_channels(img):
    if len(img.shape) == 2:
        return 1
    return img.shape[-1]

def is_colored(img):
    return num_channels(img) == 3

def is_grayescaled(img):
    return num_channels(img) == 1

def is_transparent(img):
    return num_channels(img) == 4


def plott(img1, img2=None, figsize=(6.4, 4.8)):
    plt.figure(figsize=figsize)
    
    isColoredImg1 = is_colored(img1)
    isGrayscaledImg1 = is_grayescaled(img1)
    isTransparentImg1 = is_transparent(img1)

    if img2 is None:
        # plt.figure(figsize=(10, 10))
        if isColoredImg1:
            plt.imshow(img1[..., ::-1])
        elif isGrayscaledImg1:
            plt.imshow(img1, cmap='gray')
        elif isTransparentImg1:
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
    else:
        # plt.figure(figsize=(20, 20))
        isColoredImg2 = is_colored(img2)
        isGrayscaledImg2 = is_grayescaled(img2)
        isTransparentImg2 = is_transparent(img2)

        if isColoredImg1 and isColoredImg2:
            plt.subplot(121)
            plt.imshow(img1[..., ::-1])
            plt.subplot(122)
            plt.imshow(img2[..., ::-1])
        elif isGrayscaledImg1 and isGrayscaledImg2:
            plt.subplot(121)
            plt.imshow(img1, cmap="gray")
            plt.subplot(122)
            plt.imshow(img2, cmap="gray")
        elif isColoredImg1 and isGrayscaledImg2:
            plt.subplot(121)
            plt.imshow(img1[..., ::-1])
            plt.subplot(122)
            plt.imshow(img2, cmap="gray")
        elif isGrayscaledImg1 and isColoredImg2:
            plt.subplot(121)
            plt.imshow(img1, cmap="gray")
            plt.subplot(122)
            plt.imshow(img2[..., ::-1])
        elif isTransparentImg1 and isTransparentImg2:
            plt.subplot(121)
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
            plt.subplot(122)
            plt.imshow(np.concatenate((img2[..., 2::-1], img2[..., 3:]), axis=2))
        elif isColoredImg1 and isTransparentImg2:
            plt.subplot(121)
            plt.imshow(img1[..., ::-1])
            plt.subplot(122)
            plt.imshow(np.concatenate((img2[..., 2::-1], img2[..., 3:]), axis=2))
        elif isTransparentImg1 and isColoredImg2:
            plt.subplot(121)
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
            plt.subplot(122)
            plt.imshow(img2[..., ::-1])
        elif isTransparentImg1 and isGrayscaledImg2:
            plt.subplot(121)
            plt.imshow(np.concatenate((img1[..., 2::-1], img1[..., 3:]), axis=2))
            plt.subplot(122)
            plt.imshow(img2, cmap="gray")
        elif isGrayscaledImg1 and isTransparentImg2:
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


def findContours(img, lt=128, ut=255, inv=True, retr='external', chain_approx='simple'):
    if retr == 'external':
        retr_ = cv2.RETR_EXTERNAL
    elif retr == 'tree':
        retr_ = cv2.RETR_TREE
    elif retr == 'list':
        retr_ = cv2.RETR_LIST
    else:
        print('invalid value for argument "retr"')
    if chain_approx == 'simple':
        chain_approx_ = cv2.CHAIN_APPROX_SIMPLE
    elif chain_approx == 'none':
        chain_approx_ = cv2.CHAIN_APPROX_NONE
    else:
        print('invalid value for argument "chain_approx"')
    return cv2.findContours(thresh(img, lt, ut, inv), retr_, chain_approx_)


def drawContours(img, cntrs, idx=-1, thickness=2, color=(0, 255, 0)):
    if idx != -1:
        cntrs = [cntrs[idx]]
        idx = 0
    return cv2.drawContours(img.copy(), cntrs, idx, color, thickness)


def dilate(img, ksize=3, iters=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iters)


def erode(img, ksize=3, iters=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iters)


def morphologyEx(img, morph=0, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN if morph==1 else cv2.MORPH_CLOSE, kernel)


def sort_cntrs(cntrs, hierarchy=None, key='left_to_right'):
    if key == 'left_to_right':
        reverse = False
        key = lambda x: cv2.boundingRect(x)[0]
    elif key == 'right_to_left':
        reverse = True
        key = lambda x: cv2.boundingRect(x)[0]
    elif key == 'top_to_bottom':
        reverse = False
        key = lambda x: cv2.boundingRect(x)[1]
    elif key == 'bottom_to_top':
        reverse = True
        key = lambda x: cv2.boundingRect(x)[1]
    elif key == 'area_asc':
        reverse = False
        key = cv2.contourArea
    elif key == 'area_desc':
        reverse = True
        key = cv2.contourArea
    else:
        raise Exception('Invald option for key')

    if hierarchy is not None:
        cntrs, hierarchy, nums = sort_together([cntrs, hierarchy.reshape(-1, 4), range(len(cntrs))], key=key, reverse=reverse)
        d = {}
        for i, num in enumerate(nums):
            d[num] = i
        for i in range(len(hierarchy)):
            for j in range(4):
                if hierarchy[i][j] != -1:
                    hierarchy[i][j] = d[hierarchy[i][j]]
        hierarchy = np.array(hierarchy).reshape(1, -1, 4)
        return cntrs, hierarchy
    else: return sorted(cntrs, key=key)