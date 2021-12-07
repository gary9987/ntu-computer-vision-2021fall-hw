import math
import os
import cv2
import numpy as np


def Roberts(img, threshold=12):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_REPLICATE)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            r1 = -1 * eximg[r][c] + eximg[r + 1][c + 1]
            r2 = -1 * eximg[r][c + 1] + eximg[r + 1][c]
            gradient = math.sqrt(r1 ** 2 + r2 ** 2)
            if (gradient >= threshold):
                new_img[r][c] = 0
            else:
                new_img[r][c] = 255

    return new_img


def GetGradient(img, r, c, mask1, mask2):
    offset = int(len(mask1) / 2)
    p1 = 0
    p2 = 0
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            p1 += mask1[i + offset][j + offset] * img[r + i][c + j]
            p2 += mask2[i + offset][j + offset] * img[r + i][c + j]

    return math.sqrt(p1 ** 2 + p2 ** 2)


def Prewitt(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    m1 = [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]]
    m2 = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]

    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            gradient = GetGradient(eximg, r, c, m1, m2)
            if (gradient >= threshold):
                new_img[r - 1][c - 1] = 0
            else:
                new_img[r - 1][c - 1] = 255

    return new_img


def Soble(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    m1 = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    m2 = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            gradient = GetGradient(eximg, r, c, m1, m2)
            if (gradient >= threshold):
                new_img[r - 1][c - 1] = 0
            else:
                new_img[r - 1][c - 1] = 255

    return new_img


def FreiAndChen(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    m1 = [[-1, -math.sqrt(2), -1],
          [0, 0, 0],
          [1, math.sqrt(2), 1]]
    m2 = [[-1, 0, 1],
          [-math.sqrt(2), 0, math.sqrt(2)],
          [-1, 0, 1]]

    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            gradient = GetGradient(eximg, r, c, m1, m2)
            if (gradient >= threshold):
                new_img[r - 1][c - 1] = 0
            else:
                new_img[r - 1][c - 1] = 255

    return new_img


def GetMaxK(img, r, c):
    k = []
    k.append(-3 * (img[r - 1][c - 1] + img[r - 1][c] + img[r][c - 1] + img[r + 1][c - 1] + img[r + 1][c]) + 5 * (
                img[r - 1][c + 1] + img[r][c + 1] + img[r + 1][c + 1]))
    k.append(-3 * (img[r - 1][c - 1] + img[r + 1][c + 1] + img[r][c - 1] + img[r + 1][c - 1] + img[r + 1][c]) + 5 * (
            img[r - 1][c + 1] + img[r][c + 1] + img[r - 1][c]))
    k.append(-3 * (img[r][c + 1] + img[r + 1][c + 1] + img[r][c - 1] + img[r + 1][c - 1] + img[r + 1][c]) + 5 * (
            img[r - 1][c + 1] + img[r - 1][c - 1] + img[r - 1][c]))
    k.append(-3 * (img[r][c + 1] + img[r + 1][c + 1] + img[r - 1][c + 1] + img[r + 1][c - 1] + img[r + 1][c]) + 5 * (
            img[r][c - 1] + img[r - 1][c - 1] + img[r - 1][c]))
    k.append(-3 * (img[r][c + 1] + img[r + 1][c + 1] + img[r - 1][c + 1] + img[r - 1][c] + img[r + 1][c]) + 5 * (
            img[r][c - 1] + img[r - 1][c - 1] + img[r + 1][c - 1]))
    k.append(-3 * (img[r][c + 1] + img[r + 1][c + 1] + img[r - 1][c + 1] + img[r - 1][c] + img[r - 1][c - 1]) + 5 * (
            img[r][c - 1] + img[r + 1][c] + img[r + 1][c - 1]))
    k.append(-3 * (img[r][c + 1] + img[r][c - 1] + img[r - 1][c + 1] + img[r - 1][c] + img[r - 1][c - 1]) + 5 * (
            img[r + 1][c + 1] + img[r + 1][c] + img[r + 1][c - 1]))
    k.append(-3 * (img[r + 1][c - 1] + img[r][c - 1] + img[r - 1][c + 1] + img[r - 1][c] + img[r - 1][c - 1]) + 5 * (
            img[r + 1][c + 1] + img[r + 1][c] + img[r][c + 1]))

    return max(k)


def Kirsch(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            gradient = GetMaxK(eximg, r, c)
            if (gradient >= threshold):
                new_img[r - 1][c - 1] = 0
            else:
                new_img[r - 1][c - 1] = 255

    return new_img


def GetMaxR(img, r, c):
    masks = [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]],
             [[0, 1, 2],
              [-1, 0, 1],
              [-2, -1, 0]],
             [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]],
             [[2, 1, 0],
              [1, 0, -1],
              [0, -1, -2]],
             [[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]],
             [[0, -1, -2],
              [1, 0, -1],
              [2, 1, 0]],
             [[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]],
             [[-2, -1, 0],
              [-1, 0, 1],
              [0, 1, 2]]]

    maxn = 0
    offset = 1
    for mask in masks:
        tmp = 0
        for i in range(-offset, offset + 1):
            for j in range(-offset, offset + 1):
                tmp += mask[i + offset][j + offset] * img[r + i][c + j]
        maxn = max(maxn, tmp)

    return maxn

def Robinsin(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            gradient = GetMaxR(eximg, r, c)
            if (gradient >= threshold):
                new_img[r - 1][c - 1] = 0
            else:
                new_img[r - 1][c - 1] = 255

    return new_img


def GetMaxN(img, r, c):
    masks = [[[100, 100, 100, 100, 100],
              [100, 100, 100, 100, 100],
              [0, 0, 0, 0, 0],
              [-100, -100, -100, -100, -100],
              [-100, -100, -100, -100, -100]],
             [[100, 100, 100, 100, 100],
              [100, 100, 100, 87, -32],
              [100, 92, 0, -92, -100],
              [32, -78, -100, -100, -100],
              [-100, -100, -100, -100, -100]],
             [[100, 100, 100, 32, -100],
              [100, 100, 92, -78, -100],
              [100, 100, 0, -100, -100],
              [100, 78, -92, -100, -100],
              [100, -32, -100, -100, -100]],
             [[-100, -100, 0, 100, 100],
              [-100, -100, 0, 100, 100],
              [-100, -100, 0, 100, 100],
              [-100, -100, 0, 100, 100],
              [-100, -100, 0, 100, 100]],
             [[-100, 32, 100, 100, 100],
              [-100, -78, 92, 100, 100],
              [-100, -100, 0, 100, 100],
              [-100, -100, -92, 78, 100],
              [-100, -100, -100, -32, 100]],
             [[100, 100, 100, 100, 100],
              [-32, 78, 100, 100, 100],
              [-100, -92, 0, 92, 100],
              [-100, -100, -100, -78, 32],
              [-100, -100, -100, -100, -100]]]
    maxn = 0
    offset = 2
    for mask in masks:
        tmp = 0
        for i in range(-offset, offset + 1):
            for j in range(-offset, offset + 1):
                tmp += mask[i + offset][j + offset] * img[r + i][c + j]
        maxn = max(maxn, tmp)

    return maxn

def Nevatia(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    for r in range(2, img.shape[0] + 2):
        for c in range(2, img.shape[1] + 2):
            gradient = GetMaxN(eximg, r, c)
            if (gradient >= threshold):
                new_img[r - 2][c - 2] = 0
            else:
                new_img[r - 2][c - 2] = 255

    return new_img

def apply_mask(img, r, c, mask):
    tmp = 0
    offset = 1
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            tmp += mask[i + offset][j + offset] * img[r + i][c + j]
    return tmp

def correlation(img, mask, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            if(apply_mask(eximg, r, c, mask) >= threshold):
                new_img[r-1][c-1] = 1
            elif(apply_mask(eximg, r, c, mask) > -threshold):
                new_img[r-1][c-1] = 0
            else:
                new_img[r-1][c-1] = -1

    return new_img


def zero_crossing(img):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    offset = 1
    for r in range(1, img.shape[0] + 1):
        for c in range(1, img.shape[1] + 1):
            is_cross = False
            if(eximg[r][c] == 1):
                for i in range(-offset, offset + 1):
                    for j in range(-offset, offset + 1):
                        if(i != 0 and j != 0):
                            if(eximg[r+i][c+j] == -1):
                                is_cross = True
                                break
                    if is_cross:
                        break

            new_img[r-1][c-1] = 0 if is_cross else 255
    return new_img


if __name__ == '__main__':

    output_file_path = './output/hw10/'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    m1 = [[0, 1, 0],
          [1, -4, 1],
          [0, 1, 0]]
    laplacian1 =  zero_crossing(correlation(img, m1, 15))
    cv2.imwrite(output_file_path + 'laplacian1' + '.bmp', laplacian1)

    m2 = [[1/3, 1/3, 1/3],
          [1/3, -8/3, 1/3],
          [1/3, 1/3, 1/3]]
    laplacian2 = zero_crossing(correlation(img, m2, 15))
    cv2.imwrite(output_file_path + 'laplacian2' + '.bmp', laplacian2)