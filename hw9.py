import math
import os
import random
from hw5 import opening, closing
import cv2
import numpy as np


def Roberts(img, threshold=12):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_REPLICATE)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            r1 = -1 * eximg[r][c] + eximg[r+1][c+1]
            r2 = -1 * eximg[r][c+1] + eximg[r+1][c]
            gradient = math.sqrt(r1**2 + r2**2)
            if(gradient >= threshold):
                new_img[r][c] = 0
            else:
                new_img[r][c] = 255

    return new_img


def GetGradient(img, r, c, mask1, mask2):
    offset = int(len(mask1)/2)
    p1 = 0
    p2 = 0
    for i in range(-offset, offset+1):
        for j in range(-offset, offset+1):
            p1 += mask1[i+offset][j+offset] * img[r+i][c+j]
            p2 += mask2[i+offset][j+offset] * img[r+i][c+j]

    return math.sqrt(p1**2 + p2**2)


def Prewitt(img, threshold):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    eximg = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    m1 = [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]]
    m2 = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]

    for r in range(1, img.shape[0]+1):
        for c in range(1, img.shape[1]+1):
            gradient = GetGradient(eximg, r, c, m1, m2)
            if (gradient >= threshold):
                new_img[r-1][c-1] = 0
            else:
                new_img[r-1][c-1] = 255

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


if __name__ == '__main__':

    output_file_path = './output/hw9/'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    '''
    roberts = Roberts(img, 12)
    cv2.imwrite(output_file_path + 'roberts' + '.bmp', roberts)

    prewitt = Prewitt(img, 24)
    cv2.imwrite(output_file_path + 'prewitt' + '.bmp', prewitt)

    sobel = Soble(img, 38)
    cv2.imwrite(output_file_path + 'sobel' + '.bmp', sobel)
    
    frei_and_chen = FreiAndChen(img, 30)
    cv2.imwrite(output_file_path + 'frei_and_chen' + '.bmp', frei_and_chen)
    '''
    