import math
import os
import random
from hw5 import opening, closing
import cv2
import numpy as np


def getSNR(original_img, n_img):
    original_img = (lambda x: x / 255)(original_img)
    n_img = (lambda x: x / 255)(n_img)
    mean = np.mean(original_img)
    mean_n = np.mean(n_img - original_img)
    vs = np.mean((lambda x: (x - mean) ** 2)(original_img))
    vn = np.mean((lambda x: (x - mean_n) ** 2)(n_img - original_img))
    return 20 * math.log10(math.sqrt(vs) / math.sqrt(vn))


def getGaussianNoise(img, amplitude):
    new_img = img.copy()
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            noise = int(img[i][j]) + amplitude * random.gauss(0, 1)
            if(noise > 255):
                noise = 255
            new_img[i][j] = noise
    return new_img


def getSaltAndPepper(img, threshold):
    new_img = img.copy()
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            random_value = random.uniform(0, 1)
            if(random_value <= threshold):
                new_img[i][j] = 0
            elif(random_value >= 1 - threshold):
                new_img[i][j] = 255
    return new_img


def box(img, r, c, kernel_size):
    ret = 0
    for i in range(-int(kernel_size/2), int(kernel_size/2)+1):
        for j in range(-int(kernel_size / 2), int(kernel_size / 2)+1):
            ret += img[r+i][c+j]
    return ret / kernel_size**2


def applyBoxFilter(img, kernel_size):
    offset = int(kernel_size/2)
    new_img = np.zeros((img.shape[0]-2*offset, img.shape[1]-2*offset))

    for i in range(offset, img.shape[0] - offset):
        for j in range(offset, img.shape[1] - offset):
            new_img[i-offset][j-offset] = box(img, i, j, kernel_size)

    return new_img


def median(img, r, c, kernel_size):
    ret = []
    for i in range(-int(kernel_size / 2), int(kernel_size / 2) + 1):
        for j in range(-int(kernel_size / 2), int(kernel_size / 2) + 1):
            ret.append(img[r + i][c + j])
    ret.sort()
    return ret[int(kernel_size**2 / 2)]


def applyMedianFilter(img, kernel_size):
    offset = int(kernel_size/2)
    new_img = np.zeros((img.shape[0]-2*offset, img.shape[1]-2*offset))

    for i in range(offset, img.shape[0] - offset):
        for j in range(offset, img.shape[1] - offset):
            new_img[i-offset][j-offset] = median(img, i, j, kernel_size)

    return new_img

if __name__ == '__main__':

    output_file_path = './output/hw8/'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    # ================================= Gaussian =====================================
    gaussian10 = getGaussianNoise(img, 10)
    cv2.imwrite(output_file_path + 'gaussian10' + '.bmp', gaussian10)

    gaussian30 = getGaussianNoise(img, 30)
    cv2.imwrite(output_file_path + 'gaussian30' + '.bmp', gaussian30)

    # ================================= Salt and Pepper =====================================
    salt_and_pepper005 = getSaltAndPepper(img, 0.05)
    cv2.imwrite(output_file_path + 'salt_and_pepper005' + '.bmp', salt_and_pepper005)

    salt_and_pepper010 = getSaltAndPepper(img, 0.10)
    cv2.imwrite(output_file_path + 'salt_and_pepper010' + '.bmp', salt_and_pepper010)

    # ================================= extend img =====================================
    extend_gaussian10_1_img = cv2.copyMakeBorder(gaussian10, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    extend_gaussian10_2_img = cv2.copyMakeBorder(gaussian10, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    extend_gaussian30_1_img = cv2.copyMakeBorder(gaussian30, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    extend_gaussian30_2_img = cv2.copyMakeBorder(gaussian30, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    extend_salt005_1_img = cv2.copyMakeBorder(salt_and_pepper005, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    extend_salt005_2_img = cv2.copyMakeBorder(salt_and_pepper005, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    extend_salt010_1_img = cv2.copyMakeBorder(salt_and_pepper010, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    extend_salt010_2_img = cv2.copyMakeBorder(salt_and_pepper010, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    # ================================= Box Filter =====================================
    box3_gaussian10_img = applyBoxFilter(extend_gaussian10_1_img, 3)
    cv2.imwrite(output_file_path + 'box3_gaussian10_img' + '.bmp', box3_gaussian10_img)

    box5_gaussian10_img = applyBoxFilter(extend_gaussian10_2_img, 5)
    cv2.imwrite(output_file_path + 'box5_gaussian10_img' + '.bmp', box5_gaussian10_img)

    box3_gaussian30_img = applyBoxFilter(extend_gaussian30_1_img, 3)
    cv2.imwrite(output_file_path + 'box3_gaussian30_img' + '.bmp', box3_gaussian30_img)

    box5_gaussian30_img = applyBoxFilter(extend_gaussian30_2_img, 5)
    cv2.imwrite(output_file_path + 'box5_gaussian30_img' + '.bmp', box5_gaussian30_img)

    box3_salt005_img = applyBoxFilter(extend_salt005_1_img, 3)
    cv2.imwrite(output_file_path + 'box3_salt005_img' + '.bmp', box3_salt005_img)

    box5_salt005_img = applyBoxFilter(extend_salt005_2_img, 5)
    cv2.imwrite(output_file_path + 'box5_salt005_img' + '.bmp', box5_salt005_img)

    box3_salt010_img = applyBoxFilter(extend_salt010_1_img, 3)
    cv2.imwrite(output_file_path + 'box3_salt010_img' + '.bmp', box3_salt010_img)

    box5_salt010_img = applyBoxFilter(extend_salt010_2_img, 5)
    cv2.imwrite(output_file_path + 'box5_salt010_img' + '.bmp', box5_salt010_img)

    # ================================= Median Filter =====================================
    m3_gaussian10_img = applyMedianFilter(extend_gaussian10_1_img, 3)
    cv2.imwrite(output_file_path + 'm3_gaussian10_img' + '.bmp', m3_gaussian10_img)

    m5_gaussian10_img = applyMedianFilter(extend_gaussian10_2_img, 5)
    cv2.imwrite(output_file_path + 'm5_gaussian10_img' + '.bmp', m5_gaussian10_img)

    m3_gaussian30_img = applyMedianFilter(extend_gaussian30_1_img, 3)
    cv2.imwrite(output_file_path + 'm3_gaussian30_img' + '.bmp', m3_gaussian30_img)

    m5_gaussian30_img = applyMedianFilter(extend_gaussian30_2_img, 5)
    cv2.imwrite(output_file_path + 'm5_gaussian30_img' + '.bmp', m5_gaussian30_img)

    m3_salt005_img = applyMedianFilter(extend_salt005_1_img, 3)
    cv2.imwrite(output_file_path + 'm3_salt005_img' + '.bmp', m3_salt005_img)

    m5_salt005_img = applyMedianFilter(extend_salt005_2_img, 5)
    cv2.imwrite(output_file_path + 'm5_salt005_img' + '.bmp', m5_salt005_img)

    m3_salt010_img = applyMedianFilter(extend_salt010_1_img, 3)
    cv2.imwrite(output_file_path + 'm3_salt010_img' + '.bmp', m3_salt010_img)

    m5_salt010_img = applyMedianFilter(extend_salt010_2_img, 5)
    cv2.imwrite(output_file_path + 'm5_salt010_img' + '.bmp', m5_salt010_img)

    # ================================= Opening then Closing =====================================
    mask = [[0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]]

    o_c_gaussian10 = closing(opening(gaussian10, mask), mask)
    cv2.imwrite(output_file_path + 'o_c_gaussian10' + '.bmp', o_c_gaussian10)

    o_c_gaussian30 = closing(opening(gaussian30, mask), mask)
    cv2.imwrite(output_file_path + 'o_c_gaussian30' + '.bmp', o_c_gaussian30)

    o_c_salt005 = closing(opening(salt_and_pepper005, mask), mask)
    cv2.imwrite(output_file_path + 'o_c_salt005' + '.bmp', o_c_salt005)

    o_c_salt010 = closing(opening(salt_and_pepper010, mask), mask)
    cv2.imwrite(output_file_path + 'o_c_salt010' + '.bmp', o_c_salt010)

    # ================================= Closing then Opening =====================================
    c_o_gaussian10 = opening(closing(gaussian10, mask), mask)
    cv2.imwrite(output_file_path + 'c_o_gaussian10' + '.bmp', c_o_gaussian10)

    c_o_gaussian30 = opening(closing(gaussian30, mask), mask)
    cv2.imwrite(output_file_path + 'c_o_gaussian30' + '.bmp', c_o_gaussian30)

    c_o_salt005 = opening(closing(salt_and_pepper005, mask), mask)
    cv2.imwrite(output_file_path + 'c_o_salt005' + '.bmp', c_o_salt005)

    c_o_salt010 = opening(closing(salt_and_pepper010, mask), mask)
    cv2.imwrite(output_file_path + 'c_o_salt010' + '.bmp', c_o_salt010)


    for filename in os.listdir(output_file_path):
        noise_img = cv2.imread(output_file_path+filename, cv2.IMREAD_GRAYSCALE)
        snr = getSNR(img, noise_img)
        print(filename, snr)
        