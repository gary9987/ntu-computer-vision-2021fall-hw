from hw2 import calculateDestribution
import cv2
import matplotlib.pyplot as plt
import numpy as np


def devideBy3(img):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = int(img[i][j] / 3)

    return new_img


def histEqualization(img):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    hist = calculateDestribution(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = 255*sum(hist[:int(img[i][j])]) / (img.shape[0] * img.shape[1])

    return new_img


if __name__ == '__main__':

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    cv2.imwrite('./output/hw3/a.bmp', img)
    a_hist = calculateDestribution(img)
    plt.bar(list(range(0, 256)), a_hist)
    plt.savefig('./output/hw3/a_hist.png')
    plt.close()

    b = devideBy3(img)
    cv2.imwrite('./output/hw3/b.bmp', b)
    b_hist = calculateDestribution(b)
    plt.bar(list(range(0, 256)), b_hist)
    plt.savefig('./output/hw3/b_hist.png')
    plt.close()

    c = histEqualization(b)
    cv2.imwrite('./output/hw3/c.bmp', c)
    c_hist = calculateDestribution(c)
    plt.bar(list(range(0, 256)), c_hist)
    plt.savefig('./output/hw3/c_hist.png')

