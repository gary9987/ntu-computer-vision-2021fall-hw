import cv2
import numpy as np

def maskFindLocal(new_img, img, r, c, mask, func, local_init):

    for i in range(-2, 3):
        for j in range(-2, 3):
            if (r + i >= 0 and r + i < img.shape[0] and c + j >= 0 and c + j < img.shape[1]):
                if(mask[2+i][2+j] == 1):
                    local_init = func(local_init, img[r+i][c+j])

    new_img[r][c] = local_init

def dilation(img, mask):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            maskFindLocal(new_img, img, row, col, mask, max, 0)

    return new_img


def erosion(img, mask):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            maskFindLocal(new_img, img, row, col, mask, min, 255)

    return new_img

def opening(img, mask):

    return dilation(erosion(img, mask), mask)

def closing(img, mask):

    return erosion(dilation(img, mask), mask)


if __name__ == '__main__':

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    mask = [[0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]]

    a = dilation(img, mask)
    cv2.imwrite('./output/hw5/a.bmp', a)

    b = erosion(img, mask)
    cv2.imwrite('./output/hw5/b.bmp', b)
    
    c = opening(img, mask)
    cv2.imwrite('./output/hw5/c.bmp', c)

    d = closing(img, mask)
    cv2.imwrite('./output/hw5/d.bmp', d)

