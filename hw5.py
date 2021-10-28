import cv2
import numpy as np

def maskFindLocal(new_img, img, r, c, mask, func):

    local = 0
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (r + i >= 0 and r + i < img.shape[0] and c + j >= 0 and c + j < img.shape[1]):
                if(mask[2+i][2+j] == 1):
                    local = func(local, img[r+i][c+j])

    new_img[r][c] = local

def dilation(img, mask):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            maskFindLocal(new_img, img, row, col, mask, max)

    return new_img


def maskMatch(img, r, c, mask):

    for i in range(-2, 3):
        for j in range(-2, 3):
            if (r + i >= 0 and r + i < img.shape[0] and c + j >= 0 and c + j < img.shape[1]):
                if mask[2 + i][2 + j] == 1 and img[r + i][c + j] == 0:
                    return False

    return True

def erosion(img, mask):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if maskMatch(img, row, col, mask):
                new_img[row][col] = 1

    return new_img

def opening(img, mask):

    return dilation(erosion(img, mask), mask)

def closing(img, mask):

    return erosion(dilation(img, mask), mask)


def complement(img):

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            new_img[row][col] = 0 if img[row][col] == 1 else 1

    return new_img


def hitAndMiss(img, j, k):

    img_ero_j = erosion(img, j)
    imgc_ero_k = erosion(complement(img), k)

    new_img = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img_ero_j[i][j] == 1 and imgc_ero_k[i][j] == 1):
                new_img[i][j] = 1

    return new_img


if __name__ == '__main__':

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    mask = [[0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]]

    a = dilation(img, mask)
    cv2.imwrite('./output/hw5/a.bmp', a)
    '''
    b = scaleBackTo0_255(erosion(b_img, mask))
    cv2.imwrite('./output/hw4/b.bmp', b)
    
    c = scaleBackTo0_255(opening(b_img, mask))
    cv2.imwrite('./output/hw4/c.bmp', c)

    d = scaleBackTo0_255(closing(b_img, mask))
    cv2.imwrite('./output/hw4/d.bmp', d)
    '''
