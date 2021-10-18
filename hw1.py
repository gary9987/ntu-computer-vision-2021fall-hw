import copy
import cv2

def upside_down(img):
    retimg = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            retimg[i][j] = img[img.shape[0] - 1 - i][j]
    return retimg

def right_side_left(img):
    retimg = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            retimg[i][j] = img[i][img.shape[1] - 1 - j]
    return retimg

def diagonally_flip(img):
    retimg = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            retimg[i][j] = img[j][i]
    return retimg

def binarilize(img):
    retimg = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] >= 128):
                retimg[i][j] = 255
            else:
                retimg[i][j] = 0

    return retimg

if __name__ == '__main__':

    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    upside_dowm_img = upside_down(img)
    cv2.imwrite('a.bmp', upside_dowm_img)

    horizon_flip_img = right_side_left(img)
    cv2.imwrite('b.bmp', horizon_flip_img)

    diagonally_flip_img = diagonally_flip(img)
    cv2.imwrite('c.bmp', diagonally_flip_img)

    binary = binarilize(img)
    cv2.imwrite('f.bmp', binary)
