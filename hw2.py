import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np

def binarilize(img):
    retimg = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 128:
                retimg[i][j] = 255
            else:
                retimg[i][j] = 0

    return retimg


def calculateDestribution(img):

    hist_gray = [float(0)] * 256

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            value = int(img[row][col])
            hist_gray[value] += 1

    return hist_gray


def min_nieghbors(labeled_img, row, col):
    mini = labeled_img[row][col]
    ind = [[-1, 0], [0, 1], [1, 0], [0, -1]]

    for i in range(4):
        if (row + ind[i][0] >= 0 and row + ind[i][0] < labeled_img.shape[0]):
            if (col + ind[i][1] >= 0 and col + ind[i][1] < labeled_img.shape[1]):
                if (labeled_img[row + ind[i][0]][col + ind[i][1]] != 0):
                    mini = min(mini, labeled_img[row + ind[i][0]][col + ind[i][1]])

    return mini


def top_down_pass(labeled_img):
    change = False

    for i in range(labeled_img.shape[0]):
        for j in range(labeled_img.shape[1]):
            if labeled_img[i][j] != 0:
                min_label = min_nieghbors(labeled_img, i, j)
                if min_label != labeled_img[i][j]:
                    change = True
                    labeled_img[i][j] = min_label

    return change


def buttom_up_pass(labeled_img):
    change = False

    for i in range(labeled_img.shape[0] - 1, -1, -1):
        for j in range(labeled_img.shape[1] - 1, -1, -1):
            if labeled_img[i][j] != 0:
                min_label = min_nieghbors(labeled_img, i, j)
                if min_label != labeled_img[i][j]:
                    change = True
                    labeled_img[i][j] = min_label

    return change


def connectedComponents(bin_img):
    label = 1
    labeled_img = np.zeros((bin_img.shape[0], bin_img.shape[1]))

    # init each  1-pixel a unique label
    for i in range(labeled_img.shape[0]):
        for j in range(labeled_img.shape[1]):
            if bin_img[i][j] != 0:
                labeled_img[i][j] = label
                label += 1

    flag1 = True
    flag2 = True
    while(flag1 or flag2):
        flag1 = top_down_pass(labeled_img)
        flag2 = buttom_up_pass(labeled_img)

    return labeled_img


def drawBoundingBox(labeled, origin):

    dict = {}

    ret = copy.deepcopy(origin)
    for i in range(labeled.shape[0]):
        for j in range(labeled.shape[1]):
            if labeled[i][j] != 0:
                if labeled[i][j] not in dict:
                    dict[labeled[i][j]] = [i, j, i, j, 1, i, j]  #  up right down left size row_sum col_sum
                else:
                    dict[labeled[i][j]][0] = min(dict[labeled[i][j]][0], i)
                    dict[labeled[i][j]][1] = max(dict[labeled[i][j]][1], j)
                    dict[labeled[i][j]][2] = max(dict[labeled[i][j]][2], i)
                    dict[labeled[i][j]][3] = min(dict[labeled[i][j]][3], j)
                    dict[labeled[i][j]][4] += 1
                    dict[labeled[i][j]][5] += i
                    dict[labeled[i][j]][6] += j

    filter_list = []
    for i in dict:
        if dict[i][4] >= 500:  # The component that size >= 500 pixels
            filter_list.append(dict[i])

    ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)  # transform to rgb graph
    red = (0, 0, 255)  # red
    blue = (255, 0, 0)  # blue
    thickness = 2  # 寬度 (-1 表示填滿)
    for i in filter_list:
        cv2.rectangle(ret, (i[3], i[0]), (i[1], i[2]), red, thickness)  # left-down, right-up
        cv2.circle(ret, (int(i[6]/i[4]), int(i[5]/i[4])), 5, blue, -1)

    return ret



if __name__ == '__main__':
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    binary = binarilize(img)
    cv2.imwrite('a.bmp', binary)

    destribution = calculateDestribution(img)
    plt.bar(list(range(0, 256)), destribution)
    # plt.show()
    plt.savefig('b.png')

    conponent_img = connectedComponents(binary)

    bounding_img = drawBoundingBox(conponent_img, binary)
    cv2.imwrite('c.bmp', bounding_img)