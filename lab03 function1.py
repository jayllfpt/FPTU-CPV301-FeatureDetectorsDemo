import cv2
import numpy as np


# harris corner
def find_max(R, i, j):
    max_point = 0

    for k in [-1, 0, 1]:
        for l in [-1, 0, 1]:
            if max_point < R[i + k][j + l]:
                max_point = R[i + k][j + l]
    return max_point


def matrix_multiply(I_x, I_y):
    (h, w) = I_x.shape
    result = np.empty((h, w))

    for i in range(0, h):
        for j in range(0, w):
            result[i][j] = I_x[i][j] * I_y[i][j]

    return result


def nonmax_suppression(R, i, j, max_point):
    for k in [-1, 0, 1]:
        for l in [-1, 0, 1]:
            if R[i + k][j + l] == max_point:
                continue
            else:
                R[i + k][j + l] = 0


def calculate_trace(A, B):
    return A + B


def calculate_det(A, B, C):
    det = matrix_multiply(A, B) - matrix_multiply(C, C)
    return det

# implement harris corner detect
# read img
img1 = cv2.imread('husky.jpg')
# convert to gray
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gaussian blur
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 2)
# dung sobel dao ham
I_X = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=5)
I_Y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=5)
# tuong quan giua cac dao ham
A = matrix_multiply(I_X, I_X)
B = matrix_multiply(I_X, I_Y)
C = matrix_multiply(I_Y, I_Y)
# ap dung gaussian blur len A B C
_A = cv2.GaussianBlur(A, (5, 5), 2)
_B = cv2.GaussianBlur(B, (5, 5), 2)
_C = cv2.GaussianBlur(C, (5, 5), 2)
# tinh corner response
K = 0.04
R = calculate_det(_A, _B, _C) - K * calculate_trace(_A, _B)
# threshold & lay max
_, R = cv2.threshold(R, R.max() // 500, R.max(), cv2.THRESH_BINARY)
h, w = R.shape
for i in range(1, h - 1, 2):
    for j in range(1, w - 1, 2):
        maxPixel = find_max(R, i, j)
        nonmax_suppression(R, i, j, maxPixel)

R_dst = cv2.dilate(R, None)
img1[R_dst > 0.001 * R_dst.max()] = (0, 255, 0)
cv2.imshow('implement', img1)

# using cornerHarris of cv2
img = cv2.imread('husky.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
cv2.waitKey(0)
