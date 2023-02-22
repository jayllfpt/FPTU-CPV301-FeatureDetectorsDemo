import cv2
import numpy as np
from skimage.feature import hog
# import image
img = cv2.imread('husky.jpg')
# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# image ratio: 1:2
print(img.shape)
# gaussian blur
img = cv2.GaussianBlur(img, (3, 3), 0)
# apply hog
fd, img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, channel_axis= -1)
cv2.imshow('hog by skimage', img)
cv2.waitKey(0)