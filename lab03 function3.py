import numpy as np
import cv2 as cv2

# import image
img = cv2.imread('husky.jpg', 0)
# show image
cv2.imshow('original', img)
# detect edges using canny
edges = cv2.Canny(img, 100, 200)
# show edges
cv2.imshow('edge image', edges)
cv2.waitKey(0)

