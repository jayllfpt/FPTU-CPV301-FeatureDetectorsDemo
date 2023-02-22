import cv2
import numpy as np

# Read image
img = cv2.imread('house.jpg', cv2.IMREAD_COLOR)  # road.png is the filename
# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the edges in the image using canny detector
edges = cv2.Canny(gray, 50, 200)
# Detect points that form a line
max_slider = 250
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max_slider, None, 50, 10)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image", img)
cv2.imshow("Result Image Edges", edges)
cv2.waitKey(0)