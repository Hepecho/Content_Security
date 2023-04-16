import numpy as np
import cv2 as cv

# Load an color image in grayscale
img = cv.imread('lena.jpg', 0)
rows, cols = img.shape

# Translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst1 = cv.warpAffine(img, M, (cols, rows))

# Rotation
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
dst2 = cv.warpAffine(img, M, (cols, rows))

# AffineTransform
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
dst3 = cv.warpAffine(img, M, (cols, rows))

cv.imshow('original', img)
cv.imshow('rotation', dst2)
cv.imshow('translation', dst1)
cv.imshow('Affine Transformation', dst3)

# 等待窗口
cv.waitKey(0)
cv.destroyAllWindows()
