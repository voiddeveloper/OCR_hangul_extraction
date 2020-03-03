import cv2 as cv
import numpy as np

standardFile = 'hwang/imgSet/test3.png'
imgStandard = cv.imread(standardFile, cv.IMREAD_COLOR)

imgCopySize = np.zeros_like(imgStandard)
height, width = imgStandard.shape[:2]

pixelInfo = [[0 for x in range(width)] for y in range(height)]

for i in range(0, height):
    for j in range(0, width):
        pixelInfo[i][j] = imgStandard[i, j]
        pixelInfo[i][j] = [int(pixelInfo[i][j][0] / 64) * 64 + 32, int(pixelInfo[i][j][1] / 64) * 64 + 32, int(pixelInfo[i][j][2] / 64) * 64 + 32]

for i in range(0, height):
    for j in range(0, width):
        pixel = (pixelInfo[i][j][0], pixelInfo[i][j][1], pixelInfo[i][j][2])
        cv.line(imgCopySize, (j,i), (j,i), pixel, 1)

# cv.imshow('original', imgStandard)
cv.imshow('lowGraphic', imgCopySize)
cv.waitKey(0)

