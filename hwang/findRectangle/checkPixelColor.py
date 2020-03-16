import cv2 as cv
import numpy as np

###############################################
# 특정 좌표의 색상값을 확인 해보기
###############################################

# 이미지 불러오기
imgColor = cv.imread('hwang/imgSet/findRectangle/real_4.jpg', cv.IMREAD_COLOR)

# 불러온 이미지의 높이, 너비
height, width = imgColor.shape[:2]
# print(f'height={height}', f'width={width}')

# 픽셀값을 저장할 배열 생성
pixelInfo = [[0 for x in range(width)] for y in range(height)]

# 픽셀값을 pixelInfo에 저장
for i in range(0, height):
    for j in range(0, width):
        pixelInfo[i][j] = imgColor[i, j]

# 원하는 좌표의 pixelInfo에 넣으면 해당 좌표의 색상값이 나온다.
# print(pixelInfo[63][302])
