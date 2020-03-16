import cv2 as cv
import numpy as np

###############################################
# 특정 좌표의 색상값을 확인 해보기
###############################################

# img: 이미지 파일
# x: 색상값 확인할 x 좌표
# y: 색상값 확인할 y 좌표
def checkPixelColor(img, x, y):
    # 이미지의 높이, 너비
    height, width = img.shape[:2]

    # 픽셀값을 저장할 배열 생성
    pixelInfo = [[0 for x in range(width)] for y in range(height)]

    # 픽셀값을 pixelInfo에 저장
    for i in range(0, height):
        for j in range(0, width):
            pixelInfo[i][j] = img[i, j]
    
    # 원하는 좌표의 pixelInfo에 넣으면 해당 좌표의 색상값이 나온다.    
    return pixelInfo[x][y]

