import cv2 as cv
import numpy as np

###############################################
# 기준 색과 비교 색의 픽셀 값을 비교하는 코드
###############################################

# 이미지 불러오기
imgColor = cv.imread('hwang/imgSet/test_5.jpg', cv.IMREAD_COLOR)
resizeImgColor = cv.resize(imgColor, dsize=(10, 10), interpolation= cv.INTER_AREA)

# 불러온 이미지의 높이, 너비
height, width = imgColor.shape[:2]
# height, width = resizeImgColor.shape[:2]

print(f'height={height}', f'width={width}')

# 픽셀값을 저장할 배열 생성
pixelInfo = [[0 for x in range(width)] for y in range(height)]

# 픽셀값 저장
for i in range(0, height):
    for j in range(0, width):
        pixelInfo[i][j] = imgColor[i, j]
        # pixelInfo[i][j] = resizeImgColor[i, j]


print(pixelInfo[63][302])
print(pixelInfo[62][302])
print(pixelInfo[61][302])

# 배경색 설정
backgroundColor = pixelInfo[0][0]

# 배경색과 다른 픽셀값을 찾는다.
for i in range(0, height):
    for j in range(0, width):
        if (pixelInfo[i][j][0] != pixelInfo[0][0][0]) and (pixelInfo[i][j][1] != pixelInfo[0][0][1]) and (pixelInfo[i][j][2] != pixelInfo[0][0][2]):
            # 시작 좌표 저장
            tempX = i;
            tempY = j;
            
