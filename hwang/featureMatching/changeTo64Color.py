import cv2 as cv
import numpy as np

##################################################################################
# 이미지를 낮은 색상으로 바꾸는 코드
# 트루컬러(256*256*256) => 64색(4*4*4)으로 변환한다.
#
# 이미지를 인식하는 과정에서, 화질이 나쁜 이미지들은 눈으로 볼때와 실제 픽셀값의 차이가 많다.
# 픽셀을 확대해서 보면, 주변 테두리의 픽셀값이 깨져있거나, 비슷하지만 다른 색들이 매우 많이 있다.
# 유사한 색을 하나로 묶어보기 위해, 색의 자세한 표현력을 줄여보고자 했다.
#
# ## 결과 ##
# 색을 다운그레이드 하는것은 별 문제가 없지만, 이걸 통해 얻는 결과가 만족스럽지 않다.
# 기존에 hsv 마스크에서는 구분했던 색 영역이 하나로 묶이면서, 배경과 글씨 색이 비슷할 경우 아예 구분 안되는 현상이 발생한다.
# 현재 이 코드는 64색으로 줄이다보니 너무 색의 폭을 줄여서 생긴건 아닐까 생각된다.
##################################################################################

# 이미지 로드
standardFile = 'hwang/imgSet/test3.png'
imgStandard = cv.imread(standardFile, cv.IMREAD_COLOR)

# 이미지 출력용, 이미지 종횡 크기 구하기
imgCopySize = np.zeros_like(imgStandard)
height, width = imgStandard.shape[:2]

# 이미지 픽셀 정보 확인
pixelInfo = [[0 for x in range(width)] for y in range(height)]

# 0~255 픽셀 영역을 4등분 한다. (0~63, 64~127, 128~191, 192~255)
# 각 영역의 중심값으로 값을 변경한다.
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

