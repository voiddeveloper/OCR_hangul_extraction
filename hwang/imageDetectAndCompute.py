import cv2 as cv
import numpy as np
import math

def setLabel(image, str, contour):
    (textWidth, textHeight), baseLine = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x, y, width, height = cv.boundingRect(contour)
    ptX = x + int((width - textWidth) / 2)
    ptY = y + int((height + textHeight) / 2)
    cv.rectangle(image, (ptX, ptY + baseLine), (ptX + textWidth, ptY - textHeight), (200, 200, 200), cv.FILLED)
    cv.putText(image, str, (ptX, ptY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, 8)

# 기준 이미지, 비교 이미지 설정
imgStandard = cv.imread('img/test_standard.png')
imgComparison = cv.imread('img/test_comparison.png')

########################################################################################
# 1. 기준 이미지에서 어떤 영역들이 얼마나 떨어진 위치에 있는지 값을 구한다.
########################################################################################
# 컬러 이미지를 그레이스케일로 변환
imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)

# 그레이스케일로 변환된 이미지를 바이너리로 변환
ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
# cv.imshow('original', imgStandard)
# cv.imshow('grayscale', imgGrayscale)
# cv.imshow('binary', imgBinary)
# cv.waitKey(0)

# contours = 동일한 색을 가지고 있는 영역의 경계선 정보
# RETR_EXTERNAL = contours 정보 중에서 바깥쪽 라인만 찾는다.
# CHAIN_APPROX_SIMPLE = contours 라인을 그릴 수 있는 포인트를 반환
contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 찾아낸 영역의 좌표 저장용 배열, (x, y, w, h)값을 저장한다. 
boxPoint = []
# 찾아낸 영역간의 거리 차이 저장용 배열
contoursDistance = []
rectangleCount = 0

# 영역 설정하기
for i in range(len(contours)):
    # contours로 찾아낸 물체
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)
    # 찾아낸 영역의 x,y,w,h값 저장
    boxPoint.append(cv.boundingRect(cnt))

    # 테투리로 영역 설정하기
    # cv.drawContours(imgStandard, [cnt], 0, (0, 255, 0), 2)
    # 사각형으로 영역 설정하기 
    cv.rectangle(imgStandard, (x, y), (x + w, y + h), (0, 255, 0), 1)
    rectangleCount += 1
    setLabel(imgStandard, str(rectangleCount), cnt)

# 영역별 거리값 구하기
for i in range(len(boxPoint)):
    if i != 0 :
        # distanceX = math.sqrt(pow(boxPoint[0][0] - boxPoint[i][0], 2))
        # distanceY = math.sqrt(pow(boxPoint[0][1] - boxPoint[i][1], 2))
        distanceX = boxPoint[0][0] - boxPoint[i][0]
        distanceY = boxPoint[0][1] - boxPoint[i][1]
        
        # 저장하는 값은 첫번째 영역과 다른 영역과의 x, y 거리 절대값, 첫번째 영역의 width, height 길이
        contoursDistance.append((int(distanceX), int(distanceY)))

cv.imshow('original', imgStandard)
# cv.waitKey(0)

########################################################################################
# 2. 비교 이미지에서 어떤 영역들이 있는지 찾아낸다.
########################################################################################
# 컬러 이미지를 그레이스케일로 변환
imgGrayscaleComparison = cv.cvtColor(imgComparison, cv.COLOR_BGR2GRAY)

# 그레이스케일로 변환된 이미지를 바이너리로 변환
ret, imgBinaryComparison = cv.threshold(imgGrayscaleComparison, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

# 동일한 색이 있는 영역 경계 정보 찾기
contoursComparison, hierarchyComparison = cv.findContours(imgBinaryComparison, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 찾아낸 영역의 좌표 저장용 배열, (x, y, w, h)값을 저장한다. 
boxPointComparison = []
# 찾아낸 영역간의 거리 차이 저장용 배열
contoursDistanceComparison = []


rectangleCount = 0

# 영역 설정하기
for i in range(len(contoursComparison)):
    # contours로 찾아낸 물체
    cnt = contoursComparison[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)
    # 찾아낸 영역의 x,y,w,h값 저장
    boxPointComparison.append(cv.boundingRect(cnt))

    # 테투리로 영역 설정하기
    # cv.drawContours(imgStandard, [cnt], 0, (0, 255, 0), 2)
    # 사각형으로 영역 설정하기 
    rectangleCount += 1
    cv.rectangle(imgComparison, (x, y), (x + w, y + h), (0, 255, 0), 1)
    setLabel(imgComparison, str(rectangleCount), cnt)

# 영역별 거리값 구하기
for i in range(len(boxPointComparison)):
    if i != 0 :
        distanceX = math.sqrt(pow(boxPointComparison[0][0] - boxPointComparison[i][0], 2))
        distanceY = math.sqrt(pow(boxPointComparison[0][1] - boxPointComparison[i][1], 2))
        
        # 저장하는 값은 첫번째 영역과 다른 영역과의 x, y 거리 절대값, 첫번째 영역의 width, height 길이
        contoursDistanceComparison.append((int(distanceX), int(distanceY)))

# boxPoint = 기준 이미지의 Contour (x, y, w, h) 좌표
# contoursDistance = 기준 이미지에서 1번째 Contour를 기준으로 다른 Contour들 간 얼마나 떨어져 있는지 비교한다. (x축 거리, y축 거리)

# boxPointComparison = 비교 이미지의 Contour (x, y, w, h) 좌표
# contoursDistanceComparison = 비교 이미지에서 1번째 Contour를 기준으로 다른 Contour들 간 얼마나 떨어져 있는지 비교한다. (x축 거리, y축 거리)

print(contoursDistance)
print(boxPointComparison)
print(contoursDistanceComparison)
print(boxPoint[0][2])

for i in range(len(boxPointComparison)):
    # 기준 이미지와 비교 이미지간 비율 차이
    ratio =  boxPointComparison[i][2] / boxPoint[0][2]
    
    
    a = contoursDistance[0][0] * (contoursDistance[0][0] * ratio)
    
    None

cv.imshow('comparison', imgComparison)
cv.waitKey(0)