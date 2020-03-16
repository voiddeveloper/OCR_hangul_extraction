import cv2 as cv
import os
import glob
import random
import math

#############################################################################################
# 이미지에서 글씨 덩어리를 찾아내는 코드
# findFontAspectRatio.py를 통해 알아낸 폰트의 종횡비는 다음과 같다.
# 맑은 고딕 기준
# 종횡비 세로가 가장 긴 비율 (0.7 (width / height))
# 해당 글자 : 냬, 랚, 럑, 렦, 릮
# 종횡비 가로가 가장 긴 비율 (1.286 (width / height))
# 해당 글자 : 뚀, 쬬

# 추측 1: 글씨는 자음, 모음 등 여러가지가 서로 모여서 하나의 글씨를 이룬다. 즉 contour 주변에 다른 contour가 무조건 있을 것이다.
# 추측 2: 폰트의 종횡비는 항상 일정하다. 그렇다면 추측1로 추려낸 contour 집합체의 종횡비가 폰트의 종횡비인지 확인한다.
# 즉, contour들의 위치값이 글씨와 유사하면서, 해당 contour 전체의 종횡비가 (0.7 ~ 1.286) 범위 내에 있으면 이것은 글씨일 것이다.

# ## 목표 ##
# 1. 이미지에서 contour를 찾아낸다.
# 2. 해당 contour 주변에 다른 contour가 있는지 찾아본다.
# 3. 이 contour들의 위치값이 글씨 유형과 맞는지 확인하고, 글씨 유형이라면 하나의 그룹으로 묶는다.
# 4. 하나의 그룹으로 묶인 contour 집합 전체의 종횡비를 구해본다. (0.7 ~ 1.286) 범위 내의 종횡비라면 그것을 글씨로 인식한다.
#############################################################################################

# box를 그리는데 사용할 minX, maxX, minY, maxY값 반환 메소드
# pointList는 (x, y, w, h)의 형식으로 저장된 배열값이 들어와야 한다.
def findMinMaxPoint(pointList):
    minX = pointList[0][0]
    minY = pointList[0][1]
    maxX = 0
    maxY = 0
    
    for i in range(len(pointList)):
        if pointList[i][0] < minX:
            minX = pointList[i][0]
        if pointList[i][1] < minY:
            minY = pointList[i][1]
        if pointList[i][0] + pointList[i][2] > maxX:
            maxX = pointList[i][0] + pointList[i][2]
        if pointList[i][1] + pointList[i][3] > maxY:
            maxY = pointList[i][1] + pointList[i][3]
    
    return (minX, minY, maxX, maxY)

## main 시작 ##

# 이미지 읽기 및 contour 정보 얻기
standardFile = 'hwang/imgSet/comparisonImage/test_comparison1.png'
imgStandard = cv.imread(standardFile, cv.IMREAD_COLOR)
imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)
ret, imgBinary = cv.threshold(imgGrayscale, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# contour 정보(x, y, w, h)를 boxPoint에 저장한다.
# x: contour의 시작점 x 좌표
# y: contour의 시작점 y 좌표
# w: contour의 가로 길이
# h: contour의 세로 길이
boxPoint = []

for i in range(len(contours)):
    # contours로 찾아낸 물체
    cnt = contours[i]
    x, y, w, h = cv.boundingRect(cnt)
    # 찾아낸 영역의 x,y,w,h값 저장
    boxPoint.append(cv.boundingRect(cnt))

    # 사각형으로 영역 설정하기 
    cv.rectangle(imgStandard, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    # 찾아낸 영역 테두리 그려보기
    # cv.drawContours(imgStandard, [cnt], 0, (0, 255, 0), 2)
    
# 찾아낸 contour와 인접하게 붙어있는 contour가 서로 연관성이 있는지 검사한다.
# 1. 받침이 없는 유형
# 1-1. 우측에 모음이 있다. (ex: 가, 헤, 벼, 치)
for i in range(len(boxPoint)):
    realationBox = []

    # contour 중점 구하기
    centerX = boxPoint[i][0] + (boxPoint[i][2] / 2)
    centerY = boxPoint[i][1] + (boxPoint[i][3] / 2)
    predictionXPoint = centerX + boxPoint[i][2]

    # contour 중점을 기준으로 우측으로 width 만큼 이동했을 때, 다른 contour가 있다면 서로 연관이 있다.
    for j in range(len(boxPoint)):
        if predictionXPoint > boxPoint[j][0] and predictionXPoint < boxPoint[j][0] + boxPoint[j][2] and centerY > boxPoint[j][1] and centerY < boxPoint[j][1] + boxPoint[j][3]:
            # 연관이 있는 contour를 전부 묶었을 때, 종횡비가 0.7 ~ 1.29 (width / height) 안에 들어있는지 확인한다.
            realationBox.append(boxPoint[i])
            realationBox.append(boxPoint[j])
            minmaxPoint = findMinMaxPoint(realationBox)
            width = minmaxPoint[2] - minmaxPoint[0]
            height = minmaxPoint[3] - minmaxPoint[1]

            if (width / height) >= 0.7 and (width / height) <= 1.29:
                cv.rectangle(imgStandard, (minmaxPoint[0], minmaxPoint[1]), (minmaxPoint[2], minmaxPoint[3]), (0, 0, 255), 2)


cv.rectangle(imgStandard, (minmaxPoint[0], minmaxPoint[1]), (minmaxPoint[2], minmaxPoint[3]), (0, 0, 255), 2)

##################################################################################
# 이하 작업은 진행하지 못했다.

# 1-2. 하단에 모음이 있다. (ex: 구, 로, 므)
# 1-3. 하단 + 우측에 모음이 있다. (ex: 과, 뤄, 뵈, 의)
##
# 2. 받침이 1개 있는 유형
# 2-1. 우측에 모음이 있고, 하단에 받침이 있다. (ex: 각, 답, 졏, 방)
# 2-2. 하단에 모음이 있고, 그 하단에 받침이 있다. (ex: 곡, 롤, 븝)
# 2-3. 하단 + 우측에 모음이 있고, 그 하단에 받침이 있다. (ex: 궉, 봡, 쇳)
##
# 3. 받침이 2개 있는 유형
# 3-1. 우측에 모음이 있고, 하단에 받침이 있다. (ex: 값, 닳, 힕, 졌)
# 3-2. 하단에 모음이 있고, 그 하단에 받침이 있다. (ex: 곲, 욻, 흝)
# 3-3. 하단 + 우측에 모음이 있고, 그 하단에 받침이 있다. (ex: 괎, 뤊, 흱)
##
# 4. 어디에도 해당하지 않지만 종횡비가 0.7 ~ 1.29 (width / height) 안에 들어있다.

## 문제점 ##
# 이 작업 도중 새로운 문제점을 인지했다.
# 글씨의 모양들 중 예외 사항이 매우 많다.
# 예를 들어, 하단에 모음이 있는 유형(무, 므, 브) 등은 contour 집합이 자음 + 모음으로 2개가 될 것 같지만
# '그' 같은 타입은 자음과 모음이 붙어있어서 contour가 1개로 인식된다.
# 이러한 예외사항이 폰트마다 전부 다르다. (굴림 폰트에서는 붙어있지 않는데, 나눔 폰트에서는 붙어있는 등)
# 따라서 각 폰트별, 모든 한글 글씨를 전부 확인 후 분류해야 한다.
# 이 분류 작업 도중 시간 소모가 너무 크다고 판단해서 이후 작업을 중단하게 되었다.
# 이 후 작업은 combineContours.py로 넘어간다.

cv.imshow('imgStandard', imgStandard)
cv.waitKey(0)