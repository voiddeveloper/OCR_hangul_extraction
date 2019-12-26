import cv2 as cv
import os
import glob
import random
import math

def setLabel(image, str, contour):
    (textWidth, textHeight), baseLine = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x, y, width, height = cv.boundingRect(contour)
    ptX = x + int((width - textWidth) / 2)
    ptY = y + int((height + textHeight) / 2)
    cv.rectangle(image, (ptX, ptY + baseLine), (ptX + textWidth, ptY - textHeight), (200, 200, 200), cv.FILLED)
    cv.putText(image, str, (ptX, ptY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, 8)

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

#############################################################################################
# 이미지에서 글씨 덩어리를 찾아내는 프로그램
# 맑은 고딕 기준
# 종횡비 세로가 가장 긴 비율 (0.7 (width / height))
# 해당 글자 : 냬, 랚, 럑, 렦, 릮
# 종횡비 가로가 가장 긴 비율 (1.29 (width / height))
# 해당 글자 : 뚀, 쬬
#############################################################################################

standardFile = 'hwang/imgSet/test_comparison5.jpg'
# standardFile = 'hwang/imgSet/test_standard1.png'
imgStandard = cv.imread(standardFile, cv.IMREAD_COLOR)
imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)
ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

boxPoint = []
rectangleCount = 0

for i in range(len(contours)):
    # contours로 찾아낸 물체
    cnt = contours[i]
    x, y, w, h = cv.boundingRect(cnt)
    # 찾아낸 영역의 x,y,w,h값 저장
    boxPoint.append(cv.boundingRect(cnt))

    # 테투리로 영역 설정하기
    # cv.drawContours(imgStandard, [cnt], 0, (0, 255, 0), 2)
    # 사각형으로 영역 설정하기 
    cv.rectangle(imgStandard, (x, y), (x + w, y + h), (0, 255, 0), 1)
    rectangleCount += 1
    # setLabel(imgStandard, str(rectangleCount), cnt)

##
# 찾아낸 물체와 인접하게 붙어있는 물체가 서로 연관성이 있는지 검사한다.
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




cv.imshow('imgStandard', imgStandard)
cv.waitKey(0)