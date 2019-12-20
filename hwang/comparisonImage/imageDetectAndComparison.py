import cv2 as cv
import numpy as np
import math

###############################################################
# 전역 변수 모음
# 테스트할 파일 이름
standardFile = 'hwang/imgSet/test_standard2.png'
comparisonFile = 'hwang/imgSet/test_comparison3.png'
# 기준/비교 contour * ratio간 픽셀 오차 범위 (단위: 픽셀 절대값)
errorRangePixel = 10
# errorRangePixel = 0.05
# 비교하지 않는 픽셀 범위 (단위: 픽셀 절대값)
exceptionPixel = 6
# 이미지를 분할하여 구분할 때 분할 범위
divisionCount = 3
# 영역간 픽셀 일치 오차율(단위: %)
errorEqualRate = 25
###############################################################

# 목표: 기준 이미지와 비슷한 모양을 비교 이미지에서 찾아낸다.
# 기준 이미지와 종횡비가 맞다면 크기가 달라져도 찾아내야 한다. 종횡비가 다르다면 걸러내야 한다.
# 방법
# 1. 기준 이미지에서 contour 영역을 찾아낸 후, 서로간 상호 거리를 구한다.
# 2. 비교 이미지에서 contour 영역을 찾아낸다.
# 3. 비교 이미지에서 각 contour와 기준 이미지의 contour 크기 비율을 비교한다. 비율이 똑같지 않은 것은 버린다.
# (완전 똑같을 수는 없기 때문에 오차 범위를 준다. errorRangePixel)
# 너무 작은 값은 오차율이 심해서 비교하지 않는다. exceptionPixel)
# 4. 같은 비율의 contour를 찾아냈다면, 해당 contour 주변에 기준 이미지처럼 상호 거리도 같은지 비교한다. 거리도 비율로 구하고, 똑같지 않은 것은 버린다.
# (아까와 마찬가지로 완전 똑같을 수는 없기 때문에 오차 범위를 준다. errorRangePixel)
# 5. 상호 거리가 일치하고, 상호 연결된 contours 갯수도 같다면 같은 이미지로 본다.
##################################################################################################
# 개선 사항: 여기서 추가적으로 기준 이미지의 contour와 비교 이미지의 contour 픽셀을 비교하면 좀 더 정확할 것 같다.

# 해당 위치에 label을 입력하는 메서드
# image = cv이미지
# str = label에 입력할 텍스트
# contour = contour 위치
def setLabel(image, str, contour):
    (textWidth, textHeight), baseLine = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x, y, width, height = cv.boundingRect(contour)
    ptX = x + int((width - textWidth) / 2)
    ptY = y + int((height + textHeight) / 2)
    cv.rectangle(image, (ptX, ptY + baseLine), (ptX + textWidth, ptY - textHeight), (200, 200, 200), cv.FILLED)
    cv.putText(image, str, (ptX, ptY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, 8)

# 픽셀을 n등분하여 각 영역별 얼마나 분포되어 있는지 구하는 메서드
# image = cv이미지(binary 이미지를 추천함)
# pixelPoint = image 내에 존재하는 contour 집합 리스트, (x,y,w,h)로 구성한다.
def pixelInfo(image, pixelPoint):
    pixelLocation = []
    for i in range(len(pixelPoint)):
        imgCut = image.copy()
        # contour를 기준으로 이미지를 분할한다.
        imgCut = image[pixelPoint[i][1]:pixelPoint[i][1] + pixelPoint[i][3], pixelPoint[i][0]:pixelPoint[i][0] + pixelPoint[i][2]]
        # n등분할 때 소수점이 나오지 않도록 resize 한다.
        imgResize = cv.resize(imgCut, (0, 0), fx = divisionCount, fy = divisionCount, interpolation= cv.INTER_AREA)

        # 각 contour 영역을 divisionCount값 만큼 분할한다.
        contourPixelInfo = []
        for j in range(divisionCount):
            for k in range(divisionCount):
                contourCut = imgResize.copy()
                contourCut = imgResize[k * pixelPoint[i][3]:k * pixelPoint[i][3] + pixelPoint[i][3], j * pixelPoint[i][2]:j * pixelPoint[i][2] + pixelPoint[i][2]]

                # 각 분할된 영역마다 Pixel 값이 몇%나 차지하는지 확인한다.
                height, width = contourCut.shape[:2]
                totalPixel = 0
                for m in range(0, height):
                    for n in range(0, width):
                        totalPixel += (contourCut[m, n] / 255)
                pixelRatio = round((totalPixel / (height * width)) * 100, 2)

                # 영역 차지 값을 저장한다.
                contourPixelInfo.append(pixelRatio)

        # contour별 영역 차지 값을 저장해서 반환한다.
        pixelLocation.append(contourPixelInfo)
    return pixelLocation

# 기준 이미지, 비교 이미지 설정
imgStandard = cv.imread(standardFile, cv.IMREAD_COLOR)
imgComparison = cv.imread(comparisonFile, cv.IMREAD_COLOR)

########################################################################################
# 1. 기준 이미지에서 어떤 영역들이 얼마나 떨어진 위치에 있는지 값을 구한다.
########################################################################################
# 컬러 이미지를 그레이스케일로 변환
imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)

# 그레이스케일로 변환된 이미지를 바이너리로 변환
ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

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
    # cv.rectangle(imgStandard, (x, y), (x + w, y + h), (0, 255, 0), 1)
    rectangleCount += 1
    # setLabel(imgStandard, str(rectangleCount), cnt)

# 영역별 거리값 구하기
for i in range(len(boxPoint)):
    if i != 0 :
        # distanceX = math.sqrt(pow(boxPoint[0][0] - boxPoint[i][0], 2))
        # distanceY = math.sqrt(pow(boxPoint[0][1] - boxPoint[i][1], 2))
        distanceX = boxPoint[i][0] - boxPoint[0][0]
        distanceY = boxPoint[i][1] - boxPoint[0][1]
        
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
    # cv.rectangle(imgComparison, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # setLabel(imgComparison, str(rectangleCount), cnt)
    rectangleCount += 1

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
# print('boxPoint = ', boxPoint)
# print('contoursDistance = ', contoursDistance)
# print('boxPointComparison = ', boxPointComparison)
# print('contoursDistanceComparison = ', contoursDistanceComparison)
# print('boxPoint[0][2] = ', boxPoint[0][2])

# 기준 이미지를 분할하여 각 영역마다 어느정도 픽셀이 있는지 파악하기
standardPixelInfo = pixelInfo(imgBinary, boxPoint)
comparisonPixelInfo = pixelInfo(imgBinaryComparison, boxPointComparison)
# print('standardPixelInfo = ', standardPixelInfo)
# print('comparisonPixelInfo = ', comparisonPixelInfo)

########################################################################################
# 3. 기준 이미지와 비교 이미지의 비율 차이를 구하고, 비율이 동일한지 확인한다.
########################################################################################
for i in range(len(boxPointComparison)):
    # 기준 영역과 비교 영역 간 비율 차이 (width 값으로 구하기)
    ratio =  boxPointComparison[i][2] / boxPoint[0][2]
    
    # 너무 작은 영역은 비교하지 않는다.
    # 늘어난 비율만큼을 적용해서 height값도 일치하는지 확인 (+- 오차 범위를 준다.)
    # 이 조건을 일치하지 않으면 어차피 같은 비율의 그림이 아니기 때문에 더 이상 비교할 필요가 없다.
    if boxPointComparison[i][2] > exceptionPixel and boxPointComparison[i][3] > exceptionPixel:
        # print('ratio = ', ratio)
        # print('boxPoint[0][3] * ratio = ', boxPoint[0][3] * ratio)
        # print('boxPointComparison[i][3] = ', boxPointComparison[i][3])
        if(boxPointComparison[i][3] <= (boxPoint[0][3] * ratio) + errorRangePixel and boxPointComparison[i][3] >= (boxPoint[0][3] * ratio) - errorRangePixel):
        # if(boxPointComparison[i][3] <= (boxPoint[0][3] * ratio) * (1 + errorRangePixel) and boxPointComparison[i][3] >= (boxPoint[0][3] * ratio) * (1 - errorRangePixel)):
            
            ########################################################################################
            # 4. 비율이 동일하다면, 기준 이미지와 비교 이미지의 거리 차이를 구하고, 거리가 동일한지 확인한다. (위에서 구한 비율로 거리값을 조절한다.)
            ########################################################################################
            flag = []
            for j in range(len(contoursDistance)):
                # 다음 영역이 있어야 하는 예상 위치
                expectX = boxPointComparison[i][0] + (contoursDistance[j][0] * ratio)
                expectY = boxPointComparison[i][1] + (contoursDistance[j][1] * ratio)            
                # 정말로 예상 위치에 다음 영역이 존재하는지 확인한다.
                for k in range(len(boxPointComparison)):
                    # 예상 위치에 정말 영역이 존재한다면 몇번째 영역인지 저장한다.
                    if(boxPointComparison[k][0] >= expectX - errorRangePixel and boxPointComparison[k][0] <= expectX + errorRangePixel and boxPointComparison[k][1] >= expectY - errorRangePixel and boxPointComparison[k][1] <= expectY + errorRangePixel):
                    # if(boxPointComparison[k][0] >= expectX * (1 - errorRangePixel) and boxPointComparison[k][0] <= expectX * (1 + errorRangePixel) and boxPointComparison[k][1] >= expectY * (1 - errorRangePixel) and boxPointComparison[k][1] <= expectY * (1 + errorRangePixel)):
                        flag.append(k)

            ########################################################################################
            # 5. 이 모든 조건을 전부 충족하면 기준 이미지와 유사한 이미지다.
            # 해당 영역을 차지하는 좌표의 min, max값을 구해서 테두리를 친다.
            ########################################################################################
            finalOrder = []
            if len(contoursDistance) == len(flag):
                finalOrder.append(i)
                for l in range(len(flag)):
                    finalOrder.append(flag[l])
                # print(finalOrder)

                # 기준 이미지의 영역 비율과 비교 이미지의 영역 비율이 일치하는지 확인한다.
                maxRange = 0
                for m in range(len(finalOrder)):
                    for n in range(0, pow(divisionCount, 2)):
                        # print('standardPixelInfo = ', standardPixelInfo[m][n])
                        # print('comparisonPixelInfo = ', comparisonPixelInfo[finalOrder[m]][n])
                        minmaxRange = math.sqrt(pow(standardPixelInfo[m][n] - comparisonPixelInfo[finalOrder[m]][n], 2))
                        if maxRange < minmaxRange:
                            maxRange = minmaxRange

                # print('maxRange = ', maxRange)
                if maxRange <= errorEqualRate:
                    # 테두리 치기
                    minX = boxPointComparison[finalOrder[0]][0]
                    maxX = boxPointComparison[finalOrder[0]][0] + boxPointComparison[finalOrder[0]][2]
                    minY = boxPointComparison[finalOrder[0]][1]
                    maxY = boxPointComparison[finalOrder[0]][1] + boxPointComparison[finalOrder[0]][3]

                    for m in range(len(finalOrder)):
                        if minX > boxPointComparison[finalOrder[m]][0]:
                            minX = boxPointComparison[finalOrder[m]][0]

                        if maxX < boxPointComparison[finalOrder[m]][0] + boxPointComparison[finalOrder[m]][2]:
                            maxX = boxPointComparison[finalOrder[m]][0] + boxPointComparison[finalOrder[m]][2]
                        
                        if minY > boxPointComparison[finalOrder[m]][1]:
                            minY = boxPointComparison[finalOrder[m]][1]

                        if maxY < boxPointComparison[finalOrder[m]][1] + boxPointComparison[finalOrder[m]][3]:
                            maxY = boxPointComparison[finalOrder[m]][1] + boxPointComparison[finalOrder[m]][3]

                    cv.rectangle(imgComparison, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

cv.imshow('comparison', imgComparison)
cv.waitKey(0)