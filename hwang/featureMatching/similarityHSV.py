import cv2 as cv
import numpy as np
import math
import queue
import time
import sys
from random import *
from matplotlib import pyplot as plt

# 2개의 HSV 색상값을 입력 받을 시, 2개의 색상 유사도 거리값 구하는 메소드
def findTwoColorDistanceHSV(hsvColor1, hsvColor2):
    # 2개의 색상의 공간 좌표값을 구하기 (HSV conic 모델)
    # 원뿔의 꼭지점을 원점, 세로축을 z축이라고 가정한다.
    hsvColor1H = int(hsvColor1[0])
    hsvColor1S = int(hsvColor1[1])
    hsvColor1V = int(hsvColor1[2])
    hsvColor2H = int(hsvColor2[0])
    hsvColor2S = int(hsvColor2[1])
    hsvColor2V = int(hsvColor2[2])

    x1 = (hsvColor1S * math.cos(2 * (math.pi * (hsvColor1H / 255))) * hsvColor1V) / 255
    y1 = (hsvColor1S * math.sin(2 * (math.pi * (hsvColor1H / 255))) * hsvColor1V) / 255
    z1 = hsvColor1V
    
    x2 = (hsvColor2S * math.cos(2 * (math.pi * (hsvColor2H / 255))) * hsvColor2V) / 255
    y2 = (hsvColor2S * math.sin(2 * (math.pi * (hsvColor2H / 255))) * hsvColor2V) / 255
    z2 = hsvColor2V

    # 2개의 색의 좌표값이 나오면 유클리디언 거리 공식을 이용하여 값을 구하기
    # d = sqrt{(h1-h2)^2 + (s1-s2)^2 + (v1-v2)^2}
    distance = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2)) 

    return distance

# 유사한 색끼리 그룹을 묶어주는 메소드
# 자신의 색깔과 주변 상/하/좌/우 픽셀의 색깔을 비교하여 얼마나 비슷한지 판별한다. (판별하는 방법은 findTwoColorDistanceHSV)
# 만약 비슷한 색이라고 판단되면, 하나의 그룹으로 묶어준다.
# 그룹이 생성되면, 해당 그룹의 주변 픽셀 색깔을 계산하고 또 비슷한 픽셀이 있는지 비교한다.
# 이 과정을 주변에 비슷한 픽셀이 없을 때까지 반복한다.
def pixelLinkList(img):
    startTime = time.time()

    # 기본 컬러 = BGR
    imgOriginal = img
    # BGR -> HSV로 변경
    imgHSV = cv.cvtColor(imgOriginal, cv.COLOR_BGR2HSV)

    # 픽셀값을 저장할 배열 생성
    height, width = imgOriginal.shape[:2]
    pixelInfoBGR = [[0 for x in range(height)] for y in range(width)]
    pixelInfoHSV = [[0 for x in range(height)] for y in range(width)]

    # BGR, HSV 픽셀값 저장
    for i in range(0, width):
        for j in range(0, height):
            pixelInfoBGR[i][j] = imgOriginal[j, i]
            pixelInfoHSV[i][j] = imgHSV[j, i]

    # 그림 좌표 배열 생성
    pixelPosition = []
    positionX = 0
    positionY = 0

    # 좌표값을 저장한다.
    for i in range(0, int(width * height)):
        pixelPosition.append((positionX, positionY))
        
        positionY += 1
        if positionY == height:
            positionY = 0
            positionX += 1

    # 유사한 색이라고 인정하는 색의 거리, findTwoColorDistanceHSV 메소드의 distance 값에 이용된다.
    similarityDistance = 35

    i = 0
    j = 0

    # 그림 전체 기준에서 비슷한 색깔끼리 하나의 집합으로 묶이는 리스트
    pixelList = []

    while True:
        # 주변 픽셀을 비교할 좌표 시작점
        nextPointX = i
        nextPointY = j

        # 비슷한 색상으로 인정된 픽셀 집합
        pixelLink = []

        # 최소한 자기 자신은 무조건 pixelLink에 포함된다.
        pixelLink.append((i, j))

        # dict 생성, default 값은 False
        pixelMap = {}
        for positionX in range(0, width):
            for positionY in range(0, height):
                pixelMap[str(positionX) + ',' + str(positionY)] = False

        # 해당 위치의 pixelMap 값을 True로 변환한다.
        pixelMap[str(i) + ',' + str(j)] = True

        # 나중에 추가적으로 비교해야 할 위치를 저장하는 리스트
        savePoint = queue.Queue()
        
        while True:
            # 자신의 상/하/좌/우 픽셀 색상과 자신의 색상이 얼마나 비슷한지 계산한다.
            # 단, 해당 좌표의 pixelMap 값이 False 일 경우에만 저장한다.

            # 우측 좌표 비교
            if i < width - 1:
                if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i + 1][j]):
                    if not pixelMap[str(i + 1) + ',' + str(j)]:
                        pixelLink.append((i + 1, j))
                        savePoint.put(str(i + 1) + ',' + str(j))
                        pixelMap[str(i + 1) + ',' + str(j)] = True
            
            # 아래쪽 좌표 비교
            if j < height - 1:
                if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i][j + 1]):
                    if not pixelMap[str(i) + ',' + str(j + 1)]:
                        pixelLink.append((i, j + 1))
                        savePoint.put(str(i) + ',' + str(j + 1))
                        pixelMap[str(i) + ',' + str(j + 1)] = True

            # 위쪽 좌표 비교
            if j > 0:
                if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i][j - 1]):
                    if not pixelMap[str(i) + ',' + str(j - 1)]:
                        pixelLink.append((i, j - 1))
                        savePoint.put(str(i) + ',' + str(j - 1))
                        pixelMap[str(i) + ',' + str(j - 1)] = True

            # 좌측 좌표 비교
            if i > 0:
                if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i - 1][j]):
                    if not pixelMap[str(i - 1) + ',' + str(j)]:
                        pixelLink.append((i - 1, j))
                        savePoint.put(str(i - 1) + ',' + str(j))
                        pixelMap[str(i - 1) + ',' + str(j)] = True

            # 다음에 비교할 좌표가 없다면 반복을 멈춘다.
            if savePoint.empty():
                break
            
            # 다음에 비교할 좌표값을 뽑는다.
            else:
                position = savePoint.get()
                position = position.split(',')
                i = int(position[0])
                j = int(position[1])

        # 비슷한 색깔의 픽셀 묶음이 나왔다면, 해당 값을 pixelList에 저장한다.
        pixelList.append(pixelLink)

        # print(pixelLink)

        # pixelList로 선정된 좌표는 그림 좌표에서 제거한다.
        pixelPosition = list(set(pixelPosition) - set(pixelLink))

        if len(pixelPosition) > 0:
            i = pixelPosition[0][0]
            j = pixelPosition[0][1]

        else:
            break

    # 비슷한 색의 그룹이라 묶인 pixelList를 표시하기
    for k in range(0, len(pixelList)):
        red = randint(0, 255)
        green = randint(0, 255)
        blue = randint(0, 255)

        for l in range(0, len(pixelList[k])):
            cv.line(imgOriginal, (pixelList[k][l][0], pixelList[k][l][1]), (pixelList[k][l][0], pixelList[k][l][1]), (blue, green, red), 1)
    
    print('pixelList 전체 묶음 갯수 : ', len(pixelList))
    print('1번째 pixel 묶음 pixel 갯수 : ', len(pixelList[0]))
    # print(pixelList[0])
    # for k in range(0, len(pixelList)):
    #     print(len(pixelList[k]))

    cv.imshow("test", imgOriginal)
    print('time : ', time.time() - startTime)
    cv.waitKey(0)
    
    return len(pixelList)

# pixelList의 1번째 그룹은, 전체 이미지에서 테두리가 없는 모든 영역을 하나로 다 묶는다.
# 따라서, 이 pixelList 1번째 리스트의 값을 뺀 나머지 영역을 검출해내면, 테두리가 있는 영역 부분만 검출된다.
def reverseFirstLayer(img):
    startTime = time.time()

    # 기본 컬러 = BGR
    imgOriginal = img
    # BGR -> HSV로 변경
    imgHSV = cv.cvtColor(imgOriginal, cv.COLOR_BGR2HSV)

    # 픽셀값을 저장할 배열 생성
    height, width = imgOriginal.shape[:2]
    pixelInfoBGR = [[0 for x in range(height)] for y in range(width)]
    pixelInfoHSV = [[0 for x in range(height)] for y in range(width)]

    # BGR, HSV 픽셀값 저장
    for i in range(0, width):
        for j in range(0, height):
            pixelInfoBGR[i][j] = imgOriginal[j, i]
            pixelInfoHSV[i][j] = imgHSV[j, i]

    # 그림 좌표 배열 생성
    pixelPosition = []
    positionX = 0
    positionY = 0

    # 좌표값을 저장한다.
    for i in range(0, int(width * height)):
        pixelPosition.append((positionX, positionY))
        
        positionY += 1
        if positionY == height:
            positionY = 0
            positionX += 1

    # 유사한 색이라고 인정하는 색의 거리, findTwoColorDistanceHSV 메소드의 distance 값에 이용된다.
    similarityDistance = 30

    i = 0
    j = 0

    # 비슷한 색상으로 인정된 픽셀 집합
    pixelLink = []

    # 최소한 자기 자신은 무조건 pixelLink에 포함된다.
    pixelLink.append((i, j))

    # dict 생성, default 값은 False
    pixelMap = {}
    for positionX in range(0, width):
        for positionY in range(0, height):
            pixelMap[str(positionX) + ',' + str(positionY)] = False

    # 시작 위치의 pixelMap 값을 True로 변환한다.
    pixelMap['0,0'] = True

    # 나중에 추가적으로 비교해야 할 위치를 저장하는 리스트
    savePoint = queue.Queue()
    
    while True:
        # 자신의 상/하/좌/우 픽셀 색상과 자신의 색상이 얼마나 비슷한지 계산한다.
        # 서로의 색상이 비슷하다면 비슷한 색이 있는 좌표를 저장한다.
        # 단, 해당 좌표의 pixelMap 값이 False 일 경우에만 저장한다. 저장한 후 해당 위치의 pixelMap 값을 True로 변환한다.

        # 우측 좌표 비교
        if i < width - 1:
            if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i + 1][j]):
                if not pixelMap[str(i + 1) + ',' + str(j)]:
                    pixelLink.append((i + 1, j))
                    savePoint.put(str(i + 1) + ',' + str(j))
                    pixelMap[str(i + 1) + ',' + str(j)] = True
        
        # 아래쪽 좌표 비교
        if j < height - 1:
            if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i][j + 1]):
                if not pixelMap[str(i) + ',' + str(j + 1)]:
                    pixelLink.append((i, j + 1))
                    savePoint.put(str(i) + ',' + str(j + 1))
                    pixelMap[str(i) + ',' + str(j + 1)] = True

        # 위쪽 좌표 비교
        if j > 0:
            if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i][j - 1]):
                if not pixelMap[str(i) + ',' + str(j - 1)]:
                    pixelLink.append((i, j - 1))
                    savePoint.put(str(i) + ',' + str(j - 1))
                    pixelMap[str(i) + ',' + str(j - 1)] = True

        # 좌측 좌표 비교
        if i > 0:
            if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i - 1][j]):
                if not pixelMap[str(i - 1) + ',' + str(j)]:
                    pixelLink.append((i - 1, j))
                    savePoint.put(str(i - 1) + ',' + str(j))
                    pixelMap[str(i - 1) + ',' + str(j)] = True

        # 다음에 비교할 좌표가 없다면 반복을 멈춘다.
        if savePoint.empty():
            break
        
        # 다음에 비교할 좌표값을 뽑는다.
        else:
            position = savePoint.get()
            position = position.split(',')
            i = int(position[0])
            j = int(position[1])

    # print(pixelLink)
    # pixelList로 선정된 좌표는 그림 좌표에서 제거한다.
    reverseLayer = list(set(pixelPosition) - set(pixelLink))

    imgBlack = np.zeros_like(imgOriginal)

    for k in range(0, len(reverseLayer)):
        cv.line(imgBlack, (reverseLayer[k][0], reverseLayer[k][1]), (reverseLayer[k][0], reverseLayer[k][1]), (0, 0, 255), 1)

    cv.imshow("test", imgBlack)
    print('time : ', time.time() - startTime)
    cv.waitKey(0)
    
    return pixelLink

def singleLineDraw(img, x, y, imgCopy):
    # startTime = time.time()

    # 기본 컬러 = BGR
    imgOriginal = img
    imgBlack = imgCopy

    # BGR -> HSV로 변경
    imgHSV = cv.cvtColor(imgOriginal, cv.COLOR_BGR2HSV)

    # 픽셀값을 저장할 배열 생성
    height, width = imgOriginal.shape[:2]
    pixelInfoBGR = [[0 for x in range(height)] for y in range(width)]
    pixelInfoHSV = [[0 for x in range(height)] for y in range(width)]

    # BGR, HSV 픽셀값 저장
    for i in range(0, width):
        for j in range(0, height):
            pixelInfoBGR[i][j] = imgOriginal[j, i]
            pixelInfoHSV[i][j] = imgHSV[j, i]

    # 그림 좌표 배열 생성
    pixelPosition = []
    positionX = 0
    positionY = 0

    # 좌표값을 저장한다.
    for i in range(0, int(width * height)):
        pixelPosition.append((positionX, positionY))
        
        positionY += 1
        if positionY == height:
            positionY = 0
            positionX += 1

    # 유사한 색이라고 인정하는 색의 거리, findTwoColorDistanceHSV 메소드의 distance 값에 이용된다.
    similarityDistance = 60

    startPointX = x
    startPointY = y
        
    # 원점과 비교할 좌표값
    nextPointX = startPointX
    nextPointY = startPointY

    # 비슷한 색상으로 인정된 픽셀 집합
    pixelLink = []

    # pixelPosition과 동일한 dictionary 생성, default 값은 False
    pixelMap = {}
    for i in range(len(pixelPosition)):
        pixelMap[pixelPosition[i]] = False

    # 최소한 시작점은 무조건 pixelLink에 포함된다.
    pixelLink.append((startPointX, startPointY))

    # 시작점의 pixelMap 값을 True로 변환한다.
    pixelMap[startPointX, startPointY] = True
    
    # 외곽 주변 픽셀을 찾기 위한 방향값
    direction = 0
    # 방향 선회를 몇번 했는지 체크하는 값
    directionCount = 0
    # 비교점이 바뀌었는지 체크하는 값
    moveFlag = False
    moveCount = 0
    samePointCount = False
    print(findTwoColorDistanceHSV(pixelInfoHSV[162][191], pixelInfoHSV[163][191]))
    
    # 시작점과 현재 비교점의 색상이 얼마나 차이나는지 비교한다.
    # 차이값이 similarityDistance값 이하면 비슷한 색으로 취급하고, 하나의 영역으로 묶는다.
    # 이미 묶인 좌표는 중복체크를 해서 제외한다.
    # 주변 픽셀을 비교하는 순서는 좌 -> 상 -> 우 -> 하(시계 방향) 순으로 진행한다.
    #  1 2 3
    #  0 x 4
    #  7 6 5  << 방향값 (0 ~ 7 순서로 진행한다.)
    # 한번 진행하면, 그 다음에 비교하는 순서는 방향값을 현재 방향값에서 -2 시킨다. (반시계 방향으로 90도 회전), 방향값이 음수가 되면 +8을 한다. 방향값이 8이상이 되면 -8을 한다.
    # 이 과정을 계속 반복한다.
    # 만일 현재 비교점이 시작점이 되면 종료한다.
    while True:
        # print('nextPointX = ', nextPointX, ' , nextPointY = ', nextPointY, ' direction = ', direction)
        # direction이 음수이면 +8을 한다.
        if direction < 0:
            direction += 8

        # direction이 8 이상이면 -8을 한다.
        elif direction >= 8:
            direction -= 8

        if directionCount < 8:
            # 좌측 좌표 비교 (direction = 0)
            if direction == 0:
                if nextPointX > 0:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX - 1][nextPointY]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX - 1][nextPointY]):
                        # 중복 체크
                        if pixelMap[nextPointX - 1, nextPointY] == False:
                            pixelLink.append((nextPointX - 1, nextPointY))
                            pixelMap[nextPointX - 1, nextPointY] = True

                        nextPointX -= 1
                        moveFlag = True

            # 좌상측 좌표 비교 (direction = 1)
            elif direction == 1:
                if nextPointX > 0 and nextPointY > 0:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX - 1][nextPointY - 1]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX - 1][nextPointY - 1]):
                        # 중복 체크
                        if pixelMap[nextPointX - 1, nextPointY - 1] == False:
                            pixelLink.append((nextPointX - 1, nextPointY - 1))
                            pixelMap[nextPointX - 1, nextPointY - 1] = True

                        nextPointX -= 1
                        nextPointY -= 1
                        moveFlag = True

            # 상측 좌표 비교 (direction = 2)
            elif direction == 2:
                if nextPointY > 0:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX][nextPointY - 1]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX][nextPointY - 1]):
                        # 중복 체크
                        if pixelMap[nextPointX, nextPointY - 1] == False:
                            pixelLink.append((nextPointX, nextPointY - 1))
                            pixelMap[nextPointX, nextPointY - 1] = True

                        nextPointY -= 1
                        moveFlag = True

            # 우상측 좌표 비교 (direction = 3)
            elif direction == 3:
                if nextPointX < width - 1 and nextPointY > 0:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX + 1][nextPointY - 1]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX + 1][nextPointY - 1]):
                        # 중복 체크
                        if pixelMap[nextPointX + 1, nextPointY - 1] == False:
                            pixelLink.append((nextPointX + 1, nextPointY - 1))
                            pixelMap[nextPointX + 1, nextPointY - 1] = True

                        nextPointX += 1
                        nextPointY -= 1
                        moveFlag = True

            # 우측 좌표 비교 (direction = 4)
            elif direction == 4:
                if nextPointX < width - 1:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX + 1][nextPointY]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX + 1][nextPointY]):
                        # 중복 체크
                        if pixelMap[nextPointX + 1, nextPointY] == False:
                            pixelLink.append((nextPointX + 1, nextPointY))
                            pixelMap[nextPointX + 1, nextPointY] = True
                        
                        nextPointX += 1
                        moveFlag = True

            # 우하측 좌표 비교 (direction = 5)
            elif direction == 5:
                if nextPointX < width - 1 and nextPointY < height - 1:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX + 1][nextPointY + 1]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX + 1][nextPointY + 1]):
                        # 중복 체크
                        if pixelMap[nextPointX + 1, nextPointY + 1] == False:
                            pixelLink.append((nextPointX + 1, nextPointY + 1))
                            pixelMap[nextPointX + 1, nextPointY + 1] = True
                        
                        nextPointX += 1
                        nextPointY += 1
                        moveFlag = True

            # 하측 좌표 비교 (direction = 6)
            elif direction == 6:
                if nextPointY < height - 1:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX][nextPointY + 1]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX][nextPointY + 1]):
                        # 중복 체크
                        if pixelMap[nextPointX, nextPointY + 1] == False:
                            pixelLink.append((nextPointX, nextPointY + 1))
                            pixelMap[nextPointX, nextPointY + 1] = True

                        nextPointY += 1
                        moveFlag = True

            # 좌하측 좌표 비교 (direction = 7)
            elif direction == 7:
                if nextPointX > 0 and nextPointY < height - 1:
                    if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[startPointX][startPointY], pixelInfoHSV[nextPointX - 1][nextPointY + 1]):
                    # if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[nextPointX][nextPointY], pixelInfoHSV[nextPointX - 1][nextPointY + 1]):
                        # 중복 체크
                        if pixelMap[nextPointX - 1, nextPointY + 1] == False:
                            pixelLink.append((nextPointX - 1, nextPointY + 1))
                            pixelMap[nextPointX - 1, nextPointY + 1] = True

                        nextPointX -= 1
                        nextPointY += 1
                        moveFlag = True

            # 비교점이 바뀌었는지 확인한다.
            if moveFlag:
                moveCount += 1

                # 바뀐 비교점이 시작점과 동일하다면, 한바퀴 돌아온 것이다. 반복을 종료한다.
                if startPointX == nextPointX and startPointY == nextPointY:
                    if samePointCount:
                        break
                    else:
                        samePointCount = True
                
                elif moveCount > 100000:
                    break

                # 바뀐 비교점이 시작점과 다르다면, 다음번 진행할 방향값을 반시계 방향으로 90도 회전하고, directCount = 0 으로 한다.
                else:
                    moveFlag = False
                    direction -= 2
                    directionCount = 0

                    cv.line(imgBlack, (nextPointX, nextPointY), (nextPointX, nextPointY), (0, 0, 255), 1)                    

            # 비교점이 바뀌지 않았다면, 다음에 비교할 방향을 +1하고, directCount도 +1 한다.
            else:
                direction += 1
                directionCount += 1

        # 8방향을 모두 검색했으나, nextPoint를 찾을 수 없었다면, 반복을 종료한다.
        else:
            break

