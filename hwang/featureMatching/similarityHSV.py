import cv2 as cv
import numpy as np
import math
import queue
import time

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
            
    # 유사한색으로 묶인 pixel 집합
    pixelList = []
    pixelPositionList = []

    # 유사한 색이라고 인정하는 색의 거리, findTwoColorDistanceHSV 메소드의 distance 값에 이용된다.
    similarityDistance = 10
 
    # 최초 시작점은 (0, 0)부터 시작해서 주변의 비슷한 색깔이 있는지 비교한다.
    # 이후에는 이전에 비슷한 색깔로 묶었던 영역은 제외하고, 나머지 영역 중에서 다시 반복한다.
    i = 0
    j = 0
    while True:
        # 이전에 이미 pixelList가 잡힌 부분이 있다면, 그 부분은 빼야 한다.
        if len(pixelPositionList) > 0:
            # 그림 전체 영역에서 비슷한 색으로 묶은 영역을 제거한다.
            # print('pixelPosition = ', pixelPosition)
            # print('pixelPositionList = ', pixelPositionList)
            pixelPosition = list(set(pixelPosition) - set(pixelPositionList))

            # 제거를 했는데 아직 그림 영역이 남아있다면, 남은 영역 중 최소점을 기준으로 잡고 주변의 비슷한 색깔이 있는지 찾는다.
            if len(pixelPosition) > 0:
                pixelPosition.sort()
                i = pixelPosition[0][0]
                j = pixelPosition[0][1]
                # print('i = ', i, 'j = ', j)
            
            # 그림 영역이 하나도 없다면, 모든 영역을 체크했다는 것이다. 검사를 종료한다.
            else:
                break

        # 자신의 좌표를 중심으로 상하좌우 픽셀의 HSV 거리값을 구한다.
        pixelLink = []
        heightLinkFlag = False
        widthLinkFlag = False
        savePoint = []

        # 최소한 자기 자신은 무조건 포함된다.
        pixelLink.append([i, j])

        # 초기 시작은 (0, 0) 이기 때문에 위쪽, 좌측은 비교할 필요가 없다.
        # 이후 이 작업이 반복되더라도, 자신 기준 위쪽과 좌측은 이미 비교한 영역이기 때문에 확인할 필요가 없다.
        # # 위쪽 좌표 비교
        # if j > 0:
        #     if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i][j - 1]):
        #         pixelLink.append([i, j - 1])
        #         savePoint.append([i, j - 1])
        # # 좌측 좌표 비교
        # if i > 0:
        #     if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i - 1][j]):
        #         pixelLink.append([i - 1, j])
        #         savePoint.append([i - 1, j])

        # 우측 좌표 비교, 비슷한 색상이라고 판단되면 해당 좌표를 저장한다.
        if i < width - 1:
            if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i + 1][j]):
                pixelLink.append([i + 1, j])
                savePoint.append([i + 1, j])
        
        # 아래쪽 좌표 비교, 비슷한 색상이라고 판단되면 해당 좌표를 저장한다.
        if j < height - 1:
            if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[i][j], pixelInfoHSV[i][j + 1]):
                pixelLink.append([i, j + 1])
                savePoint.append([i, j + 1])

        # savePoint 중에서 중복된 좌표를 제거한다.
        savePoint = list(set(map(tuple, savePoint)))
        # print('pixelLink = ', pixelLink, 'savePoint = ', savePoint)

        # savePoint는 비슷한 색깔로 인정된 좌표이기 때문에, 해당 좌표 기준으로 다시 주변 색깔을 검색해야 한다.
        # savePoint가 존재하는 한 무한 반복한다.
        if len(savePoint) > 0:
            while True:
                count = len(savePoint)

                # savePoint의 좌표를 구한다.
                for k in range(count):
                    # print('k = ', k)
                    tempX = savePoint[k][0]
                    tempY = savePoint[k][1]
                    
                    # savePoint의 상/하/좌/우 픽셀의 HSV 거리값을 구한다.
                    # 거리값이 유사한 색의 거리값으로 나오면, 해당 좌표를 저장하고 savePoint로 등록한다.
                    if tempX > 0:
                        if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[tempX][tempY], pixelInfoHSV[tempX - 1][tempY]):
                            # 위쪽 좌표는 중복될 수 있다. 중복된 좌표는 제거한다.
                            if [tempX - 1, tempY] in pixelLink:
                                None
                            else:
                                pixelLink.append([tempX - 1, tempY])
                                savePoint.append([tempX - 1, tempY])

                    if tempY > 0:
                        if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[tempX][tempY], pixelInfoHSV[tempX][tempY - 1]):
                            # 왼쪽 좌표는 중복될 수 있다. 중복된 좌표는 제거한다.
                            if [tempX, tempY - 1] in pixelLink:
                                None
                            else:
                                pixelLink.append([tempX, tempY - 1])                        
                                savePoint.append([tempX, tempY - 1])
                            
                    if tempX < width - 1:
                        if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[tempX][tempY], pixelInfoHSV[tempX + 1][tempY]):
                            pixelLink.append([tempX + 1, tempY])
                            savePoint.append([tempX + 1, tempY])
            
                    if tempY < height - 1:
                        if similarityDistance >= findTwoColorDistanceHSV(pixelInfoHSV[tempX][tempY], pixelInfoHSV[tempX][tempY + 1]):
                            pixelLink.append([tempX, tempY + 1])
                            savePoint.append([tempX, tempY + 1])

                # print('pixelLink = ', pixelLink, 'savePoint = ', savePoint)

                # 주변 영역을 체크한 savePoint를 삭제한다.
                for k in range(count):
                    del savePoint[0]

                savePoint = list(set(map(tuple, savePoint)))
                # print('pixelLink = ', pixelLink, 'savePoint = ', savePoint)
                # print('saveLength = ', len(savePoint), 'savePoint = ', savePoint)
                # savePoint가 하나도 없다면 계산을 그만한다.
                if len(savePoint) == 0:
                    break

        # 유사한 색의 집합으로 묶인 PixelLink 중에서 중복 좌표를 제거한 다음, 정렬한다.
        pixelLink = list(set(map(tuple, pixelLink)))
        pixelLink.sort()
        # print('pixelLink = ', pixelLink, 'savePoint = ', savePoint)

        # 해당 집합을 pixelList에 저장한다.
        pixelList.append(pixelLink)

        # 비슷한 색이라고 인정된 픽셀의 좌표를 저장한다. 
        # 그림 전체 영역에서 비슷한 색으로 묶은 영역을 제거하는 용도로 사용한다.
        for k in range(len(pixelList)):
            for l in range(len(pixelList[k])):
                pixelPositionList.append(pixelList[k][l])

    # print(pixelList)
    return len(pixelList)



def newPixelLinkList(img):
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
    similarityDistance = 40

    i = 0
    j = 0

    # 그림 전체 기준에서 비슷한 색깔끼리 하나의 집합으로 묶이는 리스트
    pixelList = []

    while True:
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

    # print(pixelList)
    for k in range(0, len(pixelList[0])):
        cv.line(imgOriginal, (pixelList[0][k][0], pixelList[0][k][1]), (pixelList[0][k][0], pixelList[0][k][1]), (0, 0, 255), 1)
    
    print('pixelList 전체 묶음 갯수 : ', len(pixelList))
    print('1번째 pixel 묶음 pixel 갯수 : ', len(pixelList[0]))
    # for k in range(0, len(pixelList)):
    #     print(len(pixelList[k]))

    cv.imshow("test", imgOriginal)
    cv.waitKey(0)
    

    return len(pixelList)

startTime = time.time()

img = cv.imread('hwang/imgSet/20200203/31.jpg')
count = newPixelLinkList(img)
# count = pixelLinkList(img)
print(count)
print('time : ', time.time() - startTime)
