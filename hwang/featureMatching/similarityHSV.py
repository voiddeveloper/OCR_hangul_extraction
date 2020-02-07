import cv2 as cv
import numpy as np
import math

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

def pixelLinkList(img):
    # 기본 컬러 = BGR
    imgOriginal = cv.imread(img, cv.IMREAD_COLOR)
    # BGR -> HSV로 변경
    imgHSV = cv.cvtColor(imgOriginal, cv.COLOR_BGR2HSV)

    # 픽셀값을 저장할 배열 생성
    height, width = imgOriginal.shape[:2]
    pixelInfoBGR = [[0 for x in range(height)] for y in range(width)]
    pixelInfoHSV = [[0 for x in range(height)] for y in range(width)]

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

    # BGR, HSV 픽셀값 저장
    for i in range(0, width):
        for j in range(0, height):
            pixelInfoBGR[i][j] = imgOriginal[j, i]
            pixelInfoHSV[i][j] = imgHSV[j, i]
            
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

                # print('pixelLink = ', pixelLink, 'savePoint = ', savePoint)

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

# pixelLinkList('hwang/imgSet/20200203/test.png')