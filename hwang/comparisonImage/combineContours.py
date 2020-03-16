import cv2 as cv

#############################################################################################
# 이미지에서 글씨 덩어리를 찾아내는 코드
# imagePositionCheck.py 작업을 이어받아 작업한다.
# 모든 contour들간 위치 연관성을 찾는 작업은 데이터가 너무 많이 필요하다.
# 그렇다면, 각각의 특징을 찾지 말고, 모든 글씨의 공통적인 부분만 체크해보자.

# 이용할 수 있는 데이터는 폰트의 종횡비는 0.7 ~ 1.286 범위 내 라는 사실이다.

# 추측 1: 글씨는 자음과 모음이 매우 밀접하게 뭉쳐있다. 그렇다면 특정 contour의 근접한 영역을 무조건 묶어 보자.
# 추측 2: 글씨를 우하단부터 검색할 경우, 받침이 없다면 무조건 모음부터 검색할 수 있다. 받침이 있어도, 일정 범위만큼만 위로 올리면 모음이 나온다.
# 이렇게 모음을 찾아낸 다음, 주변에 자음이 있을법한 방향을 검색한다. 이 방향에 contour가 있다면 하나로 묶은 후, 폰트 종횡비(0.7 ~ 1.286)를 비교해본다.

# ## 목표 ##
# 1. 이미지에서 contour를 찾아낸다.
# 2. 찾아낸 contour의 종횡비를 구해본다. 가로 비율이 매우 길거나, 세로 비율이 매우 길다면 그것은 '모음'일 수 있다.
# 3. 모음의 특징별 특정 방향에 contour가 있는지 검색한다.
#   3-1. 세로로 긴 모음('ㅏ', 'ㅕ' 등) 이면, 좌측 영역만 탐색한다.
#   3-2. 가로로 긴 모음('ㅜ', 'ㅡ' 등) 이면, 위, 아래 영역만 탐색한다.
#   3-3. 조합형 모음('ㅘ', 'ㅝ' 등) 이면, 이 모음 범위 안에 자음이 자동으로 포함되니 주변을 탐색하지 않아도 된다.
# 4. 주변에 contour 가 있다면 하나로 묶는다. 이후 다시 주변에 contour가 있는지 찾아본다.
# 5. 밀접한 contour들을 전부 묶은 다음, 종횡비를 계산한다. 만약 종횡비가 (0.7 ~ 1.286) 범위 내라면 이것은 폰트일 것이다.
# 6. 종횡비가 (0.7 ~ 1.286) 범위가 아니라면, 이전 contour 묶음을 한단계 전으로 되돌려서 다시 종횡비를 계산한다. (묶어서는 안되는 다른 contour를 묶었을 수도 있기 때문이다.)
# 7. 이 과정을 계속 반복한다.
#############################################################################################

## 제어중인 전역 변수 ##
# 인접한 contour를 묶어야 하는 거리 값
combineDistance = 0.5

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

# 이미지에서 글자를 찾는 메소드
def findFontContour(imgFile):
    # 이미지 읽기 및 contour 정보 얻기
    imgStandard = cv.imread(imgFile, cv.IMREAD_COLOR)
    imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)
    ret, imgBinary = cv.threshold(imgGrayscale, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 글씨라고 인정된 contour position 값
    fontContourPosition = []
    # 글씨라고 인정된 contour 정보(x, y, w, h)
    # x: contour의 시작점 x 좌표
    # y: contour의 시작점 y 좌표
    # w: contour의 가로 길이
    # h: contour의 세로 길이
    fontContour = []

    # contour 정보(x, y, w, h)를 boxPoint에 저장한다.
    # x: contour의 시작점 x 좌표
    # y: contour의 시작점 y 좌표
    # w: contour의 가로 길이
    # h: contour의 세로 길이
    boxPoint = []

    # contour 정보 구하기
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        boxPoint.append(cv.boundingRect(cnt))

        # 찾아낸 contour 테두리 그리기
        # cv.drawContours(imgStandard, [cnt], 0, (0, 255, 0), 2)

        # 찾아낸 contour 박스 그리기 (녹색)
        # cv.rectangle(imgStandard, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # 인접한 contour 합치기
    for i in range(len(boxPoint)):
        # 이미 font contour로 확인된 값이라면 합치지 않는다.
        if i in fontContourPosition:
            None
        # font contour로 인정되지 않은 contour이면 합치기 작업을 시작한다.
        else:
            # 폰트 font contour 종횡비 최소/최대값은 0.7 ~ 1.286다. (자세한 내용은 imagePositionCheck.py 확인)
            ###################################################
            # contour 주변에 인접한 contour가 있다면 합친다.
            # 합친 contour의 종횡비가 0.7 ~ 1.286면 font contour로 인정한다.
            # 만약 합친 contour의 종횡비가 0.7 ~ 1.286가 아닌데,
            # 주변을 합치지 않은 contour 종횡비가 0.7 ~ 1.286면, 합치기 전 contour를 font contour로 인정한다.
            ###################################################

            # whRate: contour의 종횡비
            whRate = boxPoint[i][2] / boxPoint[i][3]

            # 자신 주변에 있는 contour를 찾는다. 밀접한 contour만 탐색한다.
            # distanceRange(주변 합칠 범위): width, height 중 작은 값 * combineDistance
            if boxPoint[i][2] > boxPoint[i][3]:
                distanceRange = int(boxPoint[i][3] * combineDistance)
            else:
                distanceRange = int(boxPoint[i][2] * combineDistance)

            # contour 주변에 distanceRange만큼 영역을 넓혀서 새로운 box를 그린다.
            # 이때, contour의 특징에 따라 넓힐 방향을 다르게 해서 (x, y, w, h) 값을 구한다.
            
            # contour 종횡비가 극단적으로 세로가 길 때 (ㅣ,ㅏ 등)
            if whRate < 0.25:
                # 주변 합쳐야할 영역 (좌측으로만 검사한다.)
                x = boxPoint[i][0] - (distanceRange * 3)
                y = boxPoint[i][1]
                w = boxPoint[i][2] + (distanceRange * 3)
                h = boxPoint[i][3]
            # contour 종횡비가 가로가 길 때 (ㅜ, ㅠ 등)
            elif 10 > whRate > 2.85:
                # 주변 합쳐야할 영역 (위 아래로만 검사한다.)
                x = boxPoint[i][0]
                y = boxPoint[i][1] - (distanceRange * 2)
                w = boxPoint[i][2]
                h = boxPoint[i][3] + (distanceRange * 4)
            # contour 종횡비가 극단적으로 가로가 길 때 (ㅡ 등)
            elif whRate >= 10:
                # 주변 합쳐야할 영역 (위 아래로만 검사한다. 위 아래폭을 좀 더 크게 잡는다.)
                x = boxPoint[i][0]
                y = boxPoint[i][1] - (boxPoint[i][3] * 3)
                w = boxPoint[i][2]
                h = boxPoint[i][3] + (boxPoint[i][3] * 6)
            # 일반적인 상황일 때
            else:
                # 주변 합쳐야할 영역 (상, 하, 좌, 우 전부 검사한다.)
                x = boxPoint[i][0] - distanceRange
                y = boxPoint[i][1] - distanceRange
                w = boxPoint[i][2] + (distanceRange * 2)
                h = boxPoint[i][3] + (distanceRange * 2)

            # 찾아야 할 영역 표시(빨강)
            # cv.rectangle(imgStandard, (x,y), (x+w, y+h), (0, 0, 255), 1)

            # box 영역에 있는 모든 contour를 찾아낸다. (겹쳐있는 contour도 포함한다.)
            # 겹쳐있는 contour 정보(x, y, w, h) 리스트
            combineList = []
            # 임시 contour position값 리스트, 합친 contour 집합이 잘못되었을 경우, 이전 단계로 돌아가기 위해 이용한다.
            tempPosition = []
            
            for j in range(len(boxPoint)):
                # 이미 찾아낸 font contour는 제외한다.
                if j in fontContourPosition:
                    None
                else:
                    # 비교 대상 contour 좌표 정보
                    # comparisonX: 비교 대상 contour 시작점 x 좌표
                    # comparisonY: 비교 대상 contour 시작점 y 좌표
                    # comparisonWidth: 비교 대상 contour의 가로 길이
                    # comparisonHeight: 비교 대상 contour의 세로 길이
                    comparisonX = boxPoint[j][0]
                    comparisonY = boxPoint[j][1]
                    comparisonWidth = boxPoint[j][2]
                    comparisonHeight = boxPoint[j][3]

                    # 비교 대상 contour가 box 영역 안에 있다면 combineList에 추가한다. 그리고 추가한 contour의 position 값을 tempPosition에 기록한다.
                    if comparisonX < x+w and comparisonX + comparisonWidth > x:
                        if comparisonY < y+h and comparisonY + comparisonHeight > y:
                            combineList.append((comparisonX, comparisonY, comparisonWidth, comparisonHeight))
                            tempPosition.append(j)
                            # 합칠 영역 표시 (파랑)
                            # cv.rectangle(imgStandard, (comparisonX, comparisonY), (comparisonX + comparisonWidth, comparisonY + comparisonHeight), (255, 0, 0), 1)
        
            # 겹쳐진 contour를 모두 합쳐서 종횡비를 계산한다. 0.7 ~ 1.286에 포함되면 font contour로 인정한다.

            # minMaxPoint: combineList 전체의 시작점(x, y)과 끝점(x, y)
            # mWidth: combineList 전체의 가로 길이
            # mHeight: combineList 전체의 세로 길이
            minMaxPoint = findMinMaxPoint(combineList)
            mWidth = minMaxPoint[2] - minMaxPoint[0]
            mHeight = minMaxPoint[3] - minMaxPoint[1]

            # combineList 전체의 종횡비
            mWHRate = round((mWidth/mHeight), 3)

            if mWHRate >= 0.7 and mWHRate <= 1.286:
                # 인정된 contour position은 fontContourPosition에 저장한다.
                # tempPosition값에 저장되어 있던 모든 임시값을 fontContourPosition에 저장한다.
                for temp in tempPosition:
                    fontContourPosition.append(temp)
                
                # 글씨로 인정된 contour 내부에 작은 contour들이 추가로 있을 수도 있다. 이런 contour들도 전부 하나의 combineList로 합친다.
                for l in range(len(boxPoint)):
                    # 이미 확인된건 검사하지 않는다.
                    if l in fontContourPosition:
                        None
                    else:
                        # 비교 대상 contour 좌표 정보
                        # comparisonX: 비교 대상 contour 시작점 x 좌표
                        # comparisonY: 비교 대상 contour 시작점 y 좌표
                        # comparisonWidth: 비교 대상 contour의 가로 길이
                        # comparisonHeight: 비교 대상 contour의 세로 길이
                        comparisonX = boxPoint[l][0]
                        comparisonY = boxPoint[l][1]
                        comparisonWidth = boxPoint[l][2]
                        comparisonHeight = boxPoint[l][3]
                        
                        # 영역 안에 contour가 존재하면 해당 position을 fontContourPosition에 저장한다.
                        if comparisonX < minMaxPoint[2] and comparisonX + comparisonWidth > minMaxPoint[0]:
                            if comparisonY < minMaxPoint[3] and comparisonY + comparisonHeight > minMaxPoint[1]:
                                # combineList가 갱신되었을 경우, minMaxPoint값이 변경될 수 있기 때문에 값을 다시 계산한다.
                                combineList.append((comparisonX, comparisonY, comparisonWidth, comparisonHeight))
                                minMaxPoint = findMinMaxPoint(combineList)

                                fontContourPosition.append(l)
                
                # 최종적으로 글씨라고 인정된 영역 표시(갈색)
                cv.rectangle(imgStandard, (minMaxPoint[0], minMaxPoint[1]), (minMaxPoint[2], minMaxPoint[3]), (0,100,200), 1)
                # 글씨라고 인정된 contour들 리스트
                fontContour.append((minMaxPoint[0], minMaxPoint[1], minMaxPoint[2] - minMaxPoint[0], minMaxPoint[3] - minMaxPoint[1]))
                # cv.imshow('result', imgStandard)
                # cv.waitKey(0)

    # font contour를 전부 찾은 후, 남은 값이 있을 경우, 남은 값만 가지고 다시 한번 font contour를 찾는다.
    # 기존에 찾은 font contour는 boxPoint에서 제거한다.
    fontContourPosition.sort(reverse=True)
    for position in fontContourPosition:
        del(boxPoint[position])

    for box in boxPoint:
        if box[2] / box[3] >= 0.7 and box[2] / box[3] <= 1.286:
            fontContour.append((box[0], box[1], box[2], box[3]))
            cv.rectangle(imgStandard, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,100,200), 1)

    cv.imshow('result', imgStandard)
    cv.waitKey(0)

    return fontContour

standardFile = 'hwang/imgSet/comparisonImage/test_comparison5.png'
contourList = findFontContour(standardFile)
print(contourList)
