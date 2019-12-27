import cv2 as cv

####################################################
# 제어중인 전역 변수
# 인접한 contour를 묶어야 하는 범위값 (해당 contour의 width, height 중 작은 값의 1/combineDistance)
combineDistance = 0.5
####################################################

# image에 글씨를 출력하는 메소드, 글씨를 출력할 image, 입력할 글씨, 입력될 위치(contour)를 매개변수로 받는다.
def setLabel(image, str, contour):
    (textWidth, textHeight), baseLine = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x, y, width, height = cv.boundingRect(contour)
    ptX = x + int((width - textWidth) / 2)
    ptY = y + int((height + textHeight) / 2)
    cv.rectangle(image, (ptX, ptY + baseLine), (ptX + textWidth, ptY - textHeight), (200, 200, 200), cv.FILLED)
    cv.putText(image, str, (ptX, ptY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, 8)

# min, max값을 찾아주는 메소드
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

def findFontContour(imgFile):
    imgStandard = cv.imread(imgFile, cv.IMREAD_COLOR)
    imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)
    ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rectangleCount = 0
    boxPoint = []
    # 확인된 font contour position 값
    fontContourPosition = []
    # 최종적으로 인정된 font Contour 좌표(x, y, w, h)
    fontContour = []

    for i in range(len(contours)):
        # contours로 찾아낸 물체
        cnt = contours[i]
        x, y, w, h = cv.boundingRect(cnt)
        # 찾아낸 영역의 x,y,w,h값 저장
        boxPoint.append(cv.boundingRect(cnt))

        # 테투리로 영역 설정하기
        # cv.drawContours(imgStandard, [cnt], 0, (0, 255, 0), 2)
        # 사각형으로 영역 설정하기 (녹색)
        # cv.rectangle(imgStandard, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # setLabel(imgStandard, str(rectangleCount), cnt)
        # rectangleCount += 1

    # 인접한 contour 합치기
    for i in range(len(boxPoint)):
        # 이미 font contour로 확인된 값이라면 패스한다.
        if i in fontContourPosition:
            None
        # 확인되지 않은 contour이면 합치기 작업을 시작한다.
        else:
            # 폰트 font contour 종횡비 최소/최대값은 0.7 ~ 1.286다. (자세한 내용은 imagePositionCheck.py 확인)
            ###################################################
            # contour 주변에 인접한 contour가 있다면 합친다.
            # 합친 contour의 종횡비가 0.7 ~ 1.286면 font contour로 인정한다.
            # 만약 합친 contour의 종횡비가 0.7 ~ 1.286가 아닌데,
            # 주변을 합치지 않은 contour 종횡비가 0.7 ~ 1.286면 font contour로 인정한다.
            ###################################################
            whRate = boxPoint[i][2] / boxPoint[i][3]

            # 자신 주변에 있는 contour를 찾는다.
            # 주변 합칠 범위: width, height 중 작은 값 * combineDistance
            if boxPoint[i][2] > boxPoint[i][3]:
                distanceRange = int(boxPoint[i][3] * combineDistance)
            else:
                distanceRange = int(boxPoint[i][2] * combineDistance)

            # 종횡비가 극단적으로 세로가 길 때 (ㅣ,ㅏ 등)
            if whRate < 0.25:
                # 주변 합쳐야할 영역 (좌측으로만 검사한다.)
                x = boxPoint[i][0] - (distanceRange * 3)
                y = boxPoint[i][1]
                w = boxPoint[i][2] + (distanceRange * 3)
                h = boxPoint[i][3]
            # 종횡비가 가로가 길 때 (ㅜ, ㅠ 등)
            elif 10 > whRate > 2.85:
                # 주변 합쳐야할 영역 (위 아래로만 검사한다.)
                x = boxPoint[i][0]
                y = boxPoint[i][1] - (distanceRange * 2)
                w = boxPoint[i][2]
                h = boxPoint[i][3] + (distanceRange * 4)
            # 종횡비가 극단적으로 가로가 길 때 (ㅡ 등)
            elif whRate >= 10:
                # 주변 합쳐야할 영역 (위 아래로만 검사한다. 위 아래폭을 좀 더 크게 잡는다.)
                x = boxPoint[i][0]
                y = boxPoint[i][1] - (boxPoint[i][3] * 2)
                w = boxPoint[i][2]
                h = boxPoint[i][3] + (boxPoint[i][3] * 4)
            else:
                # 주변 합쳐야할 영역
                x = boxPoint[i][0] - distanceRange
                y = boxPoint[i][1] - distanceRange
                w = boxPoint[i][2] + (distanceRange * 2)
                h = boxPoint[i][3] + (distanceRange * 2)

            # 찾아야할 영역 표시(빨강)
            # cv.rectangle(imgStandard, (x,y), (x+w, y+h), (0, 0, 255), 1)

            # (x, y, w, h) 영역에 겹쳐있는 모든 contour를 찾아낸다.
            combineList = []
            tempPosition = []
            for j in range(len(boxPoint)):
                # 이미 찾아낸 font contour는 제외한다.
                if j in fontContourPosition:
                    None
                else:
                    comparisonX = boxPoint[j][0]
                    comparisonY = boxPoint[j][1]
                    comparisonWidth = boxPoint[j][2]
                    comparisonHeight = boxPoint[j][3]

                    if comparisonX < x+w and comparisonX + comparisonWidth > x:
                        if comparisonY < y+h and comparisonY + comparisonHeight > y:
                            combineList.append((comparisonX, comparisonY, comparisonWidth, comparisonHeight))
                            tempPosition.append(j)
                            # 합칠 영역 표시 (파랑)
                            # cv.rectangle(imgStandard, (comparisonX, comparisonY), (comparisonX + comparisonWidth, comparisonY + comparisonHeight), (255, 0, 0), 1)
        
            # 겹쳐진 contour를 모두 합쳐서 종횡비를 계산한다. 0.7 ~ 1.286에 포함되면 font contour로 인정한다.
            minMaxPoint = findMinMaxPoint(combineList)
            mWidth = minMaxPoint[2] - minMaxPoint[0]
            mHeight = minMaxPoint[3] - minMaxPoint[1]

            mWHRate = round((mWidth/mHeight), 2)
            if mWHRate >= 0.7 and mWHRate <= 1.286:
                # 인정된 contour position은 realContourPosition에 저장한다.
                for temp in tempPosition:
                    fontContourPosition.append(temp)
                # 인정된 contour 영역 안에 다른 contour들이 있는지 확인한다. 있다면 전부 포함한다.
                for l in range(len(boxPoint)):
                    # 이미 확인된건 검사하지 않는다.
                    if l in fontContourPosition:
                        None
                    else:
                        comparisonX = boxPoint[l][0]
                        comparisonY = boxPoint[l][1]
                        comparisonWidth = boxPoint[l][2]
                        comparisonHeight = boxPoint[l][3]
                        
                        # 영역 안에 contour가 존재하면 해당 position을 realContourPosition에 저장한다.
                        if comparisonX < minMaxPoint[2] and comparisonX + comparisonWidth > minMaxPoint[0]:
                            if comparisonY < minMaxPoint[3] and comparisonY + comparisonHeight > minMaxPoint[1]:
                                # 인정된 contour값이 변경되면, 다시 계산해야 한다.
                                # 새로 추가된 contour를 combineList에 추가한 후, minMaxPoint를 구한다.
                                combineList.append((comparisonX, comparisonY, comparisonWidth, comparisonHeight))
                                minMaxPoint = findMinMaxPoint(combineList)

                                fontContourPosition.append(l)
                
                cv.rectangle(imgStandard, (minMaxPoint[0], minMaxPoint[1]), (minMaxPoint[2], minMaxPoint[3]), (50,100,150), 1)
                fontContour.append((minMaxPoint[0], minMaxPoint[1], minMaxPoint[2] - minMaxPoint[0], minMaxPoint[3] - minMaxPoint[1]))
                # print("realContourPosition ", realContourPosition)
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
            cv.rectangle(imgStandard, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (50,100,150), 1)

    cv.imshow('result', imgStandard)
    cv.waitKey(0)

    return fontContour

standardFile = 'hwang/imgSet/test_comparison2.png'
contourList = findFontContour(standardFile)
print(contourList)
