import glob
import cv2 as cv

############################################################################
# 글자의 특징을 찾아 보기 위한 코드
# 추측 1: 폰트마다 종횡비의 비율이 일정할 것이다.
# 추측 2: 글씨를 이루고 있는 각 contour들의 종횡비 최저/최대 비율을 알아내면, 그 범위 내의 값은 같은 글씨일 것이다.

# ## 목표 ##
# 찾아야 할 글씨만 적혀있는 이미지를 만들고, 각 contour의 종횡비 및 글씨 전체의 종횡비를 구한다.
# 여기서 구한 값을 imagePositionCheck.py에 이용할 예정이다.
############################################################################

# 종횡비를 찾아야할 모든 이미지 파일 집합
images = glob.glob("hwang/imgSet/comparisonImage/standard/*.*")

count = 0
size = len(images)

# 이미지 내에서 contour 정보를 얻는다.
for fileName in images:    
    imgStandard = cv.imread(fileName)
    imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)
    ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # contour 정보를 boxPoint에 저장한다. 저장할 값은 (x, y, w, h) 이다.
    # x: contour 시작점 x좌표
    # y: contour 시작점 y좌표
    # w: contour 가로 길이
    # h: contour 세로 길이
    boxPoint = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv.boundingRect(cnt)
        boxPoint.append(cv.boundingRect(cnt))

    # 0번째 contour의 종횡비
    widthHeightRate = round((boxPoint[0][2] / boxPoint[0][3]), 2)
    contourCount = len(contours)
    print('name = ', fileName, 'rate = ', widthHeightRate, 'count = ', contourCount)
    
    # 글씨 전체(boxPoint)의 종횡비를 구한다.
    minX = boxPoint[0][0]
    maxX = boxPoint[0][0] + boxPoint[0][2]
    minY = boxPoint[0][1]
    maxY = boxPoint[0][1] + boxPoint[0][3]

    for m in range(len(boxPoint)):
        if minX > boxPoint[m][0]:
            minX = boxPoint[m][0]

        if maxX < boxPoint[m][0] + boxPoint[m][2]:
            maxX = boxPoint[m][0] + boxPoint[m][2]
        
        if minY > boxPoint[m][1]:
            minY = boxPoint[m][1]

        if maxY < boxPoint[m][1] + boxPoint[m][3]:
            maxY = boxPoint[m][1] + boxPoint[m][3]

    imgCut = imgStandard[minY:maxY, minX:maxX]

    fullImageWidth = maxX - minX
    fullImageHeight = maxY - minY
    
    # 글씨 전체의 종횡비
    widthHeightRate = round(fullImageWidth / fullImageHeight, 2)
    print(widthHeightRate)

    

