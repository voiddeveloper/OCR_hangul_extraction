import glob
import cv2 as cv
import random


images = glob.glob("hwang/imgSet/standard/*.*")
# images = glob.glob("C:\\Users\\narun\\Desktop\\malgun\\*.*")
count = 0
size = len(images)
randomCount = random.randrange(1, size)
tempMaxRate = 0
tempMinRate = 1

for fname in images:    
    imgStandard = cv.imread(fname)
    imgGrayscale = cv.cvtColor(imgStandard, cv.COLOR_BGR2GRAY)
    ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    boxPoint = []
    for i in range(len(contours)):
        # contours로 찾아낸 물체
        cnt = contours[i]
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        # 찾아낸 영역의 x,y,w,h값 저장
        boxPoint.append(cv.boundingRect(cnt))

    # 0번째 contour의 종횡비
    whRate = round((boxPoint[0][2] / boxPoint[0][3]), 2)
    contourCount = len(contours)
    print('name = ', fname, 'rate = ', whRate, 'count = ', contourCount)
    

    # 테두리 치기
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
    ###############################################
    # 종횡비 구하기
    whRate = round(fullImageWidth/fullImageHeight, 2)
    print(whRate)
    # if tempMinRate > whRate:
    #     tempMinRate = whRate
    # if tempMaxRate < whRate:
    #     tempMaxRate = whRate
    ###############################################
    
    cv.imshow('imgCut', imgCut)
    cv.waitKey(0)
    