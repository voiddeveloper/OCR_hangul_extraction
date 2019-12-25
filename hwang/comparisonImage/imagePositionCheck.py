import cv2 as cv
import os
import glob
import random

# def listFilesSubDir(destpath):
#     filelist = []
#     for path, subdirs, files in os.walk(destpath):
#         for filename in files:
#             f = os.path.join(path, filename)
#             if os.path.isfile(f):
#                 filelist.append(f)
    
#     return filelist

# fileList = listFilesSubDir("C:\\Users\\narun\\Desktop\\malgun")

images = glob.glob("C:\\Users\\narun\\Desktop\\malgun\\*.*")
count = 0
size = len(images)
randomCount = random.randrange(1, size)

for fname in images:
    count += 1
    if count < randomCount and count > randomCount - 5:
        
        imgStandard = cv.imread(fname)
        imgResize = cv.resize(imgStandard, (0, 0), fx = 4, fy = 4, interpolation= cv.INTER_AREA)
        
        imgGrayscale = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)
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

        imgCut = imgResize[minY:maxY, minX:maxX]
        
        width = maxX - minX
        height = maxY - minY
        divisionCount = 3
        for i in range(0, 2):
            for j in range(0, divisionCount):
                if j > 0:
                    if i == 0:
                        cv.line(imgResize, (j * int(width/divisionCount), 0), (j * int(width/divisionCount), height), (0, 0, 255), 2)
                    if i == 1:
                        cv.line(imgResize, (0, j * int(height/divisionCount)), (width, j * int(height/divisionCount)), (0, 0, 255), 2)
        

        cv.imshow('imgCut', imgResize)
        cv.waitKey(0)