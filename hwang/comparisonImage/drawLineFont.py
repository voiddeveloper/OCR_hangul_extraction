import cv2 as cv
import glob
import matplotlib.pyplot 

#############################################################################################
# 폰트 이미지를 읽고, 해당 폰트를 n등분하여 선을 긋는 프로그램
# 폰트 이미지의 종횡비도 구한다.
#############################################################################################
images = glob.glob("C:\\Users\\narun\\Desktop\\malgun\\*.*")
count = 0
size = len(images)
tempMaxRate = 0
tempMinRate = 1

for fname in images:
    count += 1
    # '운' = 종횡비 : 1.0
    # '훈' = 종횡비 : 1.0
    if fname == 'C:\\Users\\narun\\Desktop\\malgun\\10953.png':
    # if count < size:
        print(fname)
        imgStandard = cv.imread(fname)
        imgResize = cv.resize(imgStandard, (0, 0), fx = 1, fy = 1, interpolation= cv.INTER_AREA)
        
        imgGrayscale = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)
        ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        boxPoint = []
        for i in range(len(contours)):
            # contours로 찾아낸 물체
            cnt = contours[i]
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

        ###############################################
        # 종횡비 구하기
        # 맑은 고딕 기준
        # 종횡비 세로가 가장 긴 비율 (0.7 (width / height))
        # 해당 글자 : 냬, 랚, 럑, 렦, 릮
        # 종횡비 가로가 가장 긴 비율 (1.286 (width / height))
        # 해당 글자 : 뚀, 쬬
        whRate = round(width/height, 3)
        print(whRate)
        if tempMinRate > whRate:
            tempMinRate = whRate
        if tempMaxRate < whRate:
            tempMaxRate = whRate
        ###############################################
        # n등분해서 선 그리기
        # divisionCount = 3
        # for i in range(0, 2):
        #     for j in range(0, divisionCount):
        #         if j > 0:
        #             if i == 0:
        #                 cv.line(imgResize, (j * int(width/divisionCount), 0), (j * int(width/divisionCount), height), (0, 0, 255), 2)
        #             if i == 1:
        #                 cv.line(imgResize, (0, j * int(height/divisionCount)), (width, j * int(height/divisionCount)), (0, 0, 255), 2)
        ###############################################
        
        cv.imshow('imgCut', imgResize)
        cv.waitKey(0)
print(tempMinRate, tempMaxRate)