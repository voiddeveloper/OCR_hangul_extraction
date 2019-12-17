import math
import cv2 as cv
import numpy as np
import time

# 목표 : 화면에서 네모를 찾아라.
# 1. 이미지를 읽는다.
# 2. 읽어온 이미지를 HSV로 분류하고, 전체 픽셀 정보를 저장한다.
# 3. 기준 색상을 정하고, 기준색상과 다른 픽셀 값을 찾기 시작한다.
# 4. 기준 색상과 다른 픽셀 값을 찾으면 해당 픽셀과 유사한 색상 범위를 가지는 mask를 만들고, 이 mask가 적용된 이미지를 뽑는다.
# 5. mask가 적용된 이미지를 grayscale -> binary 처리한다. 이제 해당 색상만 보이는 이미지가 나온다.
# 6. 이미지 값이 존재하는 폐곡선 영역을 찾는다. 선이 너무 세밀하면 안되기 때문에 모폴로지를 적용해서 적당히 선을 뭉갠다. 
# 7. 해당 폐곡선의 꼭지점 갯수를 구한다. 꼭지점의 갯수가 4개인 것이 네모이다. 

###############################################################
# 전역 변수 모음
# 테스트할 파일 이름
testFile = 'img/real_4.jpg'
# ROI 가로, 세로 길이 (ROI 내의 평균 색상을 구하는데 사용한다. 크기가 클수록 정확한 색상을 찾기 힘들다. 대신 계산량이 줄어든다.)
# roiLength = 5
# mask 생성 시 H값의 범위 
# maskDistance = 30
# 모폴로지 적용할 kernel 크기 (커널 사이즈가 작을수록 인근한 색상을 하나의 집합으로 묶는다.)
# kernelSize = 5
# 외곽선을 단순화하기 위한 epsilon 계수 (적을수록 세밀하게, 클수록 뭉뚝하게 외곽선을 그리게 된다.)
# epsilonCount = 0.015
# 색상의 유사도 거리 (거리가 가까울수록 인접한 색상이다.)
# ranges = 20
# 찾은 사각형 카운트 수
# rectangleCount = 0
###############################################################

# 2개의 색상값을 입력 받을 시, 2개의 색상 유사도 거리값 구하는 메소드
def findTwoColorDistanceHSV(hsvColor1, hsvColor2):
    # d = sqrt{(h1-h2)^2 + (s1-s2)^2 + (v1-v2)^2}
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

    distance = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2)) 
    return distance

# 입력 받은 색상을 기준으로 mask 생성하는 메소드
def makeMaskImg(h, s, v):
    # 선택된 색상을 기준으로 비슷한 색상 lower, upper 범위 설정
    # maskDistance = cv.getTrackbarPos('maskDistance', 'imgResult')
    # s = cv.getTrackbarPos('mask_hsv_S', 'imgResult')
    # v = cv.getTrackbarPos('mask_hsv_V', 'imgResult')

    if h < maskDistance:
        lowerBlueA1 = np.array([h - maskDistance + 180, s, v])
        upperBlueA1 = np.array([180, 255, 255])
        lowerBlueA2 = np.array([0, s, v])
        upperBlueA2 = np.array([h, 255, 255])
        lowerBlueA3 = np.array([h, s, v])
        upperBlueA3 = np.array([h + maskDistance, 255, 255])

    elif h > 180 - maskDistance:
        lowerBlueA1 = np.array([h, s, v])
        upperBlueA1 = np.array([180, 255, 255])
        lowerBlueA2 = np.array([0, s, v])
        upperBlueA2 = np.array([h + maskDistance - 180, 255, 255])
        lowerBlueA3 = np.array([h - maskDistance, s, v])
        upperBlueA3 = np.array([h, 255, 255])

    else:
        lowerBlueA1 = np.array([h, s, v])
        upperBlueA1 = np.array([h + maskDistance, 255, 255])
        lowerBlueA2 = np.array([h - maskDistance, s, v])
        upperBlueA2 = np.array([h, 255, 255])
        lowerBlueA3 = np.array([h - maskDistance, s, v])
        upperBlueA3 = np.array([h, 255, 255])

    # 범위값으로 HSV에서 마스크 생성
    imgMaskA1 = cv.inRange(imgColorHsv, lowerBlueA1, upperBlueA1)
    imgMaskA2 = cv.inRange(imgColorHsv, lowerBlueA2, upperBlueA2)
    imgMaskA3 = cv.inRange(imgColorHsv, lowerBlueA3, upperBlueA3)
    temp = cv.bitwise_or(imgMaskA1, imgMaskA2)
    imgMaskResult = cv.bitwise_or(imgMaskA3, temp)

    # 모폴로지 적용
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    imgMaskResult = cv.morphologyEx(imgMaskResult, cv.MORPH_OPEN, kernel)
    imgMaskResult = cv.morphologyEx(imgMaskResult, cv.MORPH_CLOSE, kernel)

    # 최종적으로 비슷한 색만 적용된 이미지 추출
    imgResult = cv.bitwise_and(imgColorBGR, imgColorBGR, mask=imgMaskResult)

    # cv.imshow('mask', imgResult)
    return imgResult

# 입력받은 HSV 이미지를 이진화하기
def changeBinaryImage(imageColor):
    imgGrayscale = cv.cvtColor(imageColor, cv.COLOR_BGR2GRAY)
    ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    # cv.imshow('color', imageColor)
    # cv.imshow('gray', imgGrayscale)
    # cv.imshow('binary', imgBinary)
    # cv.waitKey(0)
    
    return imgBinary

# 외곽선 정보 찾기
def findContoursInfo(originalImage, imageColor, imageBinary):
    # contours = 동일한 색을 가지고 있는 영역의 경계선 정보
    # RETR_EXTERNAL = contours 정보 중에서 바깥쪽 라인만 찾는다.
    # CHAIN_APPROX_SIMPLE = contours 라인을 그릴 수 있는 포인트를 반환
    contours, hierarchy = cv.findContours(imageBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for count in contours:
        # countours의 꼭지점 갯수
        # epsilon = 근사 정확도, 외곽선 길이값 구하기 위한 값, 외곽선이 닫힌 폐외곽선 기준
        epsilon = epsilonCount * cv.arcLength(count, True)
        # approx = epsilon값에 따라 꼭지점 수를 줄여서 새로운 도형을 반환
        approx = cv.approxPolyDP(count, epsilon, True)

        # 반환된 도형의 꼭지점 갯수
        size = len(approx)
        # 네모는 꼭지점이 4개다. 4개일때만 체크한다.
        if cv.isContourConvex(approx):
            if size == 4:
                cv.line(originalImage, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
                for k in range(size - 1):
                    cv.line(originalImage, tuple(approx[k][0]), tuple(approx[k + 1][0]),(0, 255, 0), 3)
                
                global rectangleCount
                rectangleCount += 1
                setLabel(originalImage, str(rectangleCount), count)
    
    # cv.imshow('imageColor',imageColor)
    # cv.imshow('imageBinary',imageBinary)
    # cv.imshow('test',originalImage)
    # cv.waitKey(0)
    return originalImage

def setLabel(image, str, contour):
    (textWidth, textHeight), baseLine = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x, y, width, height = cv.boundingRect(contour)
    ptX = x + int((width - textWidth) / 2)
    ptY = y + int((height + textHeight) / 2)
    cv.rectangle(image, (ptX, ptY + baseLine), (ptX + textWidth, ptY - textHeight), (200, 200, 200), cv.FILLED)
    cv.putText(image, str, (ptX, ptY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, 8)

def refresh(x):
    pass
    

# start_vect=time.time()
cv.namedWindow('result')
 
# cv.createTrackbar('roiLength', 'result', 0, 10, refresh)
# cv.setTrackbarPos('roiLength', 'result', 5)

# cv.createTrackbar('maskDistance', 'result', 0, 50, refresh)
# cv.setTrackbarPos('maskDistance', 'result', 1)

# cv.createTrackbar('kernelSize', 'result', 0, 20, refresh)
# cv.setTrackbarPos('kernelSize', 'result', 2)

# cv.createTrackbar('epsilonCount', 'result', 0, 1000, refresh)
# cv.setTrackbarPos('epsilonCount', 'result', 15)

# cv.createTrackbar('ranges', 'result', 0, 50, refresh)
# cv.setTrackbarPos('ranges', 'result', 35)

cv.createTrackbar('roiLength', 'result', 0, 10, refresh)
cv.setTrackbarPos('roiLength', 'result', 5)

cv.createTrackbar('maskDistance', 'result', 0, 50, refresh)
cv.setTrackbarPos('maskDistance', 'result', 10)

cv.createTrackbar('kernelSize', 'result', 0, 20, refresh)
cv.setTrackbarPos('kernelSize', 'result', 3)

cv.createTrackbar('epsilonCount', 'result', 0, 180, refresh)
cv.setTrackbarPos('epsilonCount', 'result', 175)

cv.createTrackbar('ranges', 'result', 0, 100, refresh)
cv.setTrackbarPos('ranges', 'result', 15)

while(True):
    rectangleCount = 0
    # 1. 이미지를 BGR로 읽어오기
    imgColorBGR = cv.imread(testFile, cv.IMREAD_COLOR)
    copyImage = cv.imread(testFile, cv.IMREAD_COLOR)

    # 2. 읽어온 이미지를 BGR -> HSV로 변경
    imgColorHsv = cv.cvtColor(imgColorBGR, cv.COLOR_BGR2HSV)
    # 가로, 세로 길이 구하기
    height, width = imgColorBGR.shape[:2]
    # print(f'height={height}', f'width={width}')

    # 2. HSV 모든 픽셀 정보를 저장
    pixelInfo = [[0 for x in range(width)] for y in range(height)]
    for i in range(0, height):
        for j in range(0, width):
            pixelInfo[i][j] = imgColorHsv[i, j]

    # 3. 기준 색상 정하기 (좌측 상단 시작점)
    baseColor = pixelInfo[0][0]

    distanceArray = []
    savePixelList = []
    savePixelList.append(baseColor)

    j = 0

    # 3. 기준 색상과 다른 픽셀을 찾는다.
    while j < height:
        i = 0
        while i < width:
            # 비교군 색상을 정하고, 기준 색상과 비교군 색상의 색상 유사도 거리값을 구한다.
            comparisonColor = pixelInfo[j][i]        
            arraySize = len(savePixelList)
            
            # 저장된 픽셀 정보와 현재 비교할 색상의 유사도 값을 전부 비교한다.
            # 모든 비교값이 ranges보다 커야 한다. 하나라도 작다면 이미 사전에 비교한 색상이다.
            flag = True
            for k in range(0, arraySize):
                distance = findTwoColorDistanceHSV(savePixelList[k], comparisonColor)
                ranges = cv.getTrackbarPos('ranges', 'result')

                if ranges >= distance:
                    flag = False

            # 위에서 비교한 결과 모든 비교값이 ranges보다 크다면 (flag == true라면)
            # 만약 유사도 거리가 큰 색상을 찾았다면, 마스크를 생성한다.
            if flag == True:
                # 일단 이 색상은 비교할 색상으로 인정되었기 때문에, 픽셀 정보 배열에 저장한다.
                savePixelList.append(comparisonColor)
                # 주변 근접한 색상을 묶어서 ROI 생성
                # 비교대상 hsv 색상을 뽑는 메소드, ROI 크기를 설정하고 해당 크기 내의 평균값을 뽑는다.
                roiLength = cv.getTrackbarPos('roiLength', 'result')

                roi = imgColorBGR[j:j + roiLength, i: i + roiLength]
                roi = cv.medianBlur(roi, 3)
                hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                h, s, v =cv.split(hsv)
                h = h.mean()
                s = s.mean()
                v = v.mean()
                h = int(h)
                s = int(s)
                v = int(v)

                # 생성된 ROI를 기반으로 하는 mask 생성
                maskDistance = cv.getTrackbarPos('maskDistance', 'result')
                kernelSize = cv.getTrackbarPos('kernelSize', 'result')
                epsilonCount = cv.getTrackbarPos('epsilonCount', 'result')
                epsilonCount = float(epsilonCount)/1000

                resultImage = makeMaskImg(h, s, v)
                resultImageBinary = changeBinaryImage(resultImage)
                findContoursInfo(copyImage, resultImage, resultImageBinary)
            i += 5
        j += 5
            
    # print(savePixelList)
    cv.imshow('result', copyImage)
    print(rectangleCount)
    cv.waitKey(0)

