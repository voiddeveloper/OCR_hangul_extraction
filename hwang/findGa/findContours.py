import cv2 as cv
import numpy as np
##############################################################################################
# '가' 라는 글씨를 찾기
# 다양한 폰트, 다양한 크기의 '가' 글씨를 모두 찾아내야 한다.
# 테스트용 이미지: ../hwang/imgSet/findGa/find_1.png, find_2.png, find_3.png
##############################################################################################

## 이미지 찾는 기준 ##
# 1. 이미지 내에서 contour를 이용해서 찾아본다.
# 2. 이 이미지에는 '가' 라는 글씨 밖에 없다. 따라서 'ㄱ' 과 'ㅏ' 2종류의 contour가 적출될 것이다.
# 3. 'ㄱ'은 좌측에, 'ㅏ'는 우측에 배치되어야 한다. 따라서 contour의 우측을 검사해서, 비슷한 크기의 contour가 있는지 검색한다.
# 4. 조건에 전부 일치한 2개의 contour 집합을 '가' 라는 글씨로 체크한다.

# 이미지 불러오기
imgOriginal = cv.imread('hwang/imgSet/findGa/find_3.png', cv.IMREAD_COLOR)

# BGR copy(결과 출력용), grayscale, binary 처리된 이미지를 각각 생성
imgCopy = imgOriginal.copy()
imgGrayscale = cv.cvtColor(imgOriginal, cv.COLOR_RGB2GRAY)
ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

# contour의 우측을 검사해야 하지만, 우측에 있다고 무조건 '가' 글씨가 완성되는 것은 아니다.
# contour 크기와 위치가 서로 비슷해야 한다.
# 크기와 위치가 비슷한 기준은 다음과 같다.
# 1. contour의 중점을 구한다.
# 2. contour의 중점에서 x좌표를 width만큼 이동했는데, 다른 contour 영역 내에 있다.
# 이 조건을 만족하면 두개의 contour를 하나의 글씨 contour로 묶어준다.

# contour 구하기
# contour: contour 좌표 집합
# hierachy: contour들의 계층 구조(부모/자식 관계) 집합
contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# contour들의 좌표 및 크기 구하기 (x, y, w, h를 구한다.)
# x: contour의 x 좌표
# y: contour의 y 좌표
# w: contour의 width(가로 길이)
# h: contour의 height(세로 길이)
# 각각의 값을 box에 저장한다.
box = []
for i in range(len(contours)):
    cnt = contours[i]
    x, y, w, h = cv.boundingRect(cnt)
    box.append(cv.boundingRect(cnt))
# print('box=', box)

# contour들의 중심 좌표 구하기
realationBox = []
for i in range(len(box)):
    # contour의 중점 구하기
    # centerX: 중점 x좌표
    # centerY: 중점 y좌표
    centerX = box[i][0] + (box[i][2] / 2)
    centerY = box[i][1] + (box[i][3] / 2)
    # print('centerX=', centerX, 'centerY=', centerY)
    
    for j in range(len(box)):
        # print('x=', box[j][0], 'x+w=', box[j][0] + box[j][2], 'y=', box[j][1], 'y+h=', box[j][1] + box[j][3])

        # contour의 중점에서 해당 contour의 width값만큼 이동했을 때, 다른 contour 영역 내에 있는지 체크
        # 있다면 두개의 contour를 하나의 묶음으로 묶어주기 위해 realationBox에 저장한다.
        if centerX + box[i][2] > box[j][0] and centerX + box[i][2] < box[j][0] + box[j][2] and centerY > box[j][1] and centerY < box[j][1] + box[j][3]:
            realationBox.append((box[i], box[j]))
# print(realationBox)

# 결과 출력용 코드
# realationBox로 묶인 2개의 contour를 하나의 box로 그려준다.
# finalX: box를 그릴 x좌표 시작값
# finalY: box를 그릴 y좌표 시작값
# finalWidth: box의 width값
# finalHeight: box의 hieght값
for i in range(len(realationBox)):
    # print(realationBox[i])
    if realationBox[i][0][0] > realationBox[i][1][0]:
        finalX = realationBox[i][1][0]
        maxX = realationBox[i][0][0] + realationBox[i][0][2]
        finalWidth = maxX - finalX
    else:
        finalX = realationBox[i][0][0]
        maxX = realationBox[i][1][0] + realationBox[i][1][2]
        finalWidth = maxX - finalX

    if realationBox[i][0][1] > realationBox[i][1][1]:
        finalY = realationBox[i][1][1]
    else:
        finalY = realationBox[i][0][1]
    
    if realationBox[i][0][3] > realationBox[i][1][3]:
        finalHeight = realationBox[i][0][3]
    else:
        finalHeight = realationBox[i][1][3]
    
    cv.rectangle(imgCopy, (finalX, finalY), (finalX + finalWidth, finalY + finalHeight), (0, 0, 255), 2)

cv.imshow('contours', imgCopy)
cv.waitKey(0)
