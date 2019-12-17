import cv2 as cv
import numpy as np

# 이미지 읽어오기, rgv/grayscale/blur/binary 처리된 이미지를 각각 생성
imgOriginal = cv.imread('img/find_1.png', cv.IMREAD_COLOR)
imgCopy = imgOriginal.copy()
imgGrayscale = cv.cvtColor(imgOriginal, cv.COLOR_RGB2GRAY)
imgBlur = cv.GaussianBlur(imgGrayscale, (3, 3), 0)
ret, imgBinary = cv.threshold(imgGrayscale, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

########################################
# cornerHarris()를 이용해서 코너값 구하기
# imgBlur = np.float32(imgBlur)
# dst = cv.cornerHarris(imgBlur, 2, 5, 0.04)
# dst = cv.dilate(dst, None)

# imgCopy[dst > 0.01 * dst.max()] = [0, 0, 255]
# cv.imshow('Harris', imgCopy)
# cv.waitKey(0)
#########################################
# goodFeaturesToTrack을 이용해서 코너값 구하기
# corners = cv.goodFeaturesToTrack(imgCanny, 28, 0.01, 10)
# corners = np.int0(corners)

# for i in corners:
#     x, y = i.ravel()
#     cv.circle(imgCopy, (x, y), 3, 255, -1)
#     # print(x, y)
# cv.imshow('Corner', imgCopy)
# cv.waitKey(0)
#########################################
# imgCanny = cv.Canny(imgBlur, 100, 200)

# 외각선 따기
contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
box = []
for i in range(len(contours)):
    cnt = contours[i]
    area = cv.contourArea(cnt)
    x, y, w, h = cv.boundingRect(cnt)
    box.append(cv.boundingRect(cnt))
    # cv.drawContours(imgOriginal, [cnt], 0, (0, 255, 0), 2)

    # check = cv.isContourConvex(cnt)
    # if not check:
    #     hull = cv.convexHull(cnt)
    #     cv.drawContours(imgCopy, [hull], 0, (0, 255, 0), 1)

# print('box=', box)
realationBox = []
for i in range(len(box)):
    # cv.rectangle(imgOriginal, (box[i][0], box[i][1]), (box[i][0] + box[i][2], box[i][1] + box[i][3]), (0, 255, 0), 1)
    # 박스의 중점 구하기
    centerX = box[i][0] + (box[i][2] / 2)
    centerY = box[i][1] + (box[i][3] / 2)

    flag = True
    # print('centerX=', centerX, 'centerY=', centerY)
    for j in range(len(box)):
        if flag:
            # print('centerX + box[j][2]=', centerX + box[j][2], 'centerY=', centerY)
            flag = False
            
        # print('x=', box[j][0], 'x+w=', box[j][0] + box[j][2], 'y=', box[j][1], 'y+h=', box[j][1] + box[j][3])
        if centerX + box[i][2] > box[j][0] and centerX + box[i][2] < box[j][0] + box[j][2] and centerY > box[j][1] and centerY < box[j][1] + box[j][3]:
            # print('i=', i, 'j=',j)
            realationBox.append((box[i], box[j]))

# 'ㄱ' 과 'ㅏ'를 찾았다면 2개의 관계성을 찾아야 한다.
# 'ㄱ' 우측에 무언가 있다면 그것은 'ㅏ'일 것이다.
# print(realationBox)
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

# cv.imshow('covexhull', imgCopy)
cv.imshow('contours', imgCopy)
cv.waitKey(0)
