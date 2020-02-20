import cv2
import numpy as np
import time
import math
from matplotlib import pyplot as plt

# 검은색 이미지 개수 -
# gray 범위는 0~255인데 이 범위를 몇개로 나눌것인지
image_num = 16


# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy


# 종(0)켈레톤
def skeletonize(img):
    start_time = time.time()

    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    th, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    skel = img.copy()
    # cv2.imshow('binary_image', img)
    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv2.countNonZero(img) == 0:
            break
    print('종켈레톤 끝 : ', time.time() - start_time, '\n')

    return skel

# hsv 이미지에서 h,s,v 값 모두 평활화 작업 후 bgr로 변환
def hsvEqualized(hsv_image):
    start_time = time.time()

    h, s, v = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # h,s,v값을 히스토그램 평활화
    # equalizedH = cv2.equalizeHist(h)
    # equalizedS = cv2.equalizeHist(s)
    # equalizedV = cv2.equalizeHist(v)
    equalizedH = clahe.apply(h)
    equalizedS = clahe.apply(s)
    equalizedV = clahe.apply(v)

    # h,s,v,를 각각 평활화 작업후 를 합쳐서 새로운 hsv 이미지를 만듦.
    new_hsv_image = cv2.merge([equalizedH, equalizedS, equalizedV])
    
    print('hsv 평활화 후 bgr 이미지로 변환 : ', time.time() - start_time, '\n')
    # return new_hsv_image
    return new_hsv_image

# hsv 이미지에서 s값만 평활화 작업 후 bgr로 변환
def sEqualized(hsv_image):
    start_time = time.time()

    h, s, v = cv2.split(hsv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    height, width, depth = hsv_image.shape[:]

    # s값을 히스토그램 평활화
    # equalizedS = cv2.equalizeHist(s)
    equalizedS = clahe.apply(s)

    # s를 평활화 작업후 새로운 hsv 이미지를 만듦.
    new_s_image = cv2.merge([h, equalizedS, v])

    # hsv -> bgr
    new_s_image = cv2.cvtColor(new_s_image, cv2.COLOR_HSV2BGR)

    print('hsv 평활화 후 bgr 이미지로 변환 : ', time.time() - start_time, '\n')
    # return new_s_image
    return new_s_image

# 이미지를 grayscale로 변환 후 평활화 작업 반환되는 이미지도 grayscale
def equalizedGrayImage(img):
    start_time = time.time()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_image)

    print('gray 이미지 평활화 변환 : ', time.time() - start_time, '\n')

    return equalized


# 색의 개수 만큼 검은색 이미지를 만든다.
def createBlackImage(image):
    start_time = time.time()

    print('검은색 이미지 ' + str(image_num) + '개 만들기 시작')

    draw_image_list = []

    for i in range(0, image_num):
        black_image = np.zeros_like(image)
        draw_image_list.append(black_image)

    print('검은색 이미지 만들기 끝 : ', time.time() - start_time, '\n')

    return draw_image_list

# 검정색 이미지를 n배 크게 생성하기
# hcount = 높이 배수 (ex: 2: 세로로 2배)
# wcount = 넓이 배수 (ex: 2: 가로로 2배)
def createBlackImageMultiple(image, hcount, wcount):
    h,w,d = image.shape[:]
    image = np.zeros((h * hcount, w * wcount, d), np.uint8)
    color = tuple(reversed((0, 0, 0)))
    image[:] = color

    return image

# 통 이미지에서 원하는 위치에 이미지 붙여넣기
# dst = 통 이미지
# src = 붙여넣을 이미지
# h : 높이
# w : 넓이
# d : 깊이
# col : 행 위치
# row : 열 위치
def showMultiImage(dst, src, h, w, d, col, row):
    if d == 3:
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w] = src[0:h, 0:w] 
    elif d == 1:
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 0] = src[0:h, 0:w] 
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 1] = src[0:h, 0:w] 
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 2] = src[0:h, 0:w] 

# 검은색 이미지위에 뽑아낸 색 그리기
def blackImageDraw(x_y_line_image, black_image_list):
    start_time = time.time()
    print('검은색 이미지 위에 색 그리기 시작')
    devide_range = math.ceil(255 / image_num)
    for index, image in enumerate(black_image_list):
        pts = np.where(
            (x_y_line_image >= (devide_range * (index))) & (x_y_line_image < (devide_range * (index + 1))))
        # print(devide_range * index, devide_range * (index + 1))
        black_image_list[index][pts[0], pts[1]] = 255
        # cv2.imshow('basic' + str(index), black_image_list[index])

        # 가로로 늘리기
        kernel = np.ones((2, 4), np.uint8)
        black_image_list[index] = cv2.morphologyEx(black_image_list[index], cv2.MORPH_CLOSE, kernel, iterations=1)

        # 침식
        kernel = np.ones((2, 2), np.uint8)
        black_image_list[index] = cv2.erode(black_image_list[index], kernel, iterations=1)

        # cv2.imshow('black' + str(index), black_image_list[index])
        # cv2.waitKey(0)
    print('검은색 이미지 위에 색 그리기 끝 : ', time.time() - start_time, '\n')

    return black_image_list

def xLineYLineAdd(img):
    # 가로선 추출
    x_line_image = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    x_line_image = np.absolute(x_line_image)
    x_line_image = np.uint8(x_line_image)
    # cv2.imshow('x_line_image', x_line_image)

    # 세로선 추출
    y_line_image = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    y_line_image = np.absolute(y_line_image)
    y_line_image = np.uint8(y_line_image)
    # cv2.imshow('y_line_image', y_line_image)

    # 가로 세로 합친 이미지 보여주기
    bgr_x_line_add_y_line_image = cv2.bitwise_or(x_line_image, y_line_image)

    return bgr_x_line_add_y_line_image

def showHistogram(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(), 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    plt.show()

# 시간체크 시작
start_time = time.time()
for i in range(1, 13):
    # 이미지 경로
    image_path = 'hwang/imgSet/test_image/'+str(i)+'.jpg'

    # bgr 이미지 불러오기
    bgr_image = cv2.imread(image_path)
    height, width = bgr_image.shape[:2]
    # cv2.imshow('bgr_image', bgr_image)

    createBlackImageMultiple(bgr_image, 2, 2)

    # bgr -> hsv로 변환
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # bgr -> gray 변환
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_image', gray_image)

    # gray 이미지 경계선 추출
    # bgr_x_line_add_y_line_image = xLineYLineAdd(gray_image)
    # cv2.imshow('bgr_x_line_add_y_line_image', bgr_x_line_add_y_line_image)

    # hsv 이미지를 평활화 후 bgr 이미지로 바꾸는 작업
    new_hsv_image = sEqualized(hsv_image)
    # cv2.imshow('new_hsv_image', new_hsv_image)

    # ERROR: 2020-02-19 메소드는 hsv 이미지의 평탄화였는데, bgr 이미지를 사용하고 있었음
    # new_bgr_image = hsvEqualized(bgr_image)
    # cv2.imshow('new_bgr_image', new_bgr_image)

    # 평활화를 적용한 bgr -> gray 변환
    equlized_gray_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('equlized_gray_image', equlized_gray_image)

    # 평활화를 적용한 bgr -> gray -> 이진화 처리(TOZERO, 임계값: 127)
    ret, binary_image = cv2.threshold(equlized_gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary_image', binary_image)

    contours, hierarchy = findContour(binary_image)
    
    black_image = np.zeros_like(binary_image)

    for i, con in enumerate(contours):

        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        x, y, w, h = cv2.boundingRect(con)

        if perimeter < 12 or perimeter > height + width:
            continue

        if x < 7 and y < 7:
            continue

        # if area < 10:
        #     continue

        cv2.drawContours(black_image, contours, i, color=(255, 255, 255), thickness=-1)
        # cv2.imshow('black_image', black_image)
        # cv2.waitKey(0)

    cv2.imshow('black_image', black_image)

    x_line_add_y_line_image = xLineYLineAdd(black_image)
    cv2.imshow('x_line_add_y_line_image', x_line_add_y_line_image)

    kernel = np.ones((5,20), np.uint8)
    closing_binary_image = cv2.morphologyEx(x_line_add_y_line_image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing_binary_image', closing_binary_image)

    cv2.waitKey(0)

    # 평활화를 적용한 bgr -> gray -> 이진화 처리(TOZERO, 임계값: 127) -> 경계선 추출
    # binary_xline_yline_image = xLineYLineAdd(black_image)
    # cv2.imshow('binary_xline_yline_image', binary_xline_yline_image)
    # cv2.imwrite('abcd.jpg', binary_xline_yline_image)

    """ 추가 작업 해보는 곳 """

    """"""

    # 검은색 이미지 만들기
    # black_image_list = createBlackImage(bgr_x_line_add_y_line_image)

    # 검은색 이미지위에 gray 범위 값에 해당하는 부분 흰색으로 그리는 메서드
    # draw_image_list = blackImageDraw(bgr_x_line_add_y_line_image, black_image_list)

    # 네모영역 그리기
    contour_count = 0 # 컨투어 개수
    # for i in draw_image_list:
    #     con, hierarchy = findContour(i)
    #     for index, j in enumerate(con):
    #         x, y, w, h = cv2.boundingRect(j)
    #         cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #         contour_count = contour_count + 1

    # 종영 스켈레톤 적용 - 스켈레톤화된 이미지 반환함.
    # skel_image = skeletonize(bgr_x_line_add_y_line_image)
    # cv2.imshow('skel_image', skel_image)

    # 컨투어 찾기
    # contour, hierarchy = findContour(skel_image)

    # 네모영역 그리기
    # for i, con in enumerate(contour):
    #     x, y, w, h = cv2.boundingRect(con)
    #
    #     cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.circle(bgr_image, (int((x + x + w) / 2), int((y + y + h) / 2)), 1, (0, 0, 255), 1)
    # crop_image = new_bgr_image[y:y + h + 1, x:x + w + 1]
    # cv2.imwrite('../resultFolder/' + str(i) + '_crop.jpg', crop_image)

    # 최종이미지 출력
    cv2.imshow('result', bgr_image)

    print('컨투어 개수 : ', contour_count)
    # 시간측정 끝
    print("코드 수행시간 : ", time.time() - start_time)
    cv2.waitKey(0)
