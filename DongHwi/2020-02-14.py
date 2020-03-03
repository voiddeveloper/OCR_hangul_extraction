import cv2
import numpy as np
import time
import math

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
    cv2.imshow('binary_image', img)
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


def hsvEqualized(hsv_image):
    start_time = time.time()

    h, s, v = cv2.split(hsv_image)

    # h,s,v값을 히스토그램 평활화
    equalizedH = cv2.equalizeHist(h)
    equalizedS = cv2.equalizeHist(s)
    equalizedV = cv2.equalizeHist(v)

    # h,s,v,를 각각 평활화 작업후 를 합쳐서 새로운 hsv 이미지를 만듦.
    new_hsv_image = cv2.merge([equalizedH, equalizedS, equalizedV])

    # hsv -> bgr
    new_hsv_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2BGR)

    print('hsv 평활화 후 bgr 이미지로 변환 : ', time.time() - start_time, '\n')
    return new_hsv_image


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
        cv2.imshow('basic' + str(index), black_image_list[index])

        # 가로로 늘리기
        kernel = np.ones((8, 3), np.uint8)
        black_image_list[index] = cv2.morphologyEx(black_image_list[index], cv2.MORPH_CLOSE, kernel, iterations=1)

        # 침식
        kernel = np.ones((2, 2), np.uint8)
        black_image_list[index] = cv2.erode(black_image_list[index], kernel, iterations=1)

        # cv2.imshow('black' + str(index), black_image_list[index])
        # cv2.waitKey(0)
    print('검은색 이미지 위에 색 그리기 끝 : ', time.time() - start_time, '\n')

    return black_image_list


# 시간체크 시작
start_time = time.time()

# 이미지 경로
image_path = '../image/test_image/2.jpg'

# bgr 이미지 불러오기
bgr_image = cv2.imread(image_path)
cv2.imshow('bgr_image', bgr_image)

# bgr -> hsv로 변환
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# hsb 이미지 평활화 후 bgr 이미지로 바꾸는 작업
new_bgr_image = hsvEqualized(hsv_image)

# bgr -> gray 변환
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)

# 가로선 추출
x_line_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=1)
x_line_image = np.absolute(x_line_image)
x_line_image = np.uint8(x_line_image)
cv2.imshow('x_line_image', x_line_image)

# 세로선 추출
y_line_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=1)
y_line_image = np.absolute(y_line_image)
y_line_image = np.uint8(y_line_image)
cv2.imshow('y_line_image', y_line_image)

# 가로 세로 합친 이미지 보여주기
bgr_x_line_add_y_line_image = cv2.bitwise_or(x_line_image, y_line_image)
cv2.imshow('bgr_x_line_add_y_line_image', bgr_x_line_add_y_line_image)

""" 추가 작업 해보는 곳 """

""""""

# 검은색 이미지 만들기
black_image_list = createBlackImage(bgr_x_line_add_y_line_image)

# 검은색 이미지위에 gray 범위 값에 해당하는 부분 흰색으로 그리는 메서드
draw_image_list = blackImageDraw(bgr_x_line_add_y_line_image, black_image_list)

# 네모영역 그리기
contour_count = 0 # 컨투어 개수
for i in draw_image_list:
    con, hierarchy = findContour(i)
    for index, j in enumerate(con):
        x, y, w, h = cv2.boundingRect(j)
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        contour_count = contour_count + 1

# 종영 스켈레톤 적용 - 스켈레톤화된 이미지 반환함.
skel_image = skeletonize(bgr_x_line_add_y_line_image)
cv2.imshow('skel_image', skel_image)

# 컨투어 찾기
contour, hierarchy = findContour(skel_image)

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
