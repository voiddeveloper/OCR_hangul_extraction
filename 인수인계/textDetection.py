""" 

이 코드는 글자가 가지고 있는 획을 찾는 방식으로 진행하였고,

한 문장을 이루는 글자들은 밝기가 비슷할 것이라 생각을 하고 코드를 짰다.

즉, 글자의 가로선과 세로선을 찾은 후, 밝기가 비슷한 것들만 모아 글자스러움을 찾겠다는 말이다

전개 방식
1. bgr image -> gray_image 로 변환 : gray 이미지로 변환하는 이유는 밝기 값이 비슷한 것들을 찾기위해

2. gray image 에서 가로와 세로선을 추출해낸다. 가로와 세로를 추출하는 방법이 OpenCV 함수로 제공된다.

3. 2번까지 하게 되면 가로와 세로선만이 남은 이미지가 보여지게된다.
    여기서 gray image의 범위는 0~255인데 이 값을 16등분으로 나눈후,
    16장의 각기 다른 밝기를 가진 이미지를 만들었다..
    그렇게 하면되면 비슷한 영역의 밝기값들이 모일것이라고 생각했기 때문이다.

4. 생각한대로 밝기가 비슷한 16장을 모두 검토한 결과, 대부분 글자가 모여있는 이미지가 존재했다.

5. 따라서 16장 모든 이미지에 글자스러움 필터를 적용하여,
   글자스러움 조건에 해당하는 영역만 남겨두었더니 결과가 생각보다 괜찮았다.


문제점 : 글자의 색이 주변 배경색과 비슷하면 찾지 못함
 
"""


import cv2
import numpy as np
import time
import math

# 검은색 이미지 개수 -
# gray 범위는 0~255인데 이 범위를 몇개로 나눌것인지
image_num = 16

contour_count = 0

# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy

# hsv 이미지를 평활화 하는 함수
# 평활화란, 이미지의 밝기 분포를 고르게해서 전체적으로 이미지가 밝아보이게 하는 기법을 말함.
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

    # hsv -> bgr
    # new_hsv_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2BGR)

    print('hsv 평활화 후 bgr 이미지로 변환 : ', time.time() - start_time, '\n')
    # return new_hsv_image
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


def hangulFilter(image):
    height, width = image.shape

    point_list = []

    con, hir = findContour(image)

    black = np.zeros_like(image)

    remove_contour_index = []

    for idx, j in enumerate(con):
        x, y, w, h = cv2.boundingRect(j)

        # 둘레
        arc_len = cv2.arcLength(j, False)

        # 제외
        if arc_len > width + height or (w == width or h == height):
            remove_contour_index.append(idx)

    remove_contour_index.reverse()

    if remove_contour_index:
        for i in remove_contour_index:
            del con[i]

    for idx, j in enumerate(con):
        black = cv2.drawContours(black, con, idx, color=(255, 255, 255), thickness=-1)

    new_con, new_h = findContour(black)

    for i, con in enumerate(new_con):
        x, y, w, h = cv2.boundingRect(con)

        # 면적
        area = cv2.contourArea(con)

        # 둘레
        arc_len = round(cv2.arcLength(con, False), 2)

        percent = int((area * 100) / (w * h))

        if ((w <= 10 or h <= 10) and area < 200):
            continue

        if arc_len < 40:
            continue

        if percent < 10:
            continue

        else:
            cimg = black[y:y + h + 1, x:x + w + 1]
            cv2.imwrite(
                '../resultFolder/' + str(i) + '@@' +
                str(area) + '@@' +
                str(arc_len) + '@@' +
                str(percent) + '@@' +
                str(w) + '@@' +
                str(h) + '.jpg',
                cimg)
            point_list.append([x, y, w, h])

    return point_list


# 깨끗한 검은색 이미지위에 뽑아낸 색 그리기
def blackImageDraw(x_y_line_image, black_image_list, bgr_image):
    start_time = time.time()
    print('검은색 이미지 위에 색 그리기 시작')
    devide_range = math.ceil(255 / image_num)

    cv2.imshow('sdfsfd',x_y_line_image)
    cv2.waitKey(0)

    contour_count = 0

    point_list = []

    for index, image in enumerate(black_image_list):
        pts = np.where(
            (x_y_line_image >= (devide_range * (index))) & (x_y_line_image < (devide_range * (index + 1))))
        black_image_list[index][pts[0], pts[1]] = 255

        cv2.imshow('black_image_list[index]', black_image_list[index])
        cv2.waitKey(0)


        point_list = hangulFilter(black_image_list[index])

    print('좌표 중복 제거 전 개수 : ', len(point_list))
    point_list = list(set(map(tuple, point_list)))
    print('좌표 중복 제거 후 개수 : ', len(point_list))

    black_image = np.zeros_like(bgr_image)

    for x, y, w, h in point_list:
        cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    kernel = np.ones((1, 10), np.uint8)
    black_image = cv2.dilate(black_image, kernel, iterations=1)

    # cv2.imshow('sdfsfd',black_image)
    # cv2.waitKey(0)

    black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)

    new_con, new_hir = findContour(black_image)
    for idx, j in enumerate(new_con):
        x, y, w, h = cv2.boundingRect(j)
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

    contour_count += len(point_list)

    print('검은색 이미지 위에 색 그리기 끝 : ', time.time() - start_time, '\n')

    return black_image

# 이미지에서 가로선과 세로선이 있는 영역만 남겨두기
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
    # cv2.imshow('bgr_x_line_add_y_line_image', bgr_x_line_add_y_line_image)
    return bgr_x_line_add_y_line_image


start_time = time.time()

"""##########  1. 이미지 불러오기 및 변환 ########## """
# 이미지 경로
image_path = '../image/test_image/1.jpg'
# image_path = '../image/test_image/a1.png'

# bgr 이미지 불러오기
bgr_image = cv2.imread(image_path)
# cv2.imshow('bgr_image', bgr_image)

# # bgr -> gray 변환
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_image', gray_image)

"""##########  2. 가로선 세로선만 있는 이미지로 변환 ########## """
bgr_x_line_add_y_line_image = xLineYLineAdd(gray_image)
# cv2.imshow('bgr_x_line_add_y_line_image', bgr_x_line_add_y_line_image)


"""##########  2. 가로선 세로선만 있는 이미지로 변환 ########## """
th, bi_image = cv2.threshold(bgr_x_line_add_y_line_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 검은색 이미지 만들기
black_image_list = createBlackImage(bi_image)

# 검은색 이미지위에 gray 범위 값에 해당하는 부분 흰색으로 그리는 메서드
draw_image = blackImageDraw(bi_image, black_image_list, bgr_image)

cv2.imshow('result_image', bgr_image)

print('컨투어 개수 : ', contour_count)
# 시간측정 끝
print("코드 수행시간 : ", time.time() - start_time)
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
cv2.waitKey(0)
cv2.destroyAllWindows()
