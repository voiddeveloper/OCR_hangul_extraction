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


# 종(0)켈레톤
def skeletonize(img):
    start_time = time.time()

    # kernel 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 새로 그릴 검은 배경 이미지 생성
    skel = np.zeros(img.shape, dtype='uint8')

    # 구버젼 종켈레톤, (erode - dilate) 버전
    # while True:
    #     eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    #     temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
    #     temp = cv2.subtract(img, temp)
    #     skel = cv2.bitwise_or(skel, temp)
    #     img[:, :] = eroded[:, :]
    #     if cv2.countNonZero(img) == 0:
    #         break

    # 이미지 (팽창 - 축소) 작업을 계속해서 반복한다. 이를 통해 외부 테두리가 1픽셀씩 줄어들게 된다.
    # 테두리를 계속 깎다가 0픽셀이 되기 직전까지 반복한다.
    # while (cv2.countNonZero(img) != 0):
    while True:
        # Erode 작업 (팽창)
        erode = cv2.erode(img, kernel)
        # erode가 적용된 이미지에 opening 작업 (팽창된 이미지를 opening 처리)
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # 두 이미지의 차이 (이미지 경계선)
        subset = erode - opening

        temp_skel = cv2.bitwise_or(subset, skel)
        temp_img = erode.copy()

        if cv2.countNonZero(img) != 0:
            # 결과 표시하기
            skel = temp_skel
            # 다음 반복을 위한 img 갱신
            img = temp_img

        else:
            break
        

    print('종켈레톤 끝 : ', time.time() - start_time, '\n')
    cv2.imshow('temp_skeleton', temp_skel)
    return skel

# s값만 평활화 하는 메소드
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
    new_hsv_image = cv2.merge([h, equalizedS, v])

    # hsv -> bgr
    # new_hsv_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2BGR)

    print('hsv 평활화 후 bgr 이미지로 변환 : ', time.time() - start_time, '\n')
    # return new_hsv_image
    return new_hsv_image


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


def Filter_15(image):
    height, width = image.shape

    point_list = []
    remove_contour_index = []

    con, hir = findContour(image)
    black = np.zeros_like(image)

    for idx, j in enumerate(con):
        x, y, w, h = cv2.boundingRect(j)

        # 면적
        area = cv2.contourArea(j)

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

    kernel = np.ones((7, 3), np.uint8)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # kernel = np.ones((2, 2), np.uint8)
    # black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # kernel = np.ones((2, 2), np.uint8)
    # black = cv2.erode(black, kernel, iterations=1)

    # cv2.imshow('sfdsfd123', black)
    # cv2.waitKey(0)

    new_con, new_h = findContour(black)

    for i, con in enumerate(new_con):
        x, y, w, h = cv2.boundingRect(con)

        # 면적
        area = cv2.contourArea(con)

        # 둘레
        arc_len = round(cv2.arcLength(con, False), 2)

        percent = int((area * 100) / (w * h))

        if ((w <= 3 or h <= 3) and area < 10):
            continue

        # if arc_len < 40:
        #     continue

        # if percent < 10:
        #     continue

        else:
            cimg = black[y:y + h + 1, x:x + w + 1]
            cv2.imwrite(
                'hwang/resultFolder/' + str(i) + '@@' +
                str(area) + '@@' +
                str(arc_len) + '@@' +
                str(percent) + '@@' +
                str(w) + '@@' +
                str(h) + '.jpg',
                cimg)
            point_list.append([x, y, w, h])

    return point_list


# 검은색 이미지위에 뽑아낸 색 그리기
def blackImageDraw(x_y_line_image, black_image_list, bgr_image):
    start_time = time.time()
    print('검은색 이미지 위에 색 그리기 시작')
    devide_range = math.ceil(255 / image_num)

    global contour_count
    contour_count = 0

    height, width, c = bgr_image.shape
    point_list = []

    for index, image in enumerate(black_image_list):
        pts = np.where(
            (x_y_line_image >= (devide_range * (index))) & (x_y_line_image < (devide_range * (index + 1))))
        # print("test = ", devide_range * index, devide_range * (index + 1))
        black_image_list[index][pts[0], pts[1]] = 255

        point_list = Filter_15(black_image_list[index])

    print('좌표 중복 제거 전 개수 : ', len(point_list))
    point_list = list(set(map(tuple, point_list)))
    print('좌표 중복 제거 후 개수 : ', len(point_list))

    black_image = np.zeros_like(bgr_image)

    for x, y, w, h in point_list:
        cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
    # cv2.imshow('sdfsfd',black_image)
    # cv2.waitKey(0)

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

# 시간체크 시작
for i in range(109, 110):
    start_time = time.time()
    # 이미지 경로
    image_path = 'hwang/imgSet/test_image/' + str(i) + '.jpg'
    # image_path = '../image/test_image/a1.png'

    # bgr 이미지 불러오기
    bgr_image = cv2.imread(image_path)
    # cv2.imshow('bgr_image', bgr_image)

    height, width, c = bgr_image.shape
    print(bgr_image.shape)

    if width < 1000:
        width *= 2
        height *= 2

    # bgr
    bgr_image = cv2.resize(bgr_image, (width, height), interpolation=cv2.INTER_AREA)

    # bgr -> gray 변환
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_image', gray_image)

    # bgr -> gray 변환 -> 외곽선 검출
    bgr_x_line_add_y_line_image = xLineYLineAdd(gray_image)
    cv2.imshow('bgr_x_line_add_y_line_image', bgr_x_line_add_y_line_image)

    # bgr -> gray 변환 -> 외곽선 검출 -> (외곽선 이미지 - gray 변환 이미지)
    subtract_image = cv2.subtract(bgr_x_line_add_y_line_image, gray_image)
    cv2.imshow('subtract_image', subtract_image)

    # bgr -> gray 변환 -> 외곽선 검출 -> (외곽선 이미지 - gray 변환 이미지) -> 이진화 처리 (임계값: OTSU)
    th, bi_image = cv2.threshold(bgr_x_line_add_y_line_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary', bi_image)

    th, bi_image = cv2.threshold(bgr_x_line_add_y_line_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('127binary', bi_image)

    # bgr -> gray 변환 -> 외곽선 검출 -> (외곽선 이미지 - gray 변환 이미지) -> 이진화 처리 (임계값: OTSU) -> 종켈레톤 적용
    # skel_image = skeletonize(bi_image)
    # cv2.imshow('skeleton', skel_image)

    # 검은색 이미지 만들기
    black_image_list = createBlackImage(bi_image)

    # 검은색 이미지위에 gray 범위 값에 해당하는 부분 흰색으로 그리는 메서드
    draw_image = blackImageDraw(bi_image, black_image_list, bgr_image)

    cv2.imshow('result' + str(i), bgr_image)

    print('컨투어 개수 : ', contour_count)
    # 시간측정 끝
    print("코드 수행시간 : ", time.time() - start_time)
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
