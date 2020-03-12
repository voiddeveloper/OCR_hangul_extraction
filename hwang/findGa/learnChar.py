import cv2 as cv
import numpy as np

# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy

def xLineYLineAdd(img):
    # 가로선 추출
    x_line_image = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=1)
    x_line_image = np.absolute(x_line_image)
    x_line_image = np.uint8(x_line_image)
    # cv.imshow('x_line_image', x_line_image)

    # 세로선 추출
    y_line_image = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
    y_line_image = np.absolute(y_line_image)
    y_line_image = np.uint8(y_line_image)
    # cv.imshow('y_line_image', y_line_image)

    # 가로 세로 합친 이미지 보여주기
    bgr_x_line_add_y_line_image = cv.bitwise_or(x_line_image, y_line_image)

    return bgr_x_line_add_y_line_image

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


## main 시작 ##

for i in range(1, 6):
    # 이미지 경로
    image_path = 'hwang/imgSet/standard/00'+ str(i) + '.png'

    # bgr 이미지 불러오기
    bgr_image = cv.imread(image_path)

    # 이미지 resize
    ratio = 1
    bgr_image = cv.resize(bgr_image, dsize=(0, 0),fx=ratio,fy=ratio, interpolation=cv.INTER_LINEAR)
    height, width = bgr_image.shape[:2]
    black_image = np.zeros_like(bgr_image)

    # bgr -> gray 변환
    gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)

    # bgr -> gray 변환 -> 경계선 추출
    bgr_x_line_add_y_line_image = xLineYLineAdd(gray_image)
    
    # bgr -> gray 변환 -> 경계선 추출 -> binary 변환
    # ret, binary_image = cv.threshold(bgr_x_line_add_y_line_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # contours, hierarchy = findContour(binary_image)
    contours, hierarchy = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # print('contours = ', contours)
    # print('hierarchy = ', hierarchy)
    temp_x = width
    temp_y = height
    temp_x_w = 0
    temp_y_h = 0

    # 있는 그대로 contour 그리기
    for j in range(len(contours)):
        cnt = contours[j]
        
        # 면적
        area = cv.contourArea(cnt)
        # 둘레
        perimeter = cv.arcLength(cnt, True)
        # 종횡비
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        # 크기
        rect_area = w * h
        extent = float(area) / rect_area

        if x < temp_x:
            temp_x = x

        if y < temp_y:
            temp_y = y

        if x + w > temp_x_w:
            temp_x_w = x + w

        if y + h > temp_y_h:
            temp_y_h = y+ h

        cv.drawContours(black_image, [cnt], 0, (0, 255, 0), 1)
        print(j, '번째 area = ', area)
        print(j, '번째 perimeter = ', perimeter)
        print(j, '번째 aspect_ratio = ', aspect_ratio)
        print(j, '번째 extent = ', extent)

    # 근사법으로 contour 그리기
    # for j in range(len(contours)):
    #     cnt = contours[j]

    #     # 면적
    #     area = cv.contourArea(cnt)
    #     # 둘레
    #     perimeter = cv.arcLength(cnt, True)
    #     # 근사값 둘레
    #     epsilon = 0.025 * cv.arcLength(cnt, True)
    #     # 꼭지점
    #     approx = cv.approxPolyDP(cnt, epsilon, True)
    #     # 종횡비
    #     x, y, w, h = cv.boundingRect(cnt)
    #     aspect_ratio = float(w) / h
    #     # 크기
    #     rect_area = w * h
    #     extent = float(area) / rect_area


    #     cv.drawContours(black_image,[approx], 0, (0, 255, 0), 1)
    #     print(j, '번째 area = ', area)
    #     print(j, '번째 perimeter = ', perimeter)
    #     print(j, '번째 aspect_ratio = ', aspect_ratio)
    #     print(j, '번째 extent = ', extent)

    cv.rectangle(black_image, (temp_x, temp_y), (temp_x_w, temp_y_h), (0, 0, 255), 1)

    result_image = createBlackImageMultiple(bgr_image, 2, 3)
    showMultiImage(result_image, bgr_image, height, width, 3, 0, 0)
    showMultiImage(result_image, gray_image, height, width, 1, 0, 1)
    showMultiImage(result_image, binary_image, height, width, 1, 0, 2)
    showMultiImage(result_image, bgr_x_line_add_y_line_image, height, width, 1, 1, 0)
    showMultiImage(result_image, black_image, height, width, 3, 1, 1)

    cv.imshow("result" + str(i), result_image)
    print('width = ', temp_x_w, ' height = ', temp_y_h)
    print('1.1배 width = ', int(temp_x_w * 1.1), ' 1.1배 height = ', int(temp_y_h * 1.1))

    cv.waitKey(0)


