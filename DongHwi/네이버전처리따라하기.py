import cv2
import numpy as np
import time

"""

네이버 ocr 전처리 과정 따라하기

한개의 변수를 제외하고, 모든 변수 값을 네이버 따라함.

removehoughLineP() 함수에 
limit 변수가 있는데, 이 값은 안나와있음. 

[ 과정 ]
1. 이미지 불러오기
2. rgb -> gray 변환 // bgrToGray()
3. 경계 이미지 추출 // morphGradient()
4. 이진화 // adaptiveThresholdMEAN()
5. 글자 영역을 잘잡게 하기위한 전처리 // morphClose()
6. 불필요한 글자 영역(선) 제거 // removehoughLineP()
7. 컨투어 추출 // findContour()
8. 이미지에 글자 영역 그리기 // textDetectRect()

"""

# bgr 이미지 -> gray 이미지로 변환
def bgrToGray(image):
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image', gray_image)

    return gray_image


# 이미지에서 경계선만 남긴다.
def morphGradient(gray_image):
    """ 커널에 대해서 잘 모름 """
    kernel = np.ones((2, 2), np.uint8)

    # 팽창과정 - 글자를 조금 더 두껍게 만드는 과정
    dilation = cv2.dilate(gray_image, kernel, iterations=1)

    # 침식과정 - 글자를 조금 더 얇게 만드는 과정
    erosion = cv2.erode(gray_image, kernel, iterations=1)

    buffer1 = np.asarray(dilation)
    buffer2 = np.asarray(erosion)

    # 확대한 이미지에서 축소한 이미지를 빼면 경계선만 남은 이미지가 나온다.
    morph_gradient_image = buffer1 - buffer2

    cv2.imshow('morph_gradient_image', morph_gradient_image)

    return morph_gradient_image


# 이진화 처리
def adaptiveThresholdMEAN(image):
    adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 12)
    cv2.imshow('adaptive_mean', adaptive_mean)
    return adaptive_mean


# ((이미지를 팽창 -> 이미지를 침식)) 하는 과정을 적용하여 끊어진 점을 제거하는 과정임
# 글자와 글자가 뭉쳐져서 글자부분이 한 덩어리로 잘 묶인다고함.
def morphClose(image):
    kernel = np.ones((9, 5), np.uint8)
    closing_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow('closing_image', closing_image)
    return closing_image


# 직선을 검출하는 메서드임
# contour 추출 하기 전에 불필요한 선을 제거하는 과정
def removehoughLineP(image):

    # 글자영역을 그리기 위해서 검은 이미지 만듦
    black_image = np.zeros_like(image)

    limit = 10  # 이 값은 얼마로 정했는지 안나옴
    rho = 1
    threshold = 100  # 선 추출 정확도
    minLineLength = 80 # 추출한 선의 길이
    maxLineGap = 5 # 5픽셀 이내로 겹치는 선은 제외

    # 직선 검출
    lines = cv2.HoughLinesP(image, rho, np.pi / 360, threshold, minLineLength, maxLineGap)

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:

            # 글자 영역 그리기
            black_image = cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

            # 불필요한 선 제거 -> 세로가 limit이상인 선 제거
            if y2 - y1 > limit:
                black_image = cv2.line(black_image, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.imshow('black_image', black_image)

    return black_image


# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy

# 컨투어 영역 네모 박스 치기
def textDetectRect(bgr_image, contour):
    for i, con in enumerate(contour):
        x, y, w, h = cv2.boundingRect(con)
        bgr_image = cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow('detect_image', bgr_image)


if __name__ == '__main__':
    start_time = time.time()

    # 이미지 불러오기
    bgr_image = cv2.imread('image/test_image/6.jpg')
    cv2.imshow('bgr_image', bgr_image)

    # rgb -> gray
    gray_image = bgrToGray(bgr_image)

    # 이미지에서 경계선 추출
    morph_gradient_image = morphGradient(gray_image)

    # 이진화 처리
    adaptive_mean_image = adaptiveThresholdMEAN(morph_gradient_image)

    # 글자 영역을 잘 잡게하기 위한 전처리
    morph_close = morphClose(adaptive_mean_image)

    # 글자 영역을 검은 이미지 위에 하얀색으로 그림
    text_detect_image = removehoughLineP(morph_close)

    # 컨투어 추출
    contour, hierarchy = findContour(text_detect_image)

    # 원래 bgr 이미지에서 컨투어 찾기
    textDetectRect(bgr_image, contour)

    print(time.time() - start_time)
    cv2.waitKey(0)
