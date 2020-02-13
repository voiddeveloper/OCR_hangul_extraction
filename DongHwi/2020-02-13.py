import cv2
import numpy as np
import time


# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy


# 종(0)켈레톤
def skeletonize(img):
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    th, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    skel = img.copy()

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
    return skel


def hsvEqualized(hsv_image):
    h, s, v = cv2.split(hsv_image)

    # h,s,v값을 히스토그램 평활화
    equalizedH = cv2.equalizeHist(h)
    equalizedS = cv2.equalizeHist(s)
    equalizedV = cv2.equalizeHist(v)

    # h,s,v,를 각각 평활화 작업후 를 합쳐서 새로운 hsv 이미지를 만듦.
    new_hsv_image = cv2.merge([equalizedH, equalizedS, equalizedV])

    # hsv -> bgr
    new_hsv_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2BGR)
    return new_hsv_image


# 시간체크 시작
start_time = time.time()

# 이미지 경로
image_path = '../image/test_image/2.jpg'

# bgr 이미지 불러오기
bgr_image = cv2.imread(image_path)
cv2.imshow('bgr_image', bgr_image)

hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
new_bgr_image = hsvEqualized(hsv_image)

# bgr -> gray 변환
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

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

black_image = np.zeros_like(bgr_x_line_add_y_line_image)
pts = np.where(bgr_x_line_add_y_line_image >= 200)
black_image[pts[0], pts[1]] = 255
cv2.imshow('sdfsfd',black_image)

# 종영 스켈레톤 적용 - 스켈레톤화된 이미지 반환함.
skel_image = skeletonize(bgr_x_line_add_y_line_image)
cv2.imshow('skel_image', skel_image)

# 컨투어 찾기
contour, hierarchy = findContour(skel_image)

# 네모영역 그리기
for i, con in enumerate(contour):
    x, y, w, h = cv2.boundingRect(con)

    cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # cv2.circle(bgr_image, (int((x + x + w) / 2), int((y + y + h) / 2)), 1, (0, 0, 255), 1)
    # crop_image = new_bgr_image[y:y + h + 1, x:x + w + 1]
    # cv2.imwrite('../resultFolder/' + str(i) + '_crop.jpg', crop_image)

# 최종이미지 출력
cv2.imshow('result', bgr_image)

# 시간측정 끝
print("time : ", time.time() - start_time)
cv2.waitKey(0)
