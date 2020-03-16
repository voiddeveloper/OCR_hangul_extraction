import cv2 as cv
import numpy as np

###############################################
# '가' 라는 글씨를 찾기 코드
# findContour.py의 추가 버전
# findContour에서는 contour 영역만을 이용해서 찾았기 때문에, "가", "나", "다" 등을 구분할 수 없다.
# 새로운 특징점을 찾아본다.
# ## 목표 ##
# 글씨 픽셀을 최대한 가늘게 해서 뼈대를 찾는다.
# 뼈대의 값이 '가'로 인식되는지 확인한다.
###############################################

# 이미지 그레이스케일 로 열기
imgOri = cv.imread('hwang/imgSet/findGa/find_1.png', 0)
# cv2.imshow("img", img)
size = np.size(imgOri)
# print(size)

# 이미지 크기만큼 0만 있는 배열 생성
skel = np.zeros(imgOri.shape, np.uint8)
# print(skel)

# 이미지 이진화
# 테스트용 글씨가 검은색, 배경이 흰색이므로, 반전 이진화를 한다.
ret, img = cv.threshold(imgOri, 127, 255, cv.THRESH_BINARY_INV)
# cv2.imshow("img", img)

# 이미지 팽창(dilate) 이미지 침식(erode)를 이용하여 테두리 픽셀을 깎아낼 수 있다.
## 방법 ##
# 1. 팽창한 이미지에서 침식한 이미지를 빼면 이미지의 테두리가 검출된다.
# 2. 원본 이미지에서 이 테두리를 빼면 주변 1픽셀을 깎아낸다.
# 3. 이미지가 0픽셀이 되기 전까지 이 과정을 반복하면 중심축 1픽셀만 남은 이미지가 나온다.
done = False
element = cv.getStructuringElement(cv.MORPH_ERODE, (3, 3))

while (not done):
    eroded = cv.erode(img, element)
    temp = cv.dilate(eroded, element)
    temp = cv.subtract(img, temp)
    skel = cv.bitwise_or(skel, temp)
    img = eroded.copy()

    # 너무 많이 깎여서 남은 픽셀이 없을 경우 반복문을 중단한다.
    zeros = size - cv.countNonZero(img)
    if zeros == size:
        done = True

# 뼈대는 추출 완료
# 하지만 이 뼈대가 '가' 라는 것을 인지하는 값을 찾아내지 못했다.
cv.imshow("skel", skel)
cv.waitKey(0)
