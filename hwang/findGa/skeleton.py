import cv2 as cv
import numpy as np

###############################################
# '가' 라는 글씨를 찾기 코드
# 글씨 픽셀을 최대한 가늘게 해서 뼈대를 찾는다.
# 뼈대의 값이 '가'로 인식되는지 확인한다.
###############################################

# 이미지 그레이스케일 로 열기
imgOri = cv.imread('hwang/imgSet/find_1.png',0)
# cv2.imshow("img", img)
size = np.size(imgOri)
# print(size)

# 이미지 크기만큼 0만들어 있는 배열 생성
skel = np.zeros(imgOri.shape, np.uint8)
print(skel)

# 이미지 이진화
# ret은 임계값
# threshold의 첫번째 파라미터는 원본 이미지, 두번째는 임계값, 세번째는 이진화 시키는 픽셀의 색,
ret, img = cv.threshold(imgOri, 130, 255, cv.THRESH_BINARY_INV )
# cv2.imshow("img", img)
# cv2.waitKey(0)
# print("ret",ret)

#침식용 배열. 첫번째 파라미터는 배열 형태, 두번째는 배열 크기
element = cv.getStructuringElement(cv.MORPH_ERODE, (3, 3))
# print("element",element)
done = False

# 최소한의 두께가 나올때까지 침식 반복.
while (not done):
    eroded = cv.erode(img, element)
    temp = cv.dilate(eroded, element)
    temp = cv.subtract(img, temp)
    skel = cv.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv.countNonZero(img)
    if zeros == size:
        done = True


# cv2.imwrite("ssss.png",skel)
# cv2.imwrite("jjj.jpg",skel)

contours, hierarchy = cv.findContours(imgOri, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for count in contours:
    #countours의 꼭지점 갯수
    size = len(count)
    # print(size)

    # epsilon = 근사 정확도, 외곽선 길이값 구하기 위한 값, 외곽선이 닫힌 폐외곽선 기준
    epsilon = 0.01 * cv.arcLength(count, True)
    # approx = epsilon값에 따라 꼭지점 수를 줄여서 새로운 도형을 반환
    approx = cv.approxPolyDP(count, epsilon, True)

    # 반환된 도형의 꼭지점 갯수
    size = len(approx)
    print(size)

    cv.line(imgOri, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 2)
    for k in range(size - 1):
        cv.line(imgOri, tuple(approx[k][0]), tuple(approx[k + 1][0]),(0, 255, 0), 2)


cv.imshow("skel", skel)
cv.imshow("result", imgOri)
cv.waitKey(0)
cv.destroyAllWindows()
