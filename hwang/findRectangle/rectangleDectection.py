import cv2 as cv

###############################################
# 이미지에서 네모 찾기 코드
# 이미지를 binary 처리한 다음, 거기서 contour 정보를 찾아낸다.
# contour의 꼭지점이 4개면 네모일 것이다.
###############################################

# 이미지 파일 불러오기(컬러)
imgColor = cv.imread('hwang/imgSet/findRectangle/real_2.jpg', cv.IMREAD_COLOR)

# 컬러 이미지를 그레이스케일로 변환
imgGrayscale = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

# 그레이스케일로 변환된 이미지를 바이너리로 변환
ret, imgBinary = cv.threshold(imgGrayscale, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

# contours = 동일한 색을 가지고 있는 영역의 경계선 정보
# RETR_EXTERNAL = contours 정보 중에서 바깥쪽 라인만 찾는다.
# CHAIN_APPROX_SIMPLE = contours 라인을 그릴 수 있는 포인트를 반환
contours, hierarchy = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# contour 정보를 얻었으면, 이제 각 contour의 꼭지점 갯수가 4개인 것만 표시한다.
for count in contours:
    #countours의 꼭지점 갯수
    size = len(count)

    # epsilon = 근사 정확도, 외곽선 길이값 구하기 위한 값, 외곽선이 닫힌 폐외곽선 기준
    epsilon = 0.005 * cv.arcLength(count, True)
    # approx = epsilon값에 따라 꼭지점 수를 줄여서 새로운 도형을 반환
    approx = cv.approxPolyDP(count, epsilon, True)

    # 반환된 도형의 꼭지점 갯수
    size = len(approx)

    if size == 4:
        cv.line(imgColor, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
        for k in range(size - 1):
            cv.line(imgColor, tuple(approx[k][0]), tuple(approx[k + 1][0]),(0, 255, 0), 3)

cv.imshow('result', imgColor)
cv.waitKey(0)