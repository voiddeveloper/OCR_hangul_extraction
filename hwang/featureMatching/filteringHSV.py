import cv2
import sys
import numpy as np

##########################################################################
# hsv 색상 대역폭 확인해보기용 코드
# hsv 마스크를 생성하고, 해당 마스크에 적용되는 부분만 보이도록 만든다.

# ## 목표 ##
# 1. mask 범위를 선정한다. (범위의 기준은 hsv 색상 기준이다.)
#   1-1. 마스크 범위는 파라미터 트랙바를 통해 변경 가능하게 한다.
# 2. 생성된 mask를 이미지에 적용한다.
##########################################################################

# 아무일도 안하고 있을때 처리하는 용도
def nothing(x):
    pass

# 이미지 불러오기
image = cv2.imread('hwang/imgSet/test3.png')

# 파라미터 트랙바를 포함한 결과창 보기용 window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# 트랙바 생성
# hsv 최소값, 최대값을 입력하기 위해 6개의 트랙바가 필요하다.
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# 트랙바 초기값 설정
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# output : 마스크가 적용된 최종 이미지
output = image

# 무한루프를 돌려서 매프레임마다 마스크가 적용된 이미지를 화면에 출력한다.
while(1):

    # 트랙바의 현재값 받아오기
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # 트랙바의 값을 기준으로 마스크 min, max값 설정
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # hsv: 이미지의 hsv 버전
    # mask: min, max 범위로 설정된 마스크
    # output: 이미지에서 mask가 적용된 부분
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)

    # if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
    #     print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
    #     phMin = hMin
    #     psMin = sMin
    #     pvMin = vMin
    #     phMax = hMax
    #     psMax = sMax
    #     pvMax = vMax

    cv2.imshow('image',output)

cv2.destroyAllWindows()