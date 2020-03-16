import cv2 as cv
import numpy as np
import math

#################################################################################
# 목표
# findChar.py를 통해서 찾아낸 글씨의 특징과 색을 찾아낸다.
# 글씨는 보통 특정한 방향으로 정렬되어 있다.
# 주변에 비슷한 색이 있는 부분을 찾고, 해당 방향으로 이미지를 검색한다.

# hwang/imgSet/test_image/14.jpg
# 현재 findChar.py 테스트 결과로 나온 x, y, w, h 값
# crop_0 = 694 566 42 38
# crop_1 = 558 566 45 38
#################################################################################

def findTwoColorDistanceHSV(hsvColor1, hsvColor2):
    # 2개의 색상의 공간 좌표값을 구하기 (HSV conic 모델)
    # 원뿔의 꼭지점을 원점, 세로축을 z축이라고 가정한다.
    hsvColor1H = int(hsvColor1[0])
    hsvColor1S = int(hsvColor1[1])
    hsvColor1V = int(hsvColor1[2])
    hsvColor2H = int(hsvColor2[0])
    hsvColor2S = int(hsvColor2[1])
    hsvColor2V = int(hsvColor2[2])

    x1 = (hsvColor1S * math.cos(2 * (math.pi * (hsvColor1H / 255))) * hsvColor1V) / 255
    y1 = (hsvColor1S * math.sin(2 * (math.pi * (hsvColor1H / 255))) * hsvColor1V) / 255
    z1 = hsvColor1V
    
    x2 = (hsvColor2S * math.cos(2 * (math.pi * (hsvColor2H / 255))) * hsvColor2V) / 255
    y2 = (hsvColor2S * math.sin(2 * (math.pi * (hsvColor2H / 255))) * hsvColor2V) / 255
    z2 = hsvColor2V

    # 2개의 색의 좌표값이 나오면 유클리디언 거리 공식을 이용하여 값을 구하기
    # d = sqrt{(h1-h2)^2 + (s1-s2)^2 + (v1-v2)^2}
    distance = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2)) 

    return distance

def findTwoColorCenterHSV(hsvColor1, hsvColor2):
    # 2개의 색상의 공간 좌표값을 구하기 (HSV conic 모델)
    # 원뿔의 꼭지점을 원점, 세로축을 z축이라고 가정한다.
    hsvColor1H = int(hsvColor1[0])
    hsvColor1S = int(hsvColor1[1])
    hsvColor1V = int(hsvColor1[2])
    hsvColor2H = int(hsvColor2[0])
    hsvColor2S = int(hsvColor2[1])
    hsvColor2V = int(hsvColor2[2])

    # 2개의 색의 좌표값이 나오면 2좌표의 중점을 구한다.
    centerPoint = [(hsvColor1H + hsvColor2H) / 2, (hsvColor1S + hsvColor2S) / 2, (hsvColor1V + hsvColor2V) / 2]

    return centerPoint

def colorFilter(img_color, color_dict):
    image = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    result = img_color.copy()

    color = color_dict['color']

    lower1 = np.array(color_dict['lower_range'])
    upper1 = np.array(color_dict['upper_range'])
    mask = cv.inRange(image, lower1, upper1)
    result = cv.bitwise_and(result, result, mask=mask)

    return mask

## main 시작 ##
# 1. findChar.py를 통해 추출된 이미지의 평균 hsv값 찾기
# 이미지 경로
image_path = 'hwang/resultFolder/1_crop.jpg'

# bgr 이미지 불러오기
bgr_image = cv.imread(image_path)
height, width = bgr_image.shape[:2]
gray_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
hsv_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

black_image = np.zeros_like(bgr_image)
temp_color = []

# 이미지 내 contour 추출 및 해당 contour의 평균색상 찾기
for j in range(len(contours)):
    average_color = [0, 0, 0]

    for i in range(len(contours[j])):
        hsv_color = hsv_image[contours[j][i][0][1], contours[j][i][0][0]]
        
        if i == 0:
            average_color[0] = hsv_color[0]
            average_color[1] = hsv_color[1]
            average_color[2] = hsv_color[2]

        average_color = findTwoColorCenterHSV(average_color, hsv_color)
        
    temp_color.append(average_color)

h = 0
s = 0
v = 0
for i in range(len(temp_color)):
    h += temp_color[i][0]
    s += temp_color[i][1]
    v += temp_color[i][2]

# 전체 contour의 평균 hsv 값
average_color = [int(h / len(temp_color)), int(s / len(temp_color)), int(v / len(temp_color))]
print("contour 평균 hsv 값 = ", average_color)

# 2. 전체 이미지에서 비슷한 색상 찾기
image_path = 'hwang/imgSet/test_image/14.jpg'

# bgr 이미지 불러오기
bgr_image = cv.imread(image_path)
height, width = bgr_image.shape[:2]

# crop할 영역, 테스트 값
# crop_0 = 694 566 42 38
# crop_1 = 558 566 45 38
crop_image = bgr_image[int(566 * 0.99) : int(604 * 1.01), 0 : width]

lower_h = average_color[0] * 0.6
if lower_h < 0:
    lower_h = 0
lower_s = average_color[1] * 0.6
if lower_s < 0:
    lower_s = 0
lower_v = average_color[2] * 0.6
if lower_v < 0:
    lower_v = 0

upper_h = average_color[0] * 1.4
if upper_h > 180:
    upper_h = 180
upper_s = average_color[1] * 1.4
if upper_s > 255:
    upper_s = 255
upper_v = average_color[2] * 1.4
if upper_v > 255:
    upper_v = 255

# 마스크 필터 생성
color_dict = {}
color_dict['color'] = 'average_color'
color_dict['lower_range'] = [lower_h, lower_s, lower_v]
color_dict['upper_range'] = [upper_h, upper_s, upper_v]
filter_result_image = colorFilter(crop_image, color_dict)

crop_contours, crop_hierarchy = cv.findContours(filter_result_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

start_x = width
end_x = 0
start_y = height
end_y = 0

for i in range(len(crop_contours)):
    cnt = crop_contours[i]
    x, y, w, h = cv.boundingRect(cnt)
    if start_x > x:
        start_x = x
    if start_y > y:
        start_y = y
    if end_x < x + w:
        end_x = x + w
    if end_y < y + h:
        end_y = y + h

cv.rectangle(crop_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
cv.imshow("test", crop_image)
cv.imshow("filter_result_image", filter_result_image)

start_x = start_x
end_x = end_x
start_y = start_y + 560
end_y = end_y + 560

cv.rectangle(bgr_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
cv.imshow("test2", bgr_image)
cv.waitKey(0)
