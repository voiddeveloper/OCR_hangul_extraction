import cv2
import numpy as np
import time
import math

start_time = time.time()

# 255를 n으로 나눈 몫을 구한다.
# 몫의 숫자만큼 이미지를 만들예정
n = 70
image_num = math.ceil(255/n)
gray_range = math.ceil(255/image_num)

# 색의 개수 만큼 검은색 이미지를 만든다.
def createBlackImage(gray_image):
    start_time = time.time()

    print('검은색 이미지 ' + str(image_num) + '개 만들기 시작')

    draw_image_list = []

    for i in range(0, image_num):
        black_image = np.zeros_like(gray_image)
        draw_image_list.append(black_image)

    print('검은색 이미지 만들기 끝 : ', time.time() - start_time, '\n')

    return draw_image_list


# 검은색 이미지위에 뽑아낸 색 그리기
def blackImageDraw(gray_image, black_image_list):
    start_time = time.time()
    print('검은색 이미지 위에 색 그리기 시작')
    h, w = gray_image.shape

    count = 0

    for index, image in enumerate(black_image_list):
        pts = np.where((gray_image >= 0 + count) * (gray_image <= gray_range + count))
        print(count, ' : ', gray_range + count)
        black_image_list[index][pts[0], pts[1]] = 255
        count += gray_range
        cv2.imshow('black' + str(index), black_image_list[index])
    print('검은색 이미지 위에 색 그리기 끝 : ', time.time() - start_time, '\n')

    return black_image_list


# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy

# 컨투어 영역 네모 박스 치기
def textDetectRect(bgr_image, contour):
    for i, con in enumerate(contour):
        x, y, w, h = cv2.boundingRect(con)
        if (w/h) >= 1:
            bgr_image = cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow('detect_image', bgr_image)

# bgr이미지 불러오기
bgr_image = cv2.imread('../image/test_image/2.jpg')
cv2.imshow('bgr_image', bgr_image)

# bgr -> gray 이미지로 변환
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)

# 적응형 평활화
# 이미지의 밝기와 대비를 계산해서, 전체적인 대비를 조절해준다.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
adap_equalize_image = clahe.apply(gray_image)
# cv2.imshow('adap_equalize_image', adap_equalize_image)

# 경계선을 찾는데 세로 영역보다 가로영역을 두껍게 해서 찾는다.
kernel = np.ones((3, 7), np.uint8)
morph_gradient_image = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', morph_gradient_image)

kernel = np.ones((3, 3), np.uint8)
closing_image = cv2.morphologyEx(morph_gradient_image, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imshow('closing_image', closing_image)

# 검은 이미지 만들기
black_image = createBlackImage(gray_image)

# 검은 이미지 위에 글자라고 생각되는 영역 그리기
image_list = blackImageDraw(closing_image, black_image)

contour_time = time.time()
for image in image_list:
    contour, hierarchy = findContour(image)
    textDetectRect(bgr_image, contour)
print('원본에 네모박스 그리기 끝 : ', time.time() - contour_time, '\n')

print('Time : ', time.time() - start_time)
cv2.waitKey(0)
