"""

이 코드는 이미지가 표현할 수 있는 색의 갯수를 줄이는 작업을 한다.

bgr 이미지는 b - 255개 , g - 255개, r - 255개 총 255*255*255개의 색을 표현한다.

이 색 표현을 예를 들어 8 * 8 * 8 갯수 만큼 줄이게 되면 어느정도 비슷한 색을 하나로 묶일 거라고 생각했다.

생각처럼 유사한 색이 하나로 묶였다.

시간이 다소 걸리지만 글자와 배경이 색이 다를때 뚜렷히 구분 된다는 점에서 유용하게 사용될것 같다.

"""

import time
import cv2

# 저해상도 이미지로 바꾸기
# 256 * 256 * 256 개수의 색을 - > devide * devide * devide 개수의 색깔로 바꾼다.
def similarColorBinding(image, devide):
    start_time = time.time()

    print('이미지를 구성하는 색의 갯수 조절 시작')

    height, width, channel = image.shape

    color_list = []

    for i in range(height):
        for j in range(width):
            b, g, r = image[i][j]

            limit_range = 30
            """rgb의 차이가 limit_range이하라면 비슷한 색이라고 판단한다 
               b가 170보다 큰 경우엔 검은색으로 판단
               b가 85보다 작으면 흰색 
               나머지는 회색
            """
            if (g - limit_range < b and b < g + limit_range) and r - limit_range < b and b < r + limit_range:
                if int(g) - int(r) < limit_range or int(r) - int(g) < limit_range:
                    if b >= 170:
                        image[i][j] = (255, 255, 255)
                        color_list.append([255, 255, 255])
                        continue
                    elif b <= 85:
                        image[i][j] = (0, 0, 0)
                        color_list.append([0, 0, 0])
                        continue

            """
            픽셀을 나눈 범위에 맞게 바꿔주는 부분
            ex) devide값이 2일때 값은 0 127 255 
            """
            range_ = int(256 / devide)
            color_limit = (range_) * (devide / 2) - 1
            if b <= color_limit:
                if b==127:
                    b = (int(b / range_)-1) * (range_)
                else:
                    b = (int(b / range_) ) * (range_ )
            else:
                b = (int(b / range_) + 1) * (range_ )

            if g <= color_limit:
                if g==127:
                    g = (int(g / range_)-1) * (range_)
                else:
                    g = (int(g / range_) ) * (range_ )
            else:
                g = (int(g / range_) + 1) * (range_ )


            if r <= color_limit:
                if r==127:
                    r = (int(r / range_)-1) * (range_)
                else:
                    r = (int(r / range_) ) * (range_ )
            else:
                r = (int(r / range_) + 1) * (range_ )

            if b < 0:
                b = 0
            elif b > 255:
                b = 255
            if r < 0:
                r = 0
            elif r > 255:
                r = 255
            if g < 0:
                g = 0
            elif g > 255:
                g = 255

            image[i][j] = (b, g, r)
            color_list.append([b, g, r])

    return image

# bgr 이미지 불러오기
bgr_image = cv2.imread('../image/test_image/1.jpg')
cv2.imshow('bgr_image', bgr_image)

# 유사한 색으로 묶는 과정 - 이미지가 색을 표현 하는 갯수를 줄이는 작업이다.
binding_image = similarColorBinding(bgr_image, 8)
cv2.imshow('binding_image', binding_image)

cv2.waitKey(0)
