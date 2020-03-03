import cv2
import time
import numpy as np



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
            기준 = (range_) * (devide / 2) - 1
            if b <= 기준:
                if b==127:
                    b = (int(b / range_)-1) * (range_)
                else:
                    b = (int(b / range_) ) * (range_ )
            else:
                b = (int(b / range_) + 1) * (range_ )

            if g <= 기준:
                if g==127:
                    g = (int(g / range_)-1) * (range_)
                else:
                    g = (int(g / range_) ) * (range_ )
            else:
                g = (int(g / range_) + 1) * (range_ )


            if r <= 기준:
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


    # 색 중복 제거
    color_list = list(set(map(tuple, color_list)))
    print('색의 갯수 : ', len(color_list) , '// 이미지의 색은 총 ', len(color_list), '개로 구성됨')
    print('색의 값 : ', color_list)
    print('이미지를 구성하는 색의 갯수 조절 끝 : ', time.time() - start_time, '\n')
    return image, color_list


# 색의 개수 만큼 검은색 이미지를 만든다.
def createBlackImage(bgr_binding_image, color_list_len):
    start_time = time.time()

    print('검은색 이미지 ', color_list_len, '개 만들기 시작')

    draw_image_list = []

    for i in range(color_list_len):
        black_image = np.zeros_like(bgr_binding_image)
        draw_image_list.append(black_image)

    print('검은색 이미지 만들기 끝 : ', time.time() - start_time, '\n')

    return draw_image_list

# 검은색 이미지위에 뽑아낸 색 그리기
def blackImageDraw(binding_image, black_image_list, color_list):
    start_time = time.time()
    print('검은색 이미지 위에 색 그리기 시작')
    h, w, c = binding_image.shape

    for index, color in enumerate(color_list):
        print(color)
        pts=np.where(np.all(binding_image==color,axis=-1))
        # 원본이미지픽셀=[]
        # 원본이미지픽셀.append(binding_image[pts[0],pts[1]])
        black_image_list[index][pts[0],pts[1]]=(255,255,255)

    kernel = np.ones((2, 2), np.uint8)
    for index, draw_image in enumerate(black_image_list):
        # 선이 안이어져 있는 부분을 메꾸어준다.
        # black_image_list[index] = cv2.morphologyEx(draw_image, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imshow('black_image_list[index]', draw_image)
        cv2.waitKey(0)
        pass
    print('검은색 이미지 위에 색 그리기 끝 : ', time.time() - start_time, '\n')

    return black_image_list

# 컨투어 찾기
def findContour(image):

    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy


if __name__ == '__main__':
    start_time = time.time()

    bgr_image = cv2.imread('image/test_image/2.jpg')
    cv2.imshow('bgr_image', bgr_image)

    bgr_binding_image, color_list = similarColorBinding(bgr_image, 4)
    cv2.imshow('bgr_binding_image', bgr_binding_image)
    cv2.waitKey(0)

    black_draw_image_list = createBlackImage(bgr_binding_image, len(color_list))

    image_list = blackImageDraw(bgr_binding_image, black_draw_image_list, color_list)

    for image in black_draw_image_list:

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _ ,bi_image = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)

        contour, hierarchy = findContour(bi_image)



    print('Time : ', time.time() - start_time)
    cv2.waitKey(0)
