"""

이 코드는 '가','나' 등 받침이 없으며, 모음이 오른쪽에 붙어있는 글자만을 찾는 코드이다.

글자를 찾는 원리는 아래와 같다.

1. 우선 2개의 이미지를 준비한다.
    - 하나는 '가'라는 글자만 있는 이미지 ( 비교 이미지 )
    - 다른 하나는 여러 글자가 섞여 있는 이미지를 준비한다 ( 비교 대상 이미지 )

2. 2개의 이미지에서 ㄱ,ㄴ(모음) / ㅏ,ㅑ(자음) 과 같은 글자를 찾아 구분짓는다.

3. 모음을 찾았으면, 모음을 기준으로 오른쪽 주변을 탐색하여 자음이 있는지 확인한다.
    - 자음이 있다면 해당 모음과 자음은 한 글자라고 판단한다.
    - 자음이 없다면 글자가 아니라고 판단한다.

4. 한 글자라고 판단된 글자를 비교 이미지에서 찾았고, 비교대상 이미지에서도 한 글자라고 판된된 글자가 있다면
   각각의 글자를 이미지에서 잘라낸 후에 같은 크기의 이미지로 만든다.

5. 마지막으로 같은 크기의 이미지 2개를 픽셀단위로 일치율 계산을 한다.

6. 일치율이 90% 이상이면 내가 찾은 글자라고 판단한다.

아래 코드에 있는 find_contour(), boundingRect()등의
OpenCv 함수는 공식 홈페이지인 https://docs.opencv.org/master/d6/d00/tutorial_py_root.html에서 참고하면 된다.
 
문제점 : 
1.
'가' 라는 글자와 '카'라는 글자가 되게 비슷하여 일치율이 높게 나옴
'가', '카'를 구분지을 수 있는 알고리즘을 추가하거나 다른 방식으로 찾아야함.

2. 일치율을 계산하는 테스트 방식이 올바른 것인지 잘 모르겠음. 테스트 하는 방법에 대해서 연구가 필요함.


"""

import cv2
import time

# 이미지 리사이즈 할때 가로,세로 크기
# 두개의 이미지를 비교할때 같은 크기의 이미지로 만들기 위해서 있는 변수
thumbnail_width = 100
thumbnail_height = 100

# rgb 이미지가 들어왔을때, 해당 이미지를 이진화 이미지로 바꾸고 컨투어(특징점)를 찾는다.
# 컨투어에 대한 설명은 아래 main문에 작성 해놓음.
def 이미지_이진화_및_컨투어_찾기(image):
    # rgb -> gray ( 이미지 변환 )
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 : 127값 기준으로 작으면 0 - 흑색 , 크면 255 백색으로 나눔
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 이진화된 이미지에서 특징점 찾기
    # 특징점은 contours에 리스트 형태로 들어있다.
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 반환값은 이진화된 이미지와, 컨투어 값임
    return img_binary, contours


# 이미지에서 찾은 특징점을 가지고 자음의 특성을 가진 리스트, 모음의 특성을 가진 리스트로 구분하는 메서드
def 자음_모음_컨투어_구분(contour):
    contour = contour.copy()

    자음_contour_list = []  # 자음 컨투어 리스트가 담길 공간
    모음_contour_list = []  # 모음 컨투어 리스트가 담길 공간

    for i, con in enumerate(contour):
        x, y, x_width, y_height = cv2.boundingRect(con)

        # 모음 조건 : 맑은 고딕 기준
        # 가로의 길이가 세로의 2배보다 짧으면 모음
        # 세로의 길이가 가로의 1.7배보다 짧으면 모음
        if x_width * 2 < y_height or y_height * 1.7 < x_width:
            모음_contour_list.append(con)

        # 자음 조건 : 모음이 아닌것은 우선 자음으로 판단한다.
        else:
            자음_contour_list.append(con)

    return 자음_contour_list, 모음_contour_list


""" 
찾은 자음을 가지고 오른쪽을 탐색하는 메서드 
ex ) 가, 나 와 같이 ㄱ, ㄴ 의 오른쪽에 있는 ㅏ 라는 모음을 찾음
"""
def 자음과_연관된_모음찾기(contour_list_자음, contour_list_모음):
    연관_list_contour_index = []
    new_contour_list = []
    temp_자음 = []
    temp_모음 = []

    for 자음_index, 자음_contour in enumerate(contour_list_자음):
        x, y, x_width, y_height = cv2.boundingRect(자음_contour)

        for 모음_index, 모음_contour in enumerate(contour_list_모음):
            for j in 모음_contour:

                # 자음의 x 최대 부터 자음의 ( 가로폭 / 2 ) 만큼 더 오른쪽으로 갔을때, 모음이 있다면?
                if j[0][0] > x + x_width and j[0][0] < x + x_width + int((x_width)):
                    연관_list_contour_index.append([자음_index, 모음_index])
                    break

    for index, contour_index in enumerate(연관_list_contour_index):
        for i in contour_list_자음[contour_index[0]]:
            # print(i[0])
            temp_자음.append(i[0])

        for i in contour_list_모음[contour_index[1]]:
            # print(i[0])
            temp_모음.append(i[0])
        new_contour_list.append([temp_자음 + temp_모음])
        temp_자음.clear()
        temp_모음.clear()
    # print(len(new_contour_list))

    return new_contour_list

# 비교할 이미지에서 자음+모음으로 구분된 리스트의 영역만을 찾는다.
# 그리고 그 후에 해당 영역을 thumbnail size(100,100) 크기로 변환한다. - 비교당할 이미지와 일치율 비교를 위해
def my_image_thumbnail(my_image, my_image_contour, my_new_contour_list):

    # 만약 자음 옆에 모음이 없다면 값이 안들어있다.
    # 값이 안들어있다면 이전에 찾은 컨투어를 가지고 사각형을 만든다.
    # 자음 + 모음 조합이라면 자음과 연관된 모음찾기 메서드에서 반환받은 컨투어 사용
    if len(my_new_contour_list) == 0:
        print('자음+모음 조합 아님')

        for i, con in enumerate(my_image_contour):

            x_min = 9999
            x_max = 0
            y_min = 9999
            y_max = 0

            for index, j in enumerate(my_image_contour[i]):

                if x_min > j[0][0]:
                    x_min = j[0][0]

                if x_max < j[0][0]:
                    x_max = j[0][0]

                if y_min > j[0][1]:
                    y_min = j[0][1]

                if y_max < j[0][1]:
                    y_max = j[0][1]
            my_draw = cv2.rectangle(my_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
            my_draw = cv2.putText(my_image, "No." + str(i), (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))

            my_crop_image = my_image[y_min:y_max, x_min:x_max]

            my_crop_image = cv2.resize(my_crop_image, dsize=(thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            my_crop_gray_image = cv2.cvtColor(my_crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, my_result_crop_image = cv2.threshold(my_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    else:
        print('자음+모음 조합임')
        for i, con in enumerate(my_new_contour_list):

            x_min = 9999
            x_max = 0
            y_min = 9999
            y_max = 0
            for index, j in enumerate(my_new_contour_list[i]):
                for k in j:
                    if x_min > k[0]:
                        x_min = k[0]

                    if x_max < k[0]:
                        x_max = k[0]

                    if y_min > k[1]:
                        y_min = k[1]

                    if y_max < k[1]:
                        y_max = k[1]
                my_draw = cv2.rectangle(my_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                my_draw = cv2.putText(my_image, "No." + str(i), (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                      (255, 0, 0))

                my_crop_image = my_image[y_min:y_max, x_min:x_max]

                my_crop_image = cv2.resize(my_crop_image, dsize=(thumbnail_width, thumbnail_height),
                                           interpolation=cv2.INTER_AREA)

                # rgb -> gray
                my_crop_gray_image = cv2.cvtColor(my_crop_image, cv2.COLOR_BGR2GRAY)

                # 이미지 이진화 0 - 흑 , 255 백으로 나눔
                ret, my_result_crop_image = cv2.threshold(my_crop_gray_image, 127, 255,
                                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return my_result_crop_image, my_draw

# 비교당할 이미지에서 자음+모음으로 구분된 리스트의 영역만을 찾는다.
# 그리고 그 후에 해당 영역을 thumbnail size(100,100) 크기로 변환한다. - 비교 이미지와 일치율 비교를 위해
def op_image_thumbnail(op_image, op_image_contour, op_new_contour_list):
    result_image_list = []
    op_draw_list = []
    point_min_list = []
    point_max_list = []

    print(len(op_new_contour_list))
    # 만약 연관된 컨투어가 있다면
    if len(op_new_contour_list) != 0:

        for i, con in enumerate(op_new_contour_list):

            x_min = 9999
            x_max = 0
            y_min = 9999
            y_max = 0
            for index, j in enumerate(op_new_contour_list[i]):
                for k in j:
                    if x_min > k[0]:
                        x_min = k[0]

                    if x_max < k[0]:
                        x_max = k[0]

                    if y_min > k[1]:
                        y_min = k[1]

                    if y_max < k[1]:
                        y_max = k[1]

            op_draw = cv2.rectangle(op_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
            # op_draw = cv2.putText(op_image, "No."+ str(i), (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
            point_min_list.append([x_min, y_min])
            point_max_list.append([x_max, y_max])
            op_draw_list.append(op_draw)

            # print('\n 비교할 이미지')
            op_crop_image = op_image[y_min:y_max, x_min:x_max]

            op_crop_image = cv2.resize(op_crop_image, dsize=(thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            op_crop_gray_image = cv2.cvtColor(op_crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, op_result_crop_image = cv2.threshold(op_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            result_image_list.append(op_result_crop_image)

    else:
        for i, con in enumerate(op_image_contour):
            x, y, x_width, y_height = cv2.boundingRect(con)
            op_draw = cv2.rectangle(op_image, (x, y), (x+x_width, y+y_height), (0, 0, 255), 1)

            point_min_list.append([x, y])
            point_max_list.append([x+x_width,  y+y_height])

            op_draw_list.append(op_draw)

            # print('\n 비교할 이미지')
            op_crop_image = op_image[y: y+y_height, x:x+x_width]

            op_crop_image = cv2.resize(op_crop_image, dsize=(thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            op_crop_gray_image = cv2.cvtColor(op_crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, op_result_crop_image = cv2.threshold(op_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            result_image_list.append(op_result_crop_image)

    return result_image_list, op_draw_list, point_min_list, point_max_list

# 픽셀 단위로 이미지를 비교하는 코드
# my_image = 비교 이미지
# op_image = 비교 대상 이미지
# 두 이미지의 각 픽셀 위치를 비교하여 같은값이 얼마나 있는지 판단.
# 최종 return 값은 2개의 이미지가 몇 % 나 일치하는지 일치율을 반환한다.
def image_comparison(my_image, op_image):
    my_image_copy = my_image.copy()
    op_image_copy = op_image.copy()

    # cv2.imshow('1',image1_copy)
    # cv2.imshow('2',image2_copy)
    # cv2.waitKey(0)
    count = 0

    sum = thumbnail_width * thumbnail_height

    image1_pixel_count = 0
    image2_pixel_count = 0

    for i in range(thumbnail_width):
        for j in range(thumbnail_height):
            if my_image_copy[i][j] == 0:
                image1_pixel_count += 1
            if op_image_copy[i][j] == 0:
                image2_pixel_count += 1
            # print("%3d"%image1_copy[i][j], end=" ")
        # print('\n')

    for i in range(thumbnail_width):
        for j in range(thumbnail_height):
            # print("%3d"%image2_copy[i][j], end=" ")
            if my_image_copy[i][j] == op_image_copy[i][j]:
                count += 1
        # print('\n')

    print(image1_pixel_count, image2_pixel_count)

    # 오차율 적용
    error = image1_pixel_count / 3

    pixel_range_min = int(image1_pixel_count - error)
    pixel_range_max = int(image1_pixel_count + error)

    print(pixel_range_min, pixel_range_max, (count * 100) / sum)

    if pixel_range_min > image2_pixel_count or image2_pixel_count > pixel_range_max:
        print('사이')
        return 0

    else:
        return (count * 100) / sum


if __name__ == '__main__':

    # 코드 수행시간 측정
    start_time = time.time()

    """######################################## 1. 이미지 불러오기 #######################################"""

    # my_image_path : 비교할 이미지의 경로
    # opponent_image_path : 비교대상 이미지의 경로
    my_image_path = '../image/margen_godic/basicImage/ga.jpg'
    opponent_image_path = '../image/margen_godic/comparisonImage/comparison_image_1.jpg'

    # 비교할 이미지와 비교대상 이미지 불러오기
    my_image = cv2.imread(my_image_path, cv2.IMREAD_COLOR)
    opponent_image = cv2.imread(opponent_image_path, cv2.IMREAD_COLOR)

    """################################# 2. 이진화 이미지로 변환 및 컨투어 찾기 ##########################"""

    # 컨투어 : 물체의 특징점을 반환해 주는 OpenCV의 메서드 ( ex : 특징점이라하면 물체의 꼭짓점 같은 것을 말함 )
    # * 이진화 처리된 이미지에서 컨투어를 찾아야 원하는 특징점이 도출된다.
    #
    # 컨투어를 찾는 이유 : 물체의 특징점을 찾은후, 특징점을 연결해서 글자스러움을 찾기위함
    # 비교대상과 비교할 이미지에서 모두 컨투어를 찾는다.
    # ~~_binary는 이진화 처리된 이미지
    # ~~_contour는 물체의 특징점 좌표를 리스트 형태로 가지고 있음
    my_image_binary, my_image_contour = 이미지_이진화_및_컨투어_찾기(my_image)
    op_image_binary, op_image_contour = 이미지_이진화_및_컨투어_찾기(opponent_image)

    # 비교 대상 이진화 이미지에서 컨투어를 찾은후, OpenCV의 boundingRect(), rectangle()를 사용한다.
    # boundingRect() -> 찾은 특징점에서 x좌표, y좌표, 가로길이, 세로길이를 반환해준다.
    # rectangle() -> 이미지위에 네모 영역을 그리는 역할을 한다.
    # for i, con in enumerate(op_image_contour):
    #     x, y, x_width, y_height = cv2.boundingRect(con)
    #     cv2.rectangle(opponent_image, (x, y), (x + x_width, y + y_height), (0, 0, 255), 1)
    #
    # # 이미지 확인
    # cv2.imshow('opponent_image', opponent_image)
    # cv2.waitKey(0)

    """#################### 3. 2번에서 찾은 컨투어를 자음, 모음컨투어로 구분짓기 ##########################"""

    # 자음_모음_컨투어_구분() - 찾은 컨투어(특징점) 리스트를 가지고, 자음과 비슷한지 모음과 비슷한지 구분을 한다.
    # 자음과 모음은 구분은 해당 글자의 가로길이와 세로길이의 비율로 판단한다.
    my_image_자음_contour_list, my_image_모음_contour_list = 자음_모음_컨투어_구분(my_image_contour)
    op_image_자음_contour_list, op_image_모음_contour_list = 자음_모음_컨투어_구분(op_image_contour)

    """############## 4. 자음을 찾은 후, 그와 연관된 주변 모음을 찾아서 하나의 글자로 구분짓기 #############"""
    # 자음으로 부터 모음을 찾아서, 자음+모음이 완성되는 글자만 찾는다.
    my_new_contour_list = 자음과_연관된_모음찾기(my_image_자음_contour_list, my_image_모음_contour_list)
    op_new_contour_list = 자음과_연관된_모음찾기(op_image_자음_contour_list, op_image_모음_contour_list)

    """########## 5. 비교 이미지와 비교당할이미지에서 찾은 글자를 100,100으로 리사이즈하는 과정.   ###############"""
    """################################### * 두개의 이미지 비교를 위해   #####################################"""

    # 비교할 이미지의 썸네일을 만든다. 1장
    my_thumbnail, my_draw = my_image_thumbnail(my_image, my_image_contour, my_new_contour_list)
    # 비교당할 이미지의 썸네일을 만든다. 1장 이상
    op_thumbnail_list, op_draw_list, point_min_list, point_max_list = op_image_thumbnail(opponent_image,
                                                                                         op_image_contour,
                                                                                         op_new_contour_list)

    """########## 6. 5번에서 리사이즈한 2개의 이미지를 비교하여 일치율을 계산하는 과정.   ###############"""

    # 새로운 비교 대상 이미지 생성 - 깨끗한 이미지 위에 내가 찾은 글자만 표시하기 위해
    result_image = cv2.imread(opponent_image_path, cv2.IMREAD_COLOR)

    """ 내 이미지와 비교대상 이미지를 픽셀 단위로 비교한다. """
    for i in range(len(op_thumbnail_list)):
        # image_comparison()
        # 2개 이미지 비교
        # 일치율 반환을함
        percent = image_comparison(my_thumbnail, op_thumbnail_list[i])
        print('No.', str(i), '과 일치율 : ', percent)

        #
        """########## 7. 일치율이 90% 이상이면 내가 찾으려는 글자라고 판단   ###############"""
        """########## 8. 최종적으로 결과 이미지에 빨간 네모 박스를 그려준다.   ###############"""
        if percent >= 90:
            cv2.rectangle(result_image, (point_min_list[i][0], point_min_list[i][1]),
                          (point_max_list[i][0], point_max_list[i][1]), (0, 0, 255), 1)
            cv2.putText(result_image, "No." + str(i) + " : " + str(percent) + "%",
                        (point_min_list[i][0], point_min_list[i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))

    cv2.imshow('result_image', result_image)
    cv2.waitKey(0)

    print("걸린 시간 : ", time.time() - start_time)
