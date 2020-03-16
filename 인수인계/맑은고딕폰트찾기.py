"""

이 코드는 맑은 고딕 폰트의 자음, 모음만을 찾는 코드이다.
 
자음 - ㄱ,ㄴ,ㄷ,ㄹ,ㅁ,ㅂ,ㅅ,ㅇ,ㅈ,ㅊ,ㅋ,ㅌ,ㅍ,ㅎ,
모음 - ㅏ,ㅑ,ㅓ,ㅕ,ㅓ,ㅕ,ㅗ,ㅛ,ㅜ,ㅠ

모든 자음, 모음 글자의 특성을 가지고, 트리 형태로 모든 글자를 구분지어서 글자를 찾는다.
조건 : 글자의 외각선 기준으로 글자의 픽셀이 변경되는 지점이 몇개인지 계산함.

조건 예시 )
    중앙선가로로나누기 - 글자를 중앙선 기준으로 가로로 잘랐을때 픽셀이 바뀌는 지점을 체크한다.
    중앙선세로로나누기 - 글자를 중앙선 기준으로 세로로 잘랐을때 픽셀이 바뀌는 지점을 체크한다.
    아랫선가로로나누기 - 글자의 제일 하단 기준으로 가로로 잘랐을때 픽셀이 바뀌는 지점을 체크한다.

** 트리 조건을 세울때 글자의 특성을 그림을 그려 설계를 하고 시작을 하였었음. (현재 트리 조건에 대한 그림은 없음)
** 다음번에 이 코드를 참고할때 이와 같은 방식으로 한다면 그림을 그린 후 작업을 하는것을 권장함.

아래 코드에 있는 find_contour(), boundingRect()등의
OpenCv 함수는 공식 홈페이지인 https://docs.opencv.org/master/d6/d00/tutorial_py_root.html에서 참고하면 된다.

문제점 : 자음 모음에 대한 기준을 세웠으나, 기준이 완벽하지 않고 글자가 아닌것에 대한 예외처리가 없음

"""

import cv2
import time


def 이미지_이진화_및_컨투어_찾기(image):
    # rgb -> gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 0 - 흑 , 255 백으로 나눔
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 반환값은 이진화된 이미지와, 컨투어 값임
    return img_binary, contours

def 중앙선세로로나누기(rgb_image, basic_binary_image, point_dict):
    """
    글자를 세로 중앙으로 잘랐을때 픽셀의 색이 0 -> 255 또는 255 -> 0으로 바뀌는 지점을 찾아낸다.
    바뀌는 지점이 몇개인지 return
    """

    image_copy = rgb_image.copy()

    # 글자의 중앙을 세로로 잘랐을때, 픽셀값을 mid_list에 저장한다.
    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if j == point_dict['x_mid']:
                mid_list.append(basic_binary_image[i][j])
                draw = cv2.circle(image_copy, (j, i), 1, (0, 0, 255), 1)

    # 글자가 바뀌는 지점을 찾아낸다. 0 -> 255 또는 255 -> 0
    pixel_change_count = 0
    for index, binary_value in enumerate(mid_list):
        if index != len(mid_list) - 1:
            if mid_list[index] != mid_list[index + 1]:
                pixel_change_count += 1

    # print(pixel_change_count)
    #
    # cv2.imshow('dsf',draw)
    # cv2.waitKey(0)

    return pixel_change_count

def 중앙선가로로나누기(rgb_image, basic_binary_image, point_dict):
    """
    글자를 가로 중앙으로 잘랐을때 픽셀의 색이 0 -> 255 또는 255 -> 0으로 바뀌는 지점을 찾아낸다.
    바뀌는 지점이 몇개인지 return
    """

    image_copy = rgb_image.copy()

    # 글자의 중앙을 세로로 잘랐을때, 픽셀값을 mid_list에 저장한다.
    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'] + 1, point_dict['x_max']):
            if i == point_dict['y_mid']:
                mid_list.append(basic_binary_image[i][j])
                draw = cv2.circle(image_copy, (j, i), 1, (0, 0, 255), 1)

    # 글자가 바뀌는 지점을 찾아낸다. 0 -> 255 또는 255 -> 0
    pixel_change_count = 0
    for index, binary_value in enumerate(mid_list):
        if index != len(mid_list) - 1:
            if mid_list[index] != mid_list[index + 1]:
                pixel_change_count += 1

    # print(pixel_change_count)
    #
    # cv2.imshow('dsf',draw)
    # cv2.waitKey(0)

    return pixel_change_count

def 아랫선가로로나누기(rgb_image, basic_binary_image, point_dict):
    """
    글자를 제일 아랫선 가로로 잘랐을때 픽셀의 색이 0 -> 255 또는 255 -> 0으로 바뀌는 지점을 찾아낸다.
    바뀌는 지점이 몇개인지 return
    """

    image_copy = rgb_image.copy()

    # 윗선을 가로로 잘랐을때, 픽셀값을 mid_list에 저장한다.
    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if i == point_dict['y_max'] - 1:
                mid_list.append(basic_binary_image[i][j])
                # draw = cv2.circle(image_copy, (j,i), 1 , (0,0,255), 1)

    # 글자가 바뀌는 지점을 찾아낸다. 0 -> 255 또는 255 -> 0
    pixel_change_count = 0
    for index, binary_value in enumerate(mid_list):
        if index != len(mid_list) - 1:
            if mid_list[index] != mid_list[index + 1]:
                pixel_change_count += 1

    # print(pixel_change_count)
    #
    # cv2.imshow('dsf',draw)
    # cv2.waitKey(0)

    return pixel_change_count

def 가로선_0부터_중앙사이_세로로나누기(rgb_image, basic_binary_image, point_dict):
    """
    글자를 가로선_0부터_중앙사이_세로로 잘랐을때 픽셀의 색이 0 -> 255 또는 255 -> 0으로 바뀌는 지점을 찾아낸다.
    바뀌는 지점이 몇개인지 return
    """

    image_copy = rgb_image.copy()

    # 글자의 중앙과 제일왼쪽 중간점을 기준으로 세로로 잘랐을때, 픽셀값을 mid_list에 저장한다.
    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if j == int((point_dict['x_min'] + point_dict['x_mid']) / 2):
                mid_list.append(basic_binary_image[i][j])
                draw = cv2.circle(image_copy, (j, i), 1, (0, 0, 255), 1)

    # 글자가 바뀌는 지점을 찾아낸다. 0 -> 255 또는 255 -> 0
    pixel_change_count = 0
    for index, binary_value in enumerate(mid_list):
        if index != len(mid_list) - 1:
            if mid_list[index] != mid_list[index + 1]:
                pixel_change_count += 1

    # print(pixel_change_count)
    #
    # cv2.imshow('dsf',draw)
    # cv2.waitKey(0)

    return pixel_change_count

def 중앙상단픽셀검사(rgb_image, basic_binary_image, point_dict):
    """
    중앙상단 1개의 픽셀 검사해서 있는지 판단한다. ( 너무 작은 이미지에선 안 될 수도 있음 )
    픽셀이 검은색이면 1
    픽셀이 흰색이면 0
    """

    image_copy = rgb_image.copy()

    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if j == point_dict['x_mid'] and i == point_dict['y_min']:
                # draw = cv2.circle(image_copy, (j,i), 2, (0,0,255), 3)
                # cv2.imshow('draw',draw)
                # cv2.waitKey(0)
                if basic_binary_image[i][j] == 255:
                    return 1
    return 0

def 가장왼쪽세로선으로나누기(rgb_image, basic_binary_image, point_dict):
    """
    글자를 가장 왼쪽 세로선으로 나누었을때 픽셀의 색이 0 -> 255 또는 255 -> 0으로 바뀌는 지점을 찾아낸다.
    바뀌는 지점이 몇개인지 return
    """

    image_copy = rgb_image.copy()

    # 글자의 제일왼쪽 선을 기준으로 세로로 잘랐을때, 픽셀값을 mid_list에 저장한다.
    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if j == point_dict['x_min'] + 1:
                mid_list.append(basic_binary_image[i][j])
                # draw = cv2.circle(image_copy, (j,i), 1 , (0,0,255), 1)

    # 글자가 바뀌는 지점을 찾아낸다. 0 -> 255 또는 255 -> 0
    pixel_change_count = 0
    for index, binary_value in enumerate(mid_list):
        if index != len(mid_list) - 1:
            if mid_list[index] != mid_list[index + 1]:
                pixel_change_count += 1

    # print(pixel_change_count)
    #
    # cv2.imshow('dsf',draw)
    # cv2.waitKey(0)

    return pixel_change_count

def 중앙오른쪽픽셀검사(rgb_image, basic_binary_image, point_dict):
    """
   오른쪽 중앙 픽셀만 검사한다. ( 너무 작은 이미지에선 안 될 수도 있음 )
   픽셀이 검은색이면 1
   픽셀이 흰색이면 0
   """

    image_copy = rgb_image.copy()

    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if j == point_dict['x_max'] - 1 and i == point_dict['y_mid']:
                if basic_binary_image[i][j] == 255:
                    return 1
    return 0

def 왼쪽중앙픽셀만검사(rgb_image, basic_binary_image, point_dict):
    """
    왼쪽 중앙 픽셀만 검사한다. ( 너무 작은 이미지에선 안 될 수도 있음 )
    픽셀이 검은색이면 1
    픽셀이 흰색이면 0
    """
    image_copy = rgb_image.copy()

    mid_list = []
    for i in range(point_dict['y_min'], point_dict['y_max']):
        for j in range(point_dict['x_min'], point_dict['x_max']):
            if j == point_dict['x_min'] and i == point_dict['y_mid']:
                if basic_binary_image[i][j] == 255:
                    return 1
    return 0

def 가로세로길이비교(point_dict):
    """
    글자의 가로와 세로길이를 비교하는 메서드

    글자의 가로가 세로보다 5배 이상 길거나 세로 길이가 0이라면 -> ㅡ
    글자의 세로가 가로보다 5배 이상 길거나 가로 길이가 0이라면 -> ㅣ

    ㅡ -> return 0
    ㅣ -> return 1
    ㅗ,ㅜ -> return 2
    """

    width = point_dict['x_max'] - point_dict['x_min']
    height = point_dict['y_max'] - point_dict['y_min']

    # ㅡ
    if width > height * 5 or height == 0:
        return 0
    # ㅣ
    elif height > width * 5 or width == 0:
        return 1
    # ㅗ, ㅜ
    else:
        return 2

if __name__ == '__main__':

    start_time = time.time()

    # 이미지 불러오기
    basic_image = cv2.imread('../comparison_image_9.png')

    # 이진화 이미지 및 글자의 컨투어 반환
    # 컨투어 : 물체의 특징점 좌표를 반환해줌
    basic_binary_image, basic_contour = 이미지_이진화_및_컨투어_찾기(basic_image)

    contour_list = [] # 글자의 컨투어 좌표를 담는 공간 ( 글자의 좌표 )
    hangul_list = [] # 글자의 이름을 담는 공간 ( 'giyeok' ,'nieun' 등등)

    """ 트리형태로 글자 구분하기"""
    for i, con in enumerate(basic_contour):

        # i 번째 컨투어(물체의 특징점)이 있는 영역을 찾는다.
        x, y, x_width, y_height = cv2.boundingRect(con)

        # 컨투어(물체의 특징점)에 대한 최소 x,y 좌표, 최대 x,y 좌표, 중간값 x, y좌표 등을 딕셔너리 형태로 관리한다.
        x_mid = int((x + (x_width + x)) / 2)
        y_mid = int((y + (y_height + y)) / 2)
        point_dict = {}
        point_dict['x_min'] = x
        point_dict['x_max'] = x_width + x
        point_dict['y_min'] = y
        point_dict['y_max'] = y_height + y
        point_dict['x_mid'] = x_mid
        point_dict['y_mid'] = y_mid

        """###################### 아래 코드 부터는 컨투어가 트리 조건에 따라서 어떤 글자인지 판별된다.#################"""

        중앙_가로선_픽셀_변경_갯수 = 중앙선가로로나누기(basic_image, basic_binary_image, point_dict)
        중앙_세로선_픽셀_변경_갯수 = 중앙선세로로나누기(basic_image, basic_binary_image, point_dict)
        print('가로', 중앙_가로선_픽셀_변경_갯수)
        print('세로', 중앙_세로선_픽셀_변경_갯수)

        """#################### 중앙선을 기준으로 세로로 나누면 픽셀이 바뀌는 지점이 1 ~ 4 개로 나누어진다. ###########"""
        중앙_세로선_픽셀_변경_갯수 = 중앙선세로로나누기(basic_image, basic_binary_image, point_dict)

        # 중앙선을 기준으로 세로로 나누었을때 픽셀이 바뀌는 지점이 0개인 글자들 -> ㅗ, ㅜ, ㅡ, ㅣ 얘네를 가지고 아래 if 문을 수행함.
        if 중앙_세로선_픽셀_변경_갯수 == 0:

            """ return 0 -> ㅡ , 1 -> ㅣ, 2 -> ㅗ, ㅜ"""
            가로세로길이 = 가로세로길이비교(point_dict)

            # ㅡ
            if 가로세로길이 == 0:
                print('Eu')
                contour_list.append(con)
                hangul_list.append('Eu')
            # ㅣ
            elif 가로세로길이 == 1:
                print('i')
                contour_list.append(con)
                hangul_list.append('i')
            # ㅗ, ㅜ
            elif 가로세로길이 == 2:

                """ 아래선 기준으로 픽셀이 바뀌는 지점이 0개 -> ㅗ, 2개 -> ㅜ """
                아랫선_픽셀_변경_갯수 = 아랫선가로로나누기(basic_image, basic_binary_image, point_dict)

                # ㅗ
                if 아랫선_픽셀_변경_갯수 == 0:
                    print('O')
                    contour_list.append(con)
                    hangul_list.append('O')
                # ㅜ
                elif 아랫선_픽셀_변경_갯수 == 2:
                    print('u')
                    contour_list.append(con)
                    hangul_list.append('u')
                else:
                    print('0 - 2 - 잘못찾음')
            else:
                print('0 - 잘못찾음')

        # 중앙선을 기준으로 세로로 나누었을때 픽셀이 바뀌는 지점이 1개인 글자들 -> ㄱ, ㄴ, ㅛ, ㅈ, ㅊ, ㅛ, ㅠ 얘네를 가지고 아래 if 문을 수행함.
        elif 중앙_세로선_픽셀_변경_갯수 == 1:
            중앙_가로선_픽셀_변경_갯수 = 중앙선가로로나누기(basic_image, basic_binary_image, point_dict)

            """ 중앙선을 기준으로 가로로 나누면 픽셀이 바뀌는 지점이 1, 2, 4 개로 나누어진다. """
            # 중앙선을 기준으로 가로로 나누었을때 픽셀이 바뀌는 지점이 1개인 글자들 -> ㄱ, ㄴ
            if 중앙_가로선_픽셀_변경_갯수 == 1:
                # 있으면 1
                # 없으면 0
                """ ㄱ, ㄴ 이 남은 상태에서 중앙 상단 픽셀을 검사한다. 있다면 ㄱ / 없다면 ㄴ 이다. """
                중앙상단픽셀여부 = 중앙상단픽셀검사(basic_image, basic_binary_image, point_dict)

                # 중앙에 상단픽셀이 검은색이라면 -> ㄱ
                if 중앙상단픽셀여부 == 1:
                    print('Giyeok')
                    contour_list.append(con)
                    hangul_list.append('Giyeok')
                # 중앙에 상단픽셀이 흰색이라면 -> ㄴ
                elif 중앙상단픽셀여부 == 0:
                    print('Nieun')
                    contour_list.append(con)
                    hangul_list.append('Nieun')
                # 잘못찾음
                else:
                    print('1 - 1 - 잘못찾음')

            """ 중앙선을 기준으로 가로로 나누었을때 픽셀이 바뀌는 지점이 1개인 글자들 -> ㅅ, ㅈ, ㅊ 얘네를 가지고 아래 if 문을 수행함. """
            if 중앙_가로선_픽셀_변경_갯수 == 2:

                """ 제일 왼쪽 부분과 중간 사이 세로선을 기준으로 나눈다."""
                가로선_0부터_중앙사이_픽셀_변경_갯수 = 가로선_0부터_중앙사이_세로로나누기(basic_image, basic_binary_image, point_dict)

                # 제일 왼쪽 부분과 중간 사이 세로선을 나누었을때 픽셀 변경 지점 개수가 2개인 부분 - > ㅅ
                # ㅅ
                if 가로선_0부터_중앙사이_픽셀_변경_갯수 == 2:
                    print('Sieut')
                    contour_list.append(con)
                    hangul_list.append('Sieut')

                # 제일 왼쪽 부분과 중간 사이 세로선을 나누었을때 픽셀 변경 지점 개수가 3개인 부분 - > ㅈ
                # ㅈ
                elif 가로선_0부터_중앙사이_픽셀_변경_갯수 == 3:
                    print('Tsiwt')
                    contour_list.append(con)
                    hangul_list.append('Tsiwt')

                # 제일 왼쪽 부분과 중간 사이 세로선을 나누었을때 픽셀 변경 지점 개수가 4개인 부분 - > ㅊ
                # ㅊ
                elif 가로선_0부터_중앙사이_픽셀_변경_갯수 == 4:
                    print('Tshiwt')
                    contour_list.append(con)
                    hangul_list.append('Tshiwt')

                # 잘못찾음
                else:
                    print('1 - 2 - 잘못찾음')

            """ 중앙선을 기준으로 가로로 나누었을때 픽셀이 바뀌는 지점이 1개인 글자들 -> ㅛ, ㅠ 얘네를 가지고 아래 if 문을 수행함. """
            if 중앙_가로선_픽셀_변경_갯수 == 4:
                아랫선_픽셀_변경_갯수 = 아랫선가로로나누기(basic_image, basic_binary_image, point_dict)

                # 제일 아래부분을 가로로 선을 그엇을때 픽셀이 변경되는 지점의 개수가 0개인 부분 -> ㅛ
                if 아랫선_픽셀_변경_갯수 == 0:
                    print('yo')
                    contour_list.append(con)
                    hangul_list.append('yo')
                # 잘못찾음
                else:
                    """ 제일 아래부분을 가로로 선을 그엇을때 픽셀이 변경되는 지점의 개수가 4개인 부분 -> ㅠ , ㅈ, ㅅ """
                    가로선_0부터_중앙사이_픽셀_변경_갯수 = 가로선_0부터_중앙사이_세로로나누기(basic_image, basic_binary_image, point_dict)

                    # ㅠ
                    if 가로선_0부터_중앙사이_픽셀_변경_갯수 == 1:
                        print('yu')
                        contour_list.append(con)
                        hangul_list.append('yu')

                    # ㅅ
                    elif 가로선_0부터_중앙사이_픽셀_변경_갯수 == 2:
                        print('Sieut')
                        contour_list.append(con)
                        hangul_list.append('Sieut')
                    # ㅈ
                    elif 가로선_0부터_중앙사이_픽셀_변경_갯수 == 3:
                        print('Tsiwt')
                        contour_list.append(con)
                        hangul_list.append('Tsiwt')

                    # 잘못찾음
                    else:
                        print('1 - 4 - 잘못찾음')

        # 중앙선을 기준으로 세로로 나누었을때 픽셀이 바뀌는 지점이 1개인 글자들 -> ㄷ, ㅇ, ㅍ, ㅁ, ㅏ, ㅓ 얘네를 가지고 아래 if 문을 수행함.
        elif 중앙_세로선_픽셀_변경_갯수 == 2:
            """ 중앙 상단에 픽셀이 있다면 1, 없다면 0 """
            중앙상단픽셀여부 = 중앙상단픽셀검사(basic_image, basic_binary_image, point_dict)

            # ㄷ, ㅇ, ㅍ, ㅁ
            if 중앙상단픽셀여부 == 1:
                """ 픽셀 변경 개수 0개 -> ㄷ, ㅁ, 2개 -> ㅇ, ㅍ"""
                가장왼쪽세로선_픽셀_변경_갯수 = 가장왼쪽세로선으로나누기(basic_image, basic_binary_image, point_dict)

                # ㄷ, ㅁ
                if 가장왼쪽세로선_픽셀_변경_갯수 == 0:
                    """ 오른쪽 중앙 픽셀 여부 / 있다 -> ㅁ , 없다 -> ㄷ """
                    중앙오른쪽픽셀여부 = 중앙오른쪽픽셀검사(basic_image, basic_binary_image, point_dict)

                    # ㅁ
                    if 중앙오른쪽픽셀여부 == 1:
                        print('Mieum')
                        contour_list.append(con)
                        hangul_list.append('Mieum')
                    # ㄷ
                    elif 중앙오른쪽픽셀여부 == 0:
                        print('Digeut')
                        contour_list.append(con)
                        hangul_list.append('Digeut')
                    else:
                        print('2 - 1 - 0 - 잘못찾음')
                # ㅇ, ㅍ
                elif 가장왼쪽세로선_픽셀_변경_갯수 == 2:

                    """ 왼쪽 상단 픽셀 여부 / 있다 -> ㅍ , 없다 -> ㅇ """
                    오른쪽중앙픽셀여부 = 중앙오른쪽픽셀검사(basic_image, basic_binary_image, point_dict)

                    # ㅍ
                    if 오른쪽중앙픽셀여부 == 0:
                        print('Pieub')
                        contour_list.append(con)
                        hangul_list.append('Pieub')
                    # ㅇ
                    elif 오른쪽중앙픽셀여부 == 1:
                        print('Ieung')
                        contour_list.append(con)
                        hangul_list.append('Ieung')
                    else:
                        print('2 - 1 - 2 - 잘못찾음')


                else:
                    print('2 - 1 - 잘못찾음')

            # ㅏ, ㅓ
            elif 중앙상단픽셀여부 == 0:
                """ 픽셀 변경 개수 0개 -> ㅏ, 2개 -> ㅓ"""
                가장왼쪽세로선_픽셀_변경_갯수 = 가장왼쪽세로선으로나누기(basic_image, basic_binary_image, point_dict)

                # ㅏ
                if 가장왼쪽세로선_픽셀_변경_갯수 == 0:
                    print('A')
                    contour_list.append(con)
                    hangul_list.append('A')
                # ㅓ
                elif 가장왼쪽세로선_픽셀_변경_갯수 == 2:
                    print('Eo')
                    contour_list.append(con)
                    hangul_list.append('Eo')
                else:
                    print('2 - 0 - 잘못찾음음')

            else:
                print('2 - 잘못찾음')

        # 중앙선을 기준으로 세로로 나누었을때 픽셀이 바뀌는 지점이 3개인 글자들 -> ㄱ, ㅂ, ㅋ 얘네를 가지고 아래 if 문을 수행함.
        elif 중앙_세로선_픽셀_변경_갯수 == 3:

            """ 픽셀변경 개수 0개 -> ㅂ, 1개 -> ㄱ, 3개 -> ㅋ"""
            가장왼쪽세로선_픽셀_변경_갯수 = 가장왼쪽세로선으로나누기(basic_image, basic_binary_image, point_dict)
            # ㅂ
            if 가장왼쪽세로선_픽셀_변경_갯수 == 0:
                print('Bieup')
                contour_list.append(con)
                hangul_list.append('Bieup')
            # ㄱ
            elif 가장왼쪽세로선_픽셀_변경_갯수 == 1 or 가장왼쪽세로선_픽셀_변경_갯수 == 2:
                print('Giyeok')
                contour_list.append(con)
                hangul_list.append('Giyeok')
            # ㅋ
            elif 가장왼쪽세로선_픽셀_변경_갯수 == 3:
                print('Kieuk')
                contour_list.append(con)
                hangul_list.append('Kieuk')
            else:
                print('3 - 잘못찾음')

        # 중앙선을 기준으로 세로로 나누었을때 픽셀이 바뀌는 지점이 4개인 글자들 -> ㄹ, ㅌ, ㅑ, ㅕ 얘네를 가지고 아래 if 문을 수행함.
        elif 중앙_세로선_픽셀_변경_갯수 == 4:

            """ 중앙 상단의 픽셀을 검사하였을때, 있다면 -> ㄹ, ㅌ 없다면 ㅑ, ㅕ 이다"""
            중앙상단픽셀여부 = 중앙상단픽셀검사(basic_image, basic_binary_image, point_dict)

            # 중앙에 상단픽셀이 검은색이라면 -> ㄹ, ㅌ
            if 중앙상단픽셀여부 == 1:

                """ 글자의 가장 왼쪽을 기준으로 세로선을 잘랐을때 픽셀이 바뀌는 지점을 체크한다. 2개 -> ㄹ, 0개 ㅌ """
                가장왼쪽세로선_픽셀_변경_갯수 = 가장왼쪽세로선으로나누기(basic_image, basic_binary_image, point_dict)

                # ㅌ
                if 가장왼쪽세로선_픽셀_변경_갯수 == 0:
                    print('Tieut')
                    contour_list.append(con)
                    hangul_list.append('Tieut')
                # ㄹ
                elif 가장왼쪽세로선_픽셀_변경_갯수 == 2:
                    print('Rieul')
                    contour_list.append(con)
                    hangul_list.append('Rieul')
                else:
                    print('4 - 1 - 잘못찾음')

            # 중앙에 상단픽셀이 흰색이라면 -> ㅑ, ㅕ
            elif 중앙상단픽셀여부 == 0:
                """ 값이 있다면 ㅑ, 없다면 ㅕ """
                왼쪽중앙픽셀여부 = 왼쪽중앙픽셀만검사(basic_image, basic_binary_image, point_dict)

                # ㅑ
                if 왼쪽중앙픽셀여부 == 1:
                    print('ya')
                    contour_list.append(con)
                    hangul_list.append('ya')
                # ㅕ
                elif 왼쪽중앙픽셀여부 == 0:
                    print('yeo')
                    contour_list.append(con)
                    hangul_list.append('yeo')
                else:
                    print('4 - 0 - 잘못찾음')

            # 잘못찾음
            else:
                print('4 - 잘못찾음')

        # 중앙선을 기준으로 세로로 나누었을때 픽셀이 바뀌는 지점이 4개인 글자들 -> ㅋ 얘네를 가지고 아래 if 문을 수행함.
        elif 중앙_세로선_픽셀_변경_갯수 == 5:
            print('Kieuk')
            contour_list.append(con)
            hangul_list.append('Kieuk')

    """####################################### ↓↓↓↓ 찾은 글자 네모영역 그리기 ↓↓↓↓###############################"""
    for i, find_contour_value in enumerate(contour_list):
        x, y, x_width, y_height = cv2.boundingRect(find_contour_value)

        cv2.rectangle(basic_image, (x, y), (x+x_width, y+y_height), (0, 0, 255), 1)
        cv2.putText(basic_image, hangul_list[i], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

    cv2.imshow('result_image', basic_image)
    cv2.waitKey(0)

    print("시간 : ", time.time() - start_time)
