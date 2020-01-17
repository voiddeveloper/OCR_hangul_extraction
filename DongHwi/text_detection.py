import cv2
import numpy as np
import time

# 색 필터 - 이진화 이미지 반환
def colorFilter(img_color, color_dict):
    image = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    result = img_color.copy()

    color = color_dict['color']

    # print(color_dict)

    if color == 'red':
        lower1 = np.array(color_dict['lower_range'])
        upper1 = np.array(color_dict['upper_range'])
        lower2 = np.array(color_dict['lower_range1'])
        upper2 = np.array(color_dict['upper_range1'])

        mask1 = cv2.inRange(image, lower1, upper1)
        mask2 = cv2.inRange(image, lower2, upper2)
        mask = mask1 + mask2

        result = cv2.bitwise_and(result, result, mask=mask)

    else:
        lower1 = np.array(color_dict['lower_range'])
        upper1 = np.array(color_dict['upper_range'])
        mask = cv2.inRange(image, lower1, upper1)
        result = cv2.bitwise_and(result, result, mask=mask)

    # if color == 'orange':
    #     cv2.imshow(str(color_dict['color'])+str(color_dict['s']) + str(' , ')+str(color_dict['v']), mask)
    #     cv2.waitKey(0)

    return mask


# 컨투어 찾기
def findContour(image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours,hierarchy


# 컨투어 x,y 최소, 최대 값 찾기
def findMinMaxPoint(bi_image, image, contour, hierarchy, filter_variable):
    min_max_list = []

    remove_flag = False

    h, w, c = image.shape

    for i, con in enumerate(contour):
        remove_flag = False

        # 글자는 최소 4개의 꼭지점으로 이루어져있다.
        # 따라서 3개 이하면 글자가 아니다
        if len(con) <= 3:
            continue

        x_min = 9999
        x_max = 0
        y_min = 9999
        y_max = 0
        for index, j in enumerate(contour[i]):

            if x_min > j[0][0]:
                x_min = j[0][0]

            if x_max < j[0][0]:
                x_max = j[0][0]

            if y_min > j[0][1]:
                y_min = j[0][1]

            if y_max < j[0][1]:
                y_max = j[0][1]

        """
        1번 필터
        가로 세로 길이가 모두 10픽셀 이하인 글자는 잡지 않겠다.
        """
        if (x_max - x_min) <= filter_variable['width_limit_pixel'] and (y_max - y_min) <= filter_variable['height_limit_pixel']:
            continue
        else:

            """
            2번 필터 : 네모영역 안에 픽셀양 체크
            2번 필터를 추가하면 조금 좋은 결과를 내놓음. 
            하지만 2번 필터를 추가 하나 안하나 비슷한거 같아서 일단 주석처리 해놓음 2020 -01 -07 / 
            시간, 정확도면에서 약간 오차가 있음 /"""
            # """
            # 2번 필터
            # 네모영역안의 픽셀을 검사해서 15~50% 미만이면 글자로 판단 그리고 95%이상이면 글자로 판단( ㅡ, ㅣ ), 그렇지 않으면 잡음으로 처리한다.
            # * 글자 ( 자음, 모음 ) 영역을 그렸을때 픽셀양이 45% 이하인걸 확인.
            # """
            # white_sum = 0
            #
            # for height in range(y_min, y_max):
            #     for width in range(x_min, x_max):
            #         if bi_image[height][width] == 255:
            #             white_sum += 1
            #
            # pixel_sum = ((white_sum) / ((x_max - x_min) * (y_max - y_min)))
            #
            # # 픽셀양이 10~50% 이하이면 네모영역 그림
            # if (pixel_sum * 100 <= 55 and pixel_sum * 100 >= 10) or pixel_sum * 100 > 95:

            """
            3번 필터
            컨투어 안에서 색상이 바뀌는 횟수
            """
            #픽셀의 색을 넣을 리스트
            ll=[]
            # 이미지와 동일한 크기의 검은색 이미지를 만든다
            cimg=np.zeros_like(image)
            # 검은색 이미지에 컨투어 크기만큼 흰색으로 그림
            # 컨투어 안에도 흰색으로 차있음
            cv2.drawContours(cimg,contour,i,color=255,thickness=-1)
            #hierarchy에 해당 컨투어의 인덱스를 포함하는것 모두 검색
            index,position=np.where(hierarchy[0]==i)
            for n in range(len(index)):
                #해당컨투어를 부모로 가진것만 검색해 자식 컨투어의 영역은 검은색으로 색칠
                if position[n]==3:
                    cv2.drawContours(cimg, contour, index[n], color=0, thickness=-1)
            #흰색 좌표를 저장함
            pts=np.where(cimg==255)
            # 원본 이미지에서 해당 좌표에 어떤 색이 채워져 있는지 저장
            ll.append(image[pts[0],pts[1]])
            #중복을 제거해주는 부분
            pixel_change_count = list(set([tuple(set(ll[0])) for ll[0] in ll[0]]))
            if len(pixel_change_count) <= filter_variable['pixel_change_count']:
                ddfd = cv2.imread('test1.png')
                op_draw = cv2.drawContours(image ,contour,i,color=(0,255,0),thickness=1)
                op_draw = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                cv2.imshow('sdf', op_draw)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # 모든 필터에 통과한다면 min_max_list에 네모영역 좌표를 추가한다.
                min_max_list.append([x_min, y_min, x_max, y_max])

    return min_max_list

if __name__ == '__main__':

    start_time = time.time()

    # rgb 이미지 불러오기
    img_color = cv2.imread('test1.png')

    # cv2.imshow('image', img_color)
    # cv2.waitKey(0)

    h, w, c = img_color.shape

    print(h, w, c)

    # rgb -> hsv 변환
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)

    # 색 필터 적용후 이미지 리스트
    result_filter_list = []

    """
    # h, v
    s_range : s의 범위 / int
    v_range : v 범위 / int 
    
    # 컨투어 안에서 색상이 바뀌는 횟수 / int
    filter_variable['pixel_change_count']
    
    # 가로픽셀이 10 이하인건 제외
    filter_variable['width_limit_pixel']

    # 세로픽셀이 10 이하인건 제외
    filter_variable['height_limit_pixel']
    """

    filter_variable = {}
    s_range = 10
    v_range = 10
    filter_variable['pixel_change_count'] = 10
    filter_variable['width_limit_pixel'] = 10
    filter_variable['height_limit_pixel'] = 10


    """ 검은색 """
    color_dict = {}
    color_dict['color'] = 'black'
    color_dict['lower_range'] = [0, 0, 0]
    color_dict['upper_range'] = [179, 0, 0]
    color_dict['s'] = 0
    color_dict['v'] = 0
    black_filter_result_image = colorFilter(img_color, color_dict)
    result_filter_list.append(black_filter_result_image)

    """ 하얀색 """
    color_dict = {}
    color_dict['color'] = 'white'
    color_dict['lower_range'] = [0, 0, 180]
    color_dict['upper_range'] = [179, 50, 255]
    color_dict['s'] = 0
    color_dict['v'] = 0
    white_filter_result_image = colorFilter(img_color, color_dict)
    result_filter_list.append(white_filter_result_image)

    for s in range(s_range):
        for v in range(v_range):

            if s * (256 / s_range) < 50 and v * (256 / v_range) > 180:
                continue

            # 색상 필터 적용해서 이미지에서 원하는 색을 뽑기
            """ 빨간색 """
            color_dict = {}
            color_dict['color'] = 'red'
            color_dict['lower_range'] = [170, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [179, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['lower_range1'] = [1, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range1'] = [9, s * (256 / s_range) + (256 / s_range),
                                          v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)

            red_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(red_filter_result_image)

            """ 주황색 """
            color_dict = {}
            color_dict['color'] = 'orange'
            color_dict['lower_range'] = [10, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [27, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            orange_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(orange_filter_result_image)

            """ 노란색 """
            color_dict = {}
            color_dict['color'] = 'yellow'
            color_dict['lower_range'] = [28, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [45, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            yellow_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(yellow_filter_result_image)

            """ 초록색 """
            color_dict = {}
            color_dict['color'] = 'green'
            color_dict['lower_range'] = [46, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [70, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            green_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(green_filter_result_image)

            """ 연초록하늘색 """
            color_dict = {}
            color_dict['color'] = 'skyblue'
            color_dict['lower_range'] = [71, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [95, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            skyblue_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(skyblue_filter_result_image)

            """ 파란색 """
            color_dict = {}
            color_dict['color'] = 'blue'
            color_dict['lower_range'] = [96, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [135, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            blue_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(blue_filter_result_image)

            """ 보라색 """
            color_dict = {}
            color_dict['color'] = 'purple'
            color_dict['lower_range'] = [136, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [169, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            purple_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append(purple_filter_result_image)

    """ 
    이진화 처리된 이미지에서 컨투어를 찾는다.
    컨투어를 찾고 주변 다른 컨투어와 묶었을때 정사각형과 비슷하게 나오면 글자로 추정한다. 
    """

    # 색 필터 이미지에서 컨투어를 찾고
    # x,y 최소 / x,y 최대 값을 찾아서 새로운 new_min_max_point_list 를 만든다.
    min_max_point_list = []
    new_min_max_point_list = []

    su = 0

    for bi_image in result_filter_list:

        contour,hierarchy = findContour(bi_image)
        point_list = findMinMaxPoint(bi_image, img_color, contour, hierarchy, filter_variable)

        su += len(point_list)

        for point in point_list:
            min_max_point_list.append(point)


    print('네모영역 개수 : ', su)

    print('time : ', time.time() - start_time)
    cv2.waitKey(0)
