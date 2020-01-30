###################################################
# 이미지에서 필터 적용하기 코드 + 필터를 거친 결과물 저장하기
# resultFolder에 저장되고 있음
###################################################
""" 여러 색이 있는 이미지에서 같은 계열의 색을 찾기"""

import cv2
import numpy as np
import time
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

img_number = 0

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

    # 특정 필터만 확인하는 디버깅용
    # if color == 'black':
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
def findMinMaxPoint(bi_image, image, contour, hierarchy, filter_variable, color_dict_info):
    min_max_list = []

    global img_number

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
            #
            # """
            # 3번 필터
            # 정사각형 제외 - 맑은고딕은 네모꼴임 그래서 자음 모음이 정사각형 모양이 나올 수 없음
            # """
            # if (x_max - x_min) / (y_max - y_min) == 1.0:
            #     continue

            #픽셀의 색을 넣을 리스트
            ll=[]
            # 이미지와 동일한 크기의 검은색 이미지를 만든다
            cimg=np.zeros_like(image)
            # 검은색 이미지(cimg)에 컨투어 크기만큼 흰색으로 그림
            # 컨투어 안에도 흰색으로 차있음
            ###########################################################################
            # 2020-01-30 황준환 작업
            # contour 안에 있는 자식 contour 영역은 흰색으로 칠하면 안된다.
            # 현재 모든 contour list를 가지고 있다보니, 해당 필터에 속하지 않는 부분도 같이 잡히고 있다. (자식 contour 영역은 무슨 색인지 알 수 없다.)
            # 따라서 hierarchy 값을 참조해서 부모가 몇번 나오는지 확인한다. 홀수번째 부모가 존재하는 contour는 색을 알수 없는 contour로 본다.
            # 다만, 짝수번째 부모가 존재하는 contour는 해당 필터에 속하는 색이 될 수도 있기 때문에, 이때는 예외처리하지 않는다.
            parentCount = 0

            ##### 테스트용, 필터를 통과한 모든 contour 정보를 확인한다. #####
            # testimg = np.zeros_like(image)
            # cv2.drawContours(testimg,contour,i,color=(255,255,255),thickness=-1)
            # cv2.imshow('testimg', testimg)
            # cv2.waitKey(0)
            ##################

            # 해당 contour의 부모가 존재 하는지 확인한다.
            # 부모가 있다면 해당 부모의 contour 리스트로 타고 올라간다.
            # 타고 올라간 다음, 그 contour도 부모가 있는지 확인한다.
            # 이 과정을 부모 contour가 없을 때까지 계속 반복하고, 총 몇번의 부모가 있었는지 계산한다.
            if hierarchy[0][i][3] != -1:
                parentCount += 1
                nextPoint = hierarchy[0][i][3]
                while True:
                    if hierarchy[0][nextPoint][3] == -1:
                        break
                    else:
                        nextPoint = hierarchy[0][nextPoint][3]
                        parentCount += 1

                # print ('parentCount 값 : ', parentCount)

                # 짝수번의 부모가 있었다면 이 contour 영역은 계산해봐야 하기 때문에, 흰색으로 칠한다.
                if parentCount%2 == 0:
                    cv2.drawContours(cimg,contour,i,color=(255,255,255),thickness=-1)

            # 부모가 아예 없다면 최상단 contour 이기 때문에, 흰색으로 칠한다.
            else:
                # print ('parentCount 값 : ', parentCount)
                cv2.drawContours(cimg,contour,i,color=(255,255,255),thickness=-1)
            ###########################################################################
            # 기존 구 코드는 주석 처리 해놓음, 위에 조건문을 주석처리한 다음, 아래줄의 주석을 해제하면 기존의 모든 contour를 흰색으로 칠하는 과정으로 돌아갈 수 있음.
            # cv2.drawContours(cimg,contour,i,color=(255,255,255),thickness=-1)

            # 색 필터를 뚫고 나온 contour 확인용 디버그
            # cv2.imshow('test1', cimg)
            # cv2.waitKey(0)

            #hierarchy에 해당 컨투어의 인덱스를 포함하는것 모두 검색
            index,position=np.where(hierarchy[0]==i)
            for n in range(len(index)):
                #해당컨투어를 부모로 가진것만 검색해 자식 컨투어의 영역은 검은색으로 색칠
                if position[n]==3:
                    # #####################################################################################
                    # 2020-01-27 황준환 작업
                    # # 자식 contour의 무게 중심을 구하고, 자식 컨투어의 상/하/좌/우를 한칸씩 땡기기 알고리즘 추가
                    # 2020-01-28 황준환 작업
                    # # 한칸씩 땡긴 영역의 배경색을 인식하는 문제가 발생, 이 영역을 인식해서는 안되기 때문에 다시 주석처리 함
                    # # M = 자식 contour의 moments 값
                    # M = cv2.moments(contour[index[n]])
                    # # cx, cy = 자식 contour의 무게중심 좌표
                    # if M['m00'] != 0:
                    #     cx = int(M['m10']/M['m00'])
                    #     cy = int(M['m01']/M['m00'])
                    #     # print (cx, cy)
                    #     # print("계산전", contour[index[n]])

                    #     for cnt in contour[index[n]]:
                    #         if cnt[0][0] < cx:
                    #             cnt[0][0] = cnt[0][0] + 1
                    #         elif cnt[0][0] > cx:
                    #             cnt[0][0] = cnt[0][0] - 1

                    #         if cnt[0][1] < cy:
                    #             cnt[0][1] = cnt[0][1] + 1
                    #         elif cnt[0][1] > cy:
                    #             cnt[0][1] = cnt[0][1] - 1

                    #     # print("계산후", contour[index[n]])
                    # #####################################################################################
                    cv2.drawContours(cimg, contour, index[n], color=0, thickness=-1)

            #흰색 좌표를 저장함
            pts=np.where(cimg==255)
            # print("pts 값 : ", pts)
            # print("값 : ", len(pts[0]))

            #############################################################################
            # 2020-01-30 황준환 작업
            # 새로 추가한 알고리즘으로 인해, 모든 영역이 배제되어서 pts 값이 하나도 없는 상황이 발생할 수 있다. (parentCount의 값이 홀수인 경우)
            # 이때는 cimg가 무조건 검은색이기 때문에, 이 경우에는 결과를 저장하지 않는다.
            if len(pts[0]) != 0:
                # 원본 이미지에서 해당 좌표에 어떤 색이 채워져 있는지 저장
                ll.append(image[pts[0],pts[1]])
                #중복을 제거해주는 부분
                pixel_change_count = list(set([tuple(set(ll[0])) for ll[0] in ll[0]]))
                if len(pixel_change_count) <= filter_variable['pixel_change_count']:
                    # op_draw = cv2.drawContours(image ,contour,i,color=(0,255,0),thickness=1)
                    # op_draw = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                    # cv2.imshow('sdf', op_draw)
                    # cv2.imshow('test', cimg)
                    # cv2.waitKey(0)

                    cimg = cimg[y_min:y_max+1, x_min:x_max+1]

                    file_name = "_color_" + str(color_dict_info['color']) + \
                                 "_s_range_" + str(color_dict_info['s_range']) + \
                                 "_v_range_" + str(color_dict_info['v_range']) + \
                                 "_count_" + str(color_dict_info['pixel_change_count'])

                    cv2.imwrite("resultFolder/" + str(img_number) + "_" + file_name + ".png", cimg)
                    img_number += 1


                    # 모든 필터를 통과한 후, 최종적으로 저장할 결과물을 확인하는 디버그
                    # cv2.imshow('test', cimg)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # 모든 필터에 통과한다면 min_max_list에 네모영역 좌표를 추가한다.
                    min_max_list.append([x_min, y_min, x_max+1, y_max+1])

    return min_max_list


if __name__ == '__main__':

    start_time = time.time()

    """
       # h, v
       s_range : s 범위 / int
       v_range : v 범위 / int 
       
       # 컨투어 안에서 색상이 바뀌는 횟수 / int
       filter_variable['pixel_change_count']
       
       # 가로픽셀이 10 이하인건 제외
       filter_variable['width_limit_pixel']
    
       # 세로픽셀이 10 이하인건 제외
       filter_variable['height_limit_pixel']
       """

    filter_variable = {}
    s_range = 20
    v_range = 20
    width_limit_pixel = 5
    height_limit_pixel = 5
    pixel_change_count = 30
    filter_variable['width_limit_pixel'] = width_limit_pixel
    filter_variable['height_limit_pixel'] = height_limit_pixel
    filter_variable['pixel_change_count'] = pixel_change_count

    # rgb 이미지 불러오기
    img_color = cv2.imread('test1.png')

    # cv2.imshow('image', img_color)
    # cv2.waitKey(0)

    h, w, c = img_color.shape

    print(h, w, c)

    # rgb -> hsv 변환
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)
    img_yuv = cv2.cvtColor(img_color, cv2.COLOR_RGB2YUV)

    # 색 필터 적용후 이미지 리스트
    result_filter_list = []

    """ 검은색 """
    color_dict = {}
    color_dict['color'] = 'black'
    color_dict['lower_range'] = [0, 0, 0]
    color_dict['upper_range'] = [179, 255, 50]
    color_dict['s'] = 0
    color_dict['v'] = 0
    color_dict['s_range'] = s_range
    color_dict['v_range'] = v_range
    color_dict['width_limit_pixel'] = width_limit_pixel
    color_dict['height_limit_pixel'] = height_limit_pixel
    color_dict['pixel_change_count'] = pixel_change_count
    black_filter_result_image = colorFilter(img_color, color_dict)
    result_filter_list.append([black_filter_result_image, color_dict])

    """ 하얀색 """
    color_dict = {}
    color_dict['color'] = 'white'
    color_dict['lower_range'] = [0, 0, 180]
    color_dict['upper_range'] = [179, 50, 255]
    color_dict['s'] = 0
    color_dict['v'] = 0
    color_dict['s_range'] = s_range
    color_dict['v_range'] = v_range
    color_dict['width_limit_pixel'] = width_limit_pixel
    color_dict['height_limit_pixel'] = height_limit_pixel
    color_dict['pixel_change_count'] = pixel_change_count
    white_filter_result_image = colorFilter(img_color, color_dict)
    result_filter_list.append([white_filter_result_image, color_dict])

    for s in range(s_range):
        for v in range(v_range):

            # 하얀색 필터가 겹치면 제외
            if s * (256 / s_range) < 50 and v * (256 / v_range) > 180:
                continue

            # 검은색 필터가 겹치면 제외
            if s * (256 / s_range) < 255 and v * (256 / v_range) < 50:
                continue

            # 색상 필터 적용해서 이미지에서 원하는 색을 뽑기
            """ 빨간색 """
            color_dict = {}
            color_dict['color'] = 'red'
            color_dict['lower_range'] = [170, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [180, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['lower_range1'] = [0, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range1'] = [9, s * (256 / s_range) + (256 / s_range),
                                          v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count

            red_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([red_filter_result_image, color_dict])

            """ 주황색 """
            color_dict = {}
            color_dict['color'] = 'orange'
            color_dict['lower_range'] = [10, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [27, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count
            orange_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([orange_filter_result_image, color_dict])

            """ 노란색 """
            color_dict = {}
            color_dict['color'] = 'yellow'
            color_dict['lower_range'] = [28, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [45, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count
            yellow_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([yellow_filter_result_image, color_dict])

            """ 초록색 """
            color_dict = {}
            color_dict['color'] = 'green'
            color_dict['lower_range'] = [46, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [70, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count
            green_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([green_filter_result_image, color_dict])

            """ 연초록하늘색 """
            color_dict = {}
            color_dict['color'] = 'skyblue'
            color_dict['lower_range'] = [71, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [95, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count
            skyblue_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([skyblue_filter_result_image, color_dict])

            """ 파란색 """
            color_dict = {}
            color_dict['color'] = 'blue'
            color_dict['lower_range'] = [96, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [135, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count
            blue_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([blue_filter_result_image, color_dict])

            """ 보라색 """
            color_dict = {}
            color_dict['color'] = 'purple'
            color_dict['lower_range'] = [136, s * (256 / s_range), v * (256 / v_range)]
            color_dict['upper_range'] = [169, s * (256 / s_range) + (256 / s_range),
                                         v * (256 / v_range) + (256 / v_range)]
            color_dict['s'] = s * (256 / s_range)
            color_dict['v'] = v * (256 / v_range)
            color_dict['s_range'] = s_range
            color_dict['v_range'] = v_range
            color_dict['width_limit_pixel'] = width_limit_pixel
            color_dict['height_limit_pixel'] = height_limit_pixel
            color_dict['pixel_change_count'] = pixel_change_count
            purple_filter_result_image = colorFilter(img_color, color_dict)
            result_filter_list.append([purple_filter_result_image, color_dict])

    """ 
    이진화 처리된 이미지에서 컨투어를 찾는다.
    컨투어를 찾고 주변 다른 컨투어와 묶었을때 정사각형과 비슷하게 나오면 글자로 추정한다. 
    """

    # 색 필터 이미지에서 컨투어를 찾고
    # x,y 최소 / x,y 최대 값을 찾아서 새로운 new_min_max_point_list 를 만든다.
    min_max_point_list = []
    new_min_max_point_list = []
    point_list = []

    # print('result_filter_list : ', result_filter_list)

    for bi_image,color_dict_info in result_filter_list:

        contour, hierarchy = findContour(bi_image)
        po = findMinMaxPoint(bi_image, img_color, contour, hierarchy, filter_variable,color_dict_info)

        if po:
            for i in po:
                point_list.append(i)

    point_list = list(set(map(tuple, point_list)))
    # print('point_list : ', point_list)

    for x_min, y_min, x_max, y_max in point_list:
        op_draw = cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        cv2.imshow('asfd',op_draw)

    print('네모영역 개수 : ', len(point_list))

    print('time : ', time.time() - start_time)
    cv2.waitKey(0)
