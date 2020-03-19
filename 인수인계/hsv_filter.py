"""

# 이미지에서 필터 적용하기 코드 + 필터를 거친 결과물 저장하기
# resultFolder에 저장되고 있음
#
# ## 목표 ##
# 이미지에서 글씨에는 색이 있다. 그리고 글자의 색은 대부분 일치하는 편이다.
# 그렇다면 특정한 색만 인식하는 mask를 생성하고, 이 mask를 적용한다.
# 글자의 색이 mask 영역 안에 포함되어 있다면 글자만 검출되고, 주변 다른 색들은 검출되지 않을 것이다.
# 단, 이 알고리즘에서는 글자 내부에서 색이 변하는 글씨는 찾지 않는다.



이 코드는 이미지에서 글자는 같은 색으로 이루어져 있다는 가정을 하고 시작한다.

bgr 이미지를  hsv 이미지로 바꾸어 모든 색 대역을 여러 조각으로 나눈다.

예를 들면, 빨간색의 대역, 진한 빨간색의 대역, 노란색의 대역 이런식으로 말이다.

색상 대역을 뽑아낸 후, 각 색상 영역에서 컨투어를 찾는다.

그리고 찾아낸 컨투어들 중에서 글자스러움 필터를 적용하여 글자스러움 조건에 맞는 컨투어만 살아남는다.

최종적으로 글자라고 생각되는것 들만 결과이미지에 표시해준다.

**

기본적으로 hsv 표현법에 대한 이해가 필요하며, OpenCV에서 hsv의 범위가 어떻게 표기되고있는지 알아야한다.
그리고 컨투어에 대한 개념 이해도 필요하다.
OpenCV - hsv => https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html 참고
OpenCV - contour => https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html

ex ) 하얀색과 검은색이 hsv에선 어떠한 값에서 나타나는가?

"""


###################################################
# 이미지에서 필터 적용하기 코드 + 필터를 거친 결과물 저장하기
# resultFolder에 저장되고 있음
###################################################
""" 여러 색이 있는 이미지에서 같은 계열의 색을 찾기"""

import cv2
import numpy as np
import time

# 색 필터 - 이진화 이미지 반환한다.
# img_color는 원본인 bgr 이미지를 인자로 받는다.
# color_dict은 dictionary 구조로 되어있으며,
# 그 안의 key값은 색상, 색상범위, hsv의 s 및 v범위,
# 글자의 가로길이 제한, 세로길이 제한, 글자의 색이 바뀌는 횟수 등의 키 값이 들어있다.
def colorFilter(img_color, color_dict):
    image = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    color = color_dict['color']

    # 빨간색 색상 필터 : 빨간색은 hsv에서 h의 범위 가 0~10, 170~180 두 군데 분포하기에 2개의 범위를 더해주었음
    if color == 'red':
        lower1 = np.array(color_dict['lower_range'])
        upper1 = np.array(color_dict['upper_range'])
        lower2 = np.array(color_dict['lower_range1'])
        upper2 = np.array(color_dict['upper_range1'])

        # 해당 색상의 범위 색만 추출한다.
        mask1 = cv2.inRange(image, lower1, upper1)
        mask2 = cv2.inRange(image, lower2, upper2)
        mask = mask1 + mask2

    # 빨간색을 제외한 나머지 색상
    else:
        lower1 = np.array(color_dict['lower_range'])
        upper1 = np.array(color_dict['upper_range'])

        # 해당 색상의 범위 색만 추출한다.
        mask = cv2.inRange(image, lower1, upper1)

    # mask에는 원본 bgr 이미지에서 뽑아내고자 하는 색상의 대역만 남게된다

    # 특정 필터만 확인하는 디버깅용
    # if color == 'black':
    #     cv2.imshow(str(color_dict['color'])+str(color_dict['s']) + str(' , ')+str(color_dict['v']), mask)
    #     cv2.waitKey(0)

    return mask


# 컨투어 찾기 - 이진화 이미지가 들어온다.
def findContour(binary_image):
    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 반환
    return contours, hierarchy


# 컨투어 x,y 최소, 최대 값 찾기
# filter_variable은 딕셔너리 구조로 되어있다.
# 안에 값은 글자의 최소 가로길이, 글자의 최소 세로길이, 컨투어 안에서 색상이 바뀌는 횟수가 들어있다.
def findMinMaxPoint(image, contour, hierarchy, filter_variable):
    # min_max_list: 이 메소드의 최종 리턴값, contour의 최소, 최대값 좌표가 저장된다.
    min_max_list = []

    for i, con in enumerate(contour):

        # x: contour의 시작점 x 좌표
        # y: contour의 시작점 y 좌표
        # x_width: contour의 가로 길이
        # y_height: contour의 세로 길이
        x, y, x_width, y_height = cv2.boundingRect(con)

        """
        1번 필터
        가로 길이가 'width_limit_pixel' 픽셀 이하인 글자 + 세로 길이가 'height_limit_pixel' 픽셀 이하인 글자는 잡지 않겠다.
        """
        if x_width <= filter_variable['width_limit_pixel'] and y_height <= filter_variable[
            'height_limit_pixel']:
            continue
        else:
            # pixel_color_list: 픽셀의 색을 넣을 리스트
            pixel_color_list = []
            # 이미지와 동일한 크기의 검은색 이미지를 만든다
            cimg = np.zeros_like(image)
            ###########################################################################
            # 2020-01-30 황준환 작업
            # contour 안에 있는 자식 contour 영역은 흰색으로 칠하면 안된다.
            # 현재 모든 contour list를 가지고 있다보니, 해당 필터에 속하지 않는 부분도 같이 잡히고 있다. (자식 contour 영역은 무슨 색인지 알 수 없다.)
            # 따라서 hierarchy 값을 참조해서 부모가 몇번 나오는지 확인한다. 홀수번째 부모가 존재하는 contour는 색을 알수 없는 contour로 본다.
            # 다만, 짝수번째 부모가 존재하는 contour는 해당 필터에 속하는 색이 될 수도 있기 때문에, 이때는 예외처리하지 않는다.
            
            # parentCount: 몇번째 부모인지 확인하는 용도
            parentCount = 0

            # 해당 contour의 부모가 존재 하는지 확인한다.
            # 부모가 있다면 해당 부모의 contour 리스트로 타고 올라간다.
            # 타고 올라간 다음, 그 contour도 부모가 있는지 확인한다.
            # 이 과정을 부모 contour가 없을 때까지 계속 반복하고, 총 몇번의 부모가 있었는지 계산한다.
            crop_image=np.zeros_like(image)
            
            if hierarchy[0][i][3] != -1:
                parentCount += 1
                # 다음에 검색할 hierarchy 값
                nextPoint = hierarchy[0][i][3]

                while True:
                    # 다음에 검색할 hierarchy 값의 부모가 없다면, 반복을 멈춘다.
                    if hierarchy[0][nextPoint][3] == -1:
                        break
                    else:
                        nextPoint = hierarchy[0][nextPoint][3]
                        parentCount += 1

                # 짝수번의 부모가 있었다면 이 contour 영역은 계산해봐야 하기 때문에, 흰색으로 칠한다.
                if parentCount % 2 == 0:
                    cv2.drawContours(cimg, contour, i, color=(255, 255, 255), thickness=-1)
                    cv2.drawContours(crop_image, contour, i, color=(255, 255, 255), thickness=-1)

            # 부모가 아예 없다면 최상단 contour 이기 때문에, 흰색으로 칠한다.
            else:
                cv2.drawContours(cimg, contour, i, color=(255, 255, 255), thickness=-1)
                cv2.drawContours(crop_image, contour, i, color=(255, 255, 255), thickness=-1)
            
            # hierarchy에 해당 contour의 인덱스를 포함하는 것을 모두 검색한다.
            index, position = np.where(hierarchy[0] == i)

            for n in range(len(index)):
                # 해당컨투어를 부모로 가진것만 검색해 자식 컨투어의 영역은 검은색으로 색칠한다.
                if position[n] == 3:
                    cv2.drawContours(cimg, contour, index[n], color=(0,0,0), thickness=-1)
                    # #####################################################################################
                    # 2020-01-27 황준환 작업
                    # # 자식 contour의 무게 중심을 구하고, 자식 컨투어의 상/하/좌/우를 한칸씩 땡기기 알고리즘 추가
                    M = cv2.moments(contour[index[n]])
                    # cx, cy = 자식 contour의 무게중심 좌표
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])

                        for cnt in contour[index[n]]:
                            if cnt[0][0] < cx:
                                cnt[0][0] = cnt[0][0] + 1
                            elif cnt[0][0] > cx:
                                cnt[0][0] = cnt[0][0] - 1

                            if cnt[0][1] < cy:
                                cnt[0][1] = cnt[0][1] + 1
                            elif cnt[0][1] > cy:
                                cnt[0][1] = cnt[0][1] - 1
                        cv2.drawContours(crop_image, contour, index[n], color=(0,0,0), thickness=-1)
                    # #####################################################################################

            # 흰색으로 칠해진 좌표를 pts에 저장함
            pts = np.where(cimg == 255)

            #############################################################################
            # 2020-01-30 황준환 작업
            # 새로 추가한 알고리즘으로 인해, 모든 영역이 배제되어서 pts 값이 하나도 없는 상황이 발생할 수 있다. (parentCount의 값이 홀수인 경우)
            # 이때는 cimg가 무조건 검은색이기 때문에, 이 경우에는 결과를 저장하지 않는다.
            if len(pts[0]) != 0:
                # 원본 이미지에서 해당 좌표에 어떤 색이 채워져 있는지 저장
                pixel_color_list.append(image[pts[0], pts[1]])

                # pixel_change_count: contour 안에서 색상이 바뀌는 횟수
                # pixel_color_list에 존재하는 색상의 개수를 구한다.
                pixel_change_count = list(set([tuple(set(pixel_color_list[0])) for pixel_color_list[0] in pixel_color_list[0]]))
                
                # pixel_color_list의 갯수가 pixel_change_count보다 적다면, 글씨 contour일 것이다.
                # 해당 contour를 저장한다.
                if len(pixel_change_count) <= filter_variable['pixel_change_count']:
                    crop_image = crop_image[y:y+y_height + 1, x:x+x_width + 1]
                    # 2020 - 01 - 31 ( 동휘작업)
                    # crop이미지를 분석하여 잡음을 제거하는 메서드
                    remove_noise_flag = removeNoise(crop_image)

                    if remove_noise_flag == False:
                        # 2020-01-30 동휘 작업 ) 크롭한 이미지 저장할때 파일 이름
                        file_name = "_color_" + str(color_dict_info['color']) + \
                                    "_s_range_" + str(color_dict_info['s_range']) + \
                                    "_v_range_" + str(color_dict_info['v_range']) + \
                                    "_count_" + str(color_dict_info['pixel_change_count'])

                        # 모든 필터에 통과한다면 min_max_list에 네모영역 좌표를 추가한다.
                        min_max_list.append([x, y, x+x_width+1, y+y_height+1])

    return min_max_list

# 2020 - 01 - 31 ( 동휘작업)
# crop한 이미지를 분석하여 잡음을 제거하는 메서드
# crop_image: crop한 이미지 파일
def removeNoise(crop_image):
    h,w,c = crop_image.shape

    # ratio1 : 세로 / 가로
    # ratio2 : 가로 / 세로
    ratio1 = (h/w)
    ratio2 = (w/h)

    # error: 종횡비 기준
    error = 5

    # 글자가 차지하는 영역을 세기 위한 변수, 흰색일때만 카운트함
    count = 0

    for i in range(0,h):
        for j in range(0,w):
            color = crop_image[i][j][0]

            """ 
            필터 1 : 
            글자라고 생각되는 부분은 흰색으로 채워져있다.
            1. 흰색 픽셀 위,아래,왼쪽,오른쪽을 체크한다.
            2. 체크한 픽셀중에서 흰색이 하나도 없다면 잡음으로 처리한다.
            """
            try:
                top = crop_image[i-1][j][0]
            except:
                top = crop_image[i][j][0]
            try:
                left = crop_image[i][j-1][0]
            except:
                left = crop_image[i][j][0]
            try:
                down = crop_image[i+1][j][0]
            except:
                down = crop_image[i][j][0]
            try:
                right = crop_image[i][j+1][0]
            except:
                right = crop_image[i][j][0]

            # 체크한 픽셀의 유무를 확인하는 용도
            flag_count = 0

            if color == 255:
                if top == 255:
                    flag_count += 1
                if left == 255:
                    flag_count += 1
                if down == 255:
                    flag_count += 1
                if right == 255:
                    flag_count += 1

                # 체크한 픽셀이 하나도 없다면 잡음
                if flag_count == 0:
                    return True
                count += 1

    # 글자가 차지하는 영역의 비율
    percent = int((count * 100) / (h * w))

    # 잡음
    """
        return True -> 잡음
        return False -> 글자
    """

    """ 
    필터 2 글자가 차지하는 영역이 몇 퍼센트인가?
    글자가 차지하는 영역이 너무 작거나, 너무 많다면 글자가 아니다 
    """
    # 잡음
    if percent < 10 or 70 < percent and percent < 95:
        return True
    elif percent >= 95:
        """ 
        필터 3 : 종횡비
        글자라고 생각되는 영역이 95%이상일때,
        종횡비를 구하여 종횡비가 설정한 오차 이상일때만 글자로 인정
        ex) 영역안에 뚱뚱한 네모가 잡혔을때 -> 제거 할 수 있음 
        """
        # 종횡비
        # ratio1 : 세로 / 가로
        # ratio2 : 가로 / 세로
        # error: 종횡비 기준

        # 잡음
        if ratio1 < error and ratio2 < error:
            return True
        # 글자
        else:
            return False
    # 글자
    else:
        return False


if __name__ == '__main__':

    start_time = time.time()
    """####################################################################################"""
    """ ####################### 1. 색상필터 및 글자 조건 설정 하기 #########################"""
    """####################################################################################"""

    filter_variable = {}
    s_range = 40 # h,s,v 중 s값 s 값을 40등분 하겠다.
    v_range = 20 # h,s,v 중 v값 v 값을 20등분 하겠다.
    width_limit_pixel = 10 # 글자의 최소 가로길이
    height_limit_pixel = 10 # 글자의 최소 세로길이
    pixel_change_count = 10 # 컨투어 안에서 색상이 바뀌는 횟수
    filter_variable['width_limit_pixel'] = width_limit_pixel # 글자의 최소 가로길이
    filter_variable['height_limit_pixel'] = height_limit_pixel # 글자의 최소 세로길이
    filter_variable['pixel_change_count'] = pixel_change_count # 컨투어 안에서 색상이 바뀌는 횟수

    """####################################################################################"""
    """ ################################## 2. 이미지 불러오기 ##############################"""
    """####################################################################################"""

    # rgb 이미지 불러오기
    img_color = cv2.imread('../test1.png')
    h, w, c = img_color.shape

    """####################################################################################"""
    """ ############################ 3. bgr -> hsv 이미지로 변환하기 ########################"""
    """####################################################################################"""

    # rgb -> hsv 변환
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)


    """####################################################################################"""
    """ ############################ 4. 원하는 색의 대역만 뽑아내는 과정 ###################"""
    """####################################################################################"""
    # colorFilter() 는 내가 입력한 범위의 색상 대역만 추출하여 이미지를 반환해준다.

    # 색 필터 적용 한 이미지를 보관하기 위한 리스트 : 이 이미지에서 컨투어를 뽑아내어 글자인지 판단하는 작업을 할 예정이다.
    result_filter_list = []

    # 검은색과 하얀색은 색이 없다라고 표현을 한다고 함.
    # 그래서 빨주노초파남보와 같은 색과는 별개로 따로 분리해서 추출한다.

    # 검은색 색상 필터 설정 값 및 검은색 대역 이미지 뽑아내기
    color_dict = {}
    color_dict['color'] = 'black'
    color_dict['lower_range'] = [0, 0, 0] # 검은색 대역의 최소 범위
    color_dict['upper_range'] = [179, 255, 50] # 검은색 대역의 최대 범위
    color_dict['s'] = 0
    color_dict['v'] = 0
    color_dict['s_range'] = s_range
    color_dict['v_range'] = v_range
    color_dict['width_limit_pixel'] = width_limit_pixel
    color_dict['height_limit_pixel'] = height_limit_pixel
    color_dict['pixel_change_count'] = pixel_change_count
    black_filter_result_image = colorFilter(img_color, color_dict)
    result_filter_list.append([black_filter_result_image, color_dict])

    # 하얀색 색상 필터 설정 값 및 하얀색 대역 이미지 뽑아내기
    color_dict = {}
    color_dict['color'] = 'white'
    color_dict['lower_range'] = [0, 0, 180] # 하얀색 대역의 최대 범위
    color_dict['upper_range'] = [179, 50, 255] # 하얀색 대역의 최대 범위
    color_dict['s'] = 0
    color_dict['v'] = 0
    color_dict['s_range'] = s_range
    color_dict['v_range'] = v_range
    color_dict['width_limit_pixel'] = width_limit_pixel
    color_dict['height_limit_pixel'] = height_limit_pixel
    color_dict['pixel_change_count'] = pixel_change_count
    white_filter_result_image = colorFilter(img_color, color_dict)
    result_filter_list.append([white_filter_result_image, color_dict])

    # 1번에서 설정한 s_range, v_range값은 40, 20이였다.
    # 총 색상필터를 적용할 이미지는 40 * 20 * 7(빨, 주, 노, 초, 연초록하늘, 파랑, 보라)개 이다.
    for s in range(s_range):
        for v in range(v_range):

            # 하얀색 필터와 영역이 겹치면 제외
            if s * (256 / s_range) < 50 and v * (256 / v_range) > 180:
                continue

            # 검은색 필터와 영역이 겹치면 제외
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


    """####################################################################################"""
    """ ############ 5. 색상필터에서 걸러진 이미지로 부터 글자스러운 컨투어를 찾는 과정 #######"""
    """####################################################################################"""

    # 색 필터 이미지에서 컨투어를 찾고
    # x,y 최소 / x,y 최대 값을 찾아서 새로운 new_min_max_point_list 를 만든다.
    min_max_point_list = []
    new_min_max_point_list = []
    result_point_list = [] # 최종 글자스러움을 가진 영역 좌표만 넣는 변수

    for bi_image, color_dict_info in result_filter_list:
        # 색상 필터에서 나온 이미지에서 컨투어를 찾는과정
        contour, hierarchy = findContour(bi_image)

        # findMinMaxPoint() - 글자스러운 컨투어를 찾으면서 글자의 영역 좌표를 반환해주는 함수
        contour_position = findMinMaxPoint(img_color, contour, hierarchy, filter_variable)

        # 글자스러운 컨투어가 있다면 해당 컨투어의 네모 영역 좌표를 저장한다.
        if contour_position:
            for i in contour_position:
                result_point_list.append(i)

    # 중복된 영역이 있다며 제거한다.
    result_point_list = list(set(map(tuple, result_point_list)))

    """####################################################################################"""
    """ ############ 6. 최종적으로 이미지위에 글자스러움을 가진 좌표영역을 빨간색으로 그린다. #######"""
    """####################################################################################"""
    for x_min, y_min, x_max, y_max in result_point_list:
        cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        cv2.imshow('result_image', img_color)

    print('네모영역 개수 : ', len(result_point_list))
    print('time : ', time.time() - start_time)
    cv2.waitKey(0)
