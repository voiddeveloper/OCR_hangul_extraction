import cv2
import json

def 이미지_이진화_및_컨투어_찾기(image):
    # rgb -> gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 이미지 이진화 0 - 흑 , 255 백으로 나눔
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    try:
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        no, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('image_binary', img_binary)
    # cv2.waitKey(0)

    # 반환값은 이진화된 이미지와, 컨투어 값임
    return img_binary, contours


def 자음_모음_컨투어_구분(image, contour):
    contour = contour.copy()
    image_copy = image.copy()

    자음_contour_list = []
    모음_contour_list = []

    for i, con in enumerate(contour):
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

        width = x_max - x_min
        height = y_max - y_min

        # 모음 조건 : 맑은 고딕 기준
        # 가로의 길이가 세로의 2배보다 짧으면 모음
        # 세로의 길이가 가로의 1.7배보다 짧으면 모음
        if width * 2 < height or height * 1.7 < width:
            모음_contour_list.append(con)
            # image_copy = cv2.rectangle(image_copy, (x_min, y_min),(x_max, y_max), (0, 0, 255), 2)

        # 자음 조건
        else:
            자음_contour_list.append(con)
            # image_copy = cv2.rectangle(image_copy, (x_min, y_min),(x_max, y_max), (0, 255, 0), 2)

        # cv2.imshow('sdf',image_copy)
        # cv2.waitKey(0)

    return 자음_contour_list, 모음_contour_list


def 픽셀수_세기and사각형4등분(이미지, 이진화_이미지, 정답, y_min, y_max, x_min, x_max):

    for x in range(y_min, y_max):
        for y in range(x_min, x_max):
            if 이진화_이미지[x][y] == 255:
                정답 += 1

    정답 = round(((정답 * 100) / ((x_max - x_min) * (y_max - y_min))),2)
    print("ㅃㅈㄷㅂㅈㄷ", 정답)

    draw = cv2.rectangle(이미지, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    cv2.line(draw, (x_min, int((y_max - y_min) / 2 + y_min)), (x_max, int((y_max - y_min) / 2 + y_min)), (0, 0, 255), 1)
    cv2.line(draw, (int((x_max - x_min) / 2 + x_min), y_min), (int((x_max - x_min) / 2 + x_min), y_max), (0, 0, 255), 1)

    왼쪽상단_픽셀 = 0
    오른쪽상단_픽셀 = 0
    왼쪽하단_픽셀 = 0
    오른쪽하단_픽셀 = 0
    # 왼쪽 상단
    print("왼쪽상단")

    for i in range(y_min, int((y_max - y_min) / 2 + y_min)):
        for j in range(x_min, int((x_max - x_min) / 2 + x_min)):
            if 이진화_이미지[i][j] == 0:
                pass
            else:
                왼쪽상단_픽셀 = 왼쪽상단_픽셀 + 1

    print("오른쪽상단")
    # 오른쪽 상단
    for i in range(y_min, int((y_max - y_min) / 2 + y_min)):
        for j in range(int((x_max - x_min) / 2 + x_min), x_max):
            if 이진화_이미지[i][j] == 0:
                pass
            else:
                오른쪽상단_픽셀 = 오른쪽상단_픽셀 + 1

    print("오른쪽하단")
    # 오른쪽 하단
    for i in range(int((y_max - y_min) / 2 + y_min), y_max):
        for j in range(int((x_max - x_min) / 2 + x_min), x_max):
            if 이진화_이미지[i][j] == 0:
                pass
            else:
                오른쪽하단_픽셀 = 오른쪽하단_픽셀 + 1

    print("왼쪽 하단")
    for i in range(int((y_max - y_min) / 2 + y_min), y_max):
        for j in range(x_min, int((x_max - x_min) / 2 + x_min)):
            if 이진화_이미지[i][j] == 0:
                pass
            else:
                왼쪽하단_픽셀 = 왼쪽하단_픽셀 + 1
        list_ = []
        list_.append(왼쪽상단_픽셀)
        list_.append(오른쪽상단_픽셀)
        list_.append(왼쪽하단_픽셀)
        list_.append(오른쪽하단_픽셀)
        max_pixel = max(list_)
        # print(max_pixel)
        픽셀많은부분=""
        if max_pixel == 왼쪽상단_픽셀:
            cv2.circle(draw, (x_min, y_min), 2, (255, 0, 0), 2)
            픽셀많은부분="왼쪽상단"
        if max_pixel == 오른쪽상단_픽셀:
            cv2.circle(draw, (x_max, y_min), 2, (255, 0, 0), 2)
            픽셀많은부분="오른쪽상단"
        if max_pixel == 왼쪽하단_픽셀:
            cv2.circle(draw, (x_min, y_max), 2, (255, 0, 0), 2)
            픽셀많은부분="왼쪽하단"
        if max_pixel == 오른쪽하단_픽셀:
            cv2.circle(draw, (x_max, y_max), 2, (255, 0, 0), 2)
            픽셀많은부분="오른쪽하단"
        try:
            왼쪽상단_비율=round(max_pixel/왼쪽상단_픽셀,2)
        except:
            왼쪽상단_비율=0
        try:
            오른쪽상단_비율=round(max_pixel/오른쪽상단_픽셀,2)
        except:
            오른쪽상단_비율=0
        try:
            왼쪽하단_비율=round(max_pixel/왼쪽하단_픽셀,2)
        except:
            왼쪽하단_비율=0
        try:
            오른쪽하단_비율=round(max_pixel/오른쪽하단_픽셀,2)
        except:
            오른쪽하단_비율=0
        비율=[]
        비율.append(왼쪽상단_비율)
        비율.append(오른쪽상단_비율)
        비율.append(왼쪽하단_비율)
        비율.append(오른쪽하단_비율)

    # 픽셀수 반환 , 사각형 그린 이미지 반환
    return 비율,픽셀많은부분,정답,왼쪽상단_픽셀, 오른쪽상단_픽셀, 왼쪽하단_픽셀, 오른쪽하단_픽셀, draw


def 컨투어_박스(이미지경로, 이진화_image, my_image_contour):
    result={}

    이미지 = cv2.imread(이미지경로, cv2.IMREAD_COLOR)

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
        정답 = 0
        비율,픽셀많은부분,전체픽셀비율,왼쪽상단_픽셀, 오른쪽상단_픽셀, 왼쪽하단_픽셀, 오른쪽하단_픽셀, 픽셀4등분이미지 = 픽셀수_세기and사각형4등분(이미지, 이진화_image, 정답, y_min, y_max, x_min, x_max)
        dic={}
        dic['전체픽셀']=전체픽셀비율
        dic['x_min']=x_min
        dic['x_max']=x_max
        dic['y_min']=y_min
        dic['y_max']=y_max
        dic['가장많은부분']=픽셀많은부분
        dic['비율']=비율
        # print(dic)
        print(왼쪽상단_픽셀, 오른쪽상단_픽셀)
        print(왼쪽하단_픽셀, 오른쪽하단_픽셀)
        print("")
        result[len(result)]=dic
    cv2.imshow('sdf', 픽셀4등분이미지)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("")
    return  result

if __name__ == '__main__':
    """
    이미지 불러오기 : 폰트 맑은 고딕
    my_image : 비교할 이미지
    """
    my_image_path = 'ka.png'
    my_image = cv2.imread(my_image_path, cv2.IMREAD_COLOR)
    my_image_binary, my_image_contour = 이미지_이진화_및_컨투어_찾기(my_image)
    원본 = 컨투어_박스(my_image_path, my_image_binary, my_image_contour)
    # print(원본)
    for i in 원본.keys():
        print(i)
        for j in 원본[i]:
            print(j,원본[i][j])
        print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("")

    op_image_path = 'single2_ga.png'
    op_image = cv2.imread(op_image_path, cv2.IMREAD_COLOR)
    op_image_binary, op_image_contour = 이미지_이진화_및_컨투어_찾기(op_image)
    비교당할이미지 = 컨투어_박스(op_image_path, op_image_binary, op_image_contour)
    # print(비교당할이미지)
    for i in 비교당할이미지.keys():
        print(i)
        for j in 비교당할이미지[i]:
            print(j,비교당할이미지[i][j])
            # pass
        print("")
    # my_image_자음_contour_list, my_image_모음_contour_list = 자음_모음_컨투어_구분(my_image, my_image_contour)
