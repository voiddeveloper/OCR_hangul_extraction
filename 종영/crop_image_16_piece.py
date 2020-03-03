import cv2
import time

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

# def 픽셀수_세기and사각형n등분(이진화_이미지, x_max, y_max, n):
#
#     print(x_max, y_max)
#
#     x_cnt = 0
#     y_cnt = 0
#
#     cnt = 0
#
#     for i in range(x_cnt * int(x_max/4), x_max, int(x_max/4)):
#         for j in range(y_cnt * int(y_max/4), y_max, int(y_max/4)):
#
#             print( (y_cnt * int(x_max/4)) , i), ((y_cnt * int(x_max/4)) )
#
#             y_cnt += 1
#             pass
#
#         x_cnt += 1
#         y_cnt = 0
#
#
#     return


def 픽셀수_세기and사각형n등분(크롭_이미지, x_max, y_max, n):
    h,w=크롭_이미지.shape

    white_image = cv2.imread('image/white.png')
    start=0

    """ 크롭한 이미지를 16조각으로 나눔"""

    for i in range(0,n):
        for u in range(0,n):
            count=0
            for j in range(int(i*(h/4)),int(i*(h/4)+h/4)):
                for k in range(int(u * (w / 4)), int(u * (w / 4) + (w / 4))):
                    start+=1
                    if 크롭_이미지[j][k] == 255:
                        count+=1
            # print(count, " ")
    # print(start)

def 컨투어_박스(이미지경로, 이진화_image, image_contour, n):

    for i, con in enumerate(image_contour):
        이미지 = cv2.imread(이미지경로, cv2.IMREAD_COLOR)

        x_min = 9999
        x_max = 0
        y_min = 9999
        y_max = 0
        for index, j in enumerate(image_contour[i]):

            if x_min > j[0][0]:
                x_min = j[0][0]

            if x_max < j[0][0]:
                x_max = j[0][0]

            if y_min > j[0][1]:
                y_min = j[0][1]

            if y_max < j[0][1]:
                y_max = j[0][1]



        crop_image = 이미지[y_min:y_max, x_min:x_max]

        height, width, channel = crop_image.shape

        if x_max - x_min > 4 and y_max - y_min > 4:
            crop_image = cv2.resize(crop_image, dsize=(width * n, height * n), interpolation=cv2.INTER_AREA)
            
            height, width, channel = crop_image.shape

            # rgb -> gray
            crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, result_crop_image = cv2.threshold(crop_image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            픽셀수_세기and사각형n등분(result_crop_image, (x_max - x_min) * n, (y_max - y_min) * n, n)

if __name__ == '__main__':

    start_time = time.time()

    """
    이미지 불러오기 : 폰트 맑은 고딕
    my_image : 비교할 이미지
    """
    my_image_path = 'image/ga1.png'
    my_image = cv2.imread(my_image_path, cv2.IMREAD_COLOR)
    my_image_binary, my_image_contour = 이미지_이진화_및_컨투어_찾기(my_image)
    원본 = 컨투어_박스(my_image_path, my_image_binary, my_image_contour, 4)

    h,w,c = my_image.shape

    op_image_path = 'image/ga_test.png'
    op_image = cv2.imread(op_image_path, cv2.IMREAD_COLOR)
    op_image_binary, op_image_contour = 이미지_이진화_및_컨투어_찾기(op_image)
    비교당할이미지 = 컨투어_박스(op_image_path, op_image_binary, op_image_contour, 4)

    # #픽셀 많은 부분이 같은것 골라내기
    # result={}
    # for j in 비교당할이미지.keys():
    #     count=0
    #     for i in 원본.keys():
    #         if 원본[i]['가장많은부분'] in 비교당할이미지[j]['가장많은부분']:
    #             count+=1
    #             break
    #     if count !=0:
    #         result[len(result)]=비교당할이미지[j]
    #
    # #픽셀양으로 거르기 (픽셀양의비율)
    # aa={}
    # for j in result.keys():
    #     count=0
    #     for i in 원본.keys():
    #         if 원본[i]['전체픽셀']-5<=result[j]['전체픽셀'] and 원본[i]['전체픽셀']+5>=result[j]['전체픽셀'] and 원본[i]['가장많은부분'] in result[j]['가장많은부분']:
    #             aa[len(aa)]=result[j]
    #             break
    # #거리 비율
    #
    #
    # for i in aa.keys():
    #     op_image = cv2.rectangle(op_image, (aa[i]['x_min'], aa[i]['y_min']),(aa[i]['x_max'], aa[i]['y_max']), (0, 0, 255), 2)
    #     print(aa[i])
    #
    #     cv2.imshow("asd",op_image)
    #     cv2.waitKey(0)

    print(time.time() - start_time)
