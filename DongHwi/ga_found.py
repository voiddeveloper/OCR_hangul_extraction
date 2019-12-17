import cv2
import numpy as np
from PIL import Image
import math


resize_width = 50
resize_height = 50

""" 이거는 안쓰고 있음 , 글자의 뼈대만 추출해내는 코드 """
def image_skeleton(image):

    kernel = np.ones((3, 3), np.uint8)

    height, width = image.shape
    print("가로 : ", width, " 세로 : ",height)
    size=np.size(image)
    skel=np.zeros(image.shape,np.uint8)

    blur= cv2.GaussianBlur(image,(5,5),0)
    ret,thrs=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv=cv2.bitwise_not(thrs)

    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(inv,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(inv,temp)
        skel = cv2.bitwise_or(skel,temp)
        inv = eroded.copy()

        zeros = size - cv2.countNonZero(inv)

        if zeros==size:
            done = True

    skel = cv2.dilate(skel,kernel,iterations=2)
    skel = cv2.erode(skel,kernel,iterations=1)

    return skel

def my_image_find_contour(my_image):
    # print('내 이미지 \n')

    image = my_image.copy()

    # rgb -> gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 0 - 흑 , 255 백으로 나눔
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    no, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min = 9999
    x_max = 0
    y_min = 9999
    y_max = 0

    for i,con in enumerate(contours):

        for index, j in enumerate(contours[i]):

            # 좌표의 x최소, y최소, x최대, y최대 찾기
            if x_min > j[0][0]:
                x_min = j[0][0]

            if x_max < j[0][0]:
                x_max = j[0][0]

            if y_min > j[0][1]:
                y_min = j[0][1]

            if y_max < j[0][1]:
                y_max = j[0][1]

        # 글자를 찾았으면 찾은 영역만 잘라낸다.
        crop_image = image[y_min:y_max, x_min:x_max]

        # 잘라낸 이미지를 원본이미지와 동일하게 리사이즈한다.
        crop_image = cv2.resize(crop_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)

        # rgb -> gray
        crop_gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

        # 이미지 이진화 0 - 흑 , 255 백으로 나눔
        ret, result_image = cv2.threshold(crop_gray_image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    my_draw = cv2.rectangle(image, (x_min, y_min),(x_max, y_max), (255, 0, 255), 1)
    # cv2.imshow('sdf',draw)
    # cv2.waitKey(0)

    return result_image, my_draw


def op_image_find_contour(my_image):

    image = my_image.copy()

    # rgb -> gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 0 - 흑 , 255 백으로 나눔
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
    no, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min = 9999
    x_max = 0
    y_min = 9999
    y_max = 0

    result_image_list = []
    put_text_count = 1

    for i,con in enumerate(contours):

        new_contours = []

        for index, j in enumerate(contours[i]):

            if x_min > j[0][0]:
                x_min = j[0][0]

            if x_max < j[0][0]:
                x_max = j[0][0]

            if y_min > j[0][1]:
                y_min = j[0][1]

            if y_max < j[0][1]:
                y_max = j[0][1]

        # 컨투어 0,1 번째, 2,3번째 등 2개씩(바로 옆) 묶는다.
        new_contours.append([j[0][0],j[0][1]])
        if i != 0 and i % 2 == 1:
            op_draw = cv2.rectangle(image, (x_min, y_min),(x_max, y_max), (255, 0, 255), 1)
            op_draw = cv2.putText(image, "No."+ str(put_text_count), (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
            put_text_count += 1

            # print('\n 비교할 이미지')
            crop_image = image[y_min:y_max, x_min:x_max]

            crop_image = cv2.resize(crop_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            crop_gray_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, result_image = cv2.threshold(crop_gray_image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            result_image_list.append(result_image)

            x_min = 9999
            x_max = 0
            y_min = 9999
            y_max = 0
            new_contours.clear()

    return result_image_list, op_draw

def image_comparison(image1, image2):

    count = 0
    for i in range(resize_width):
        for j in range(resize_height):
            if image1[i][j] == image2[i][j]:
                count += 1

    return ((count * 100)/(resize_width * resize_height))

if __name__ == '__main__':

    my_image = cv2.imread('image/ga.png', cv2.IMREAD_COLOR)
    op_image = cv2.imread('image/ga_test.png', cv2.IMREAD_COLOR)

    my_result_image, my_draw = my_image_find_contour(my_image)
    result_image_list, op_draw = op_image_find_contour(op_image)

    for i in range(len(result_image_list)):
        percent = image_comparison(my_result_image, result_image_list[i])

        print('No.',str(i+1),'과 일치율 : ', percent)

    cv2.imshow('나의 이미지',my_draw)
    cv2.imshow('비교 이미지',op_draw)
    cv2.waitKey(0)

    # 내가 가지고 있는 이미지(글자 ㄱ) 의 뼈대만 추려낸다.
    # skeleton_image = image_skeleton(img)
