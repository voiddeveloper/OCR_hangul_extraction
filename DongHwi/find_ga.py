import cv2
import time

resize_width = 100
resize_height = 100

def 이미지_이진화_및_컨투어_찾기(image):

    # rgb -> gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 0 - 흑 , 255 백으로 나눔
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 글자의 외각만 찾기, 좌표들은 contours에 들어있음
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

    for i,con in enumerate(contour):
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

def 자음과_연관된_모음찾기(contour_list_자음, contour_list_모음):
    연관_list_contour_index = []
    new_contour_list = []
    temp_자음 = []
    temp_모음 = []

    # print('qw',len(contour_list_자음))
    # print('qw',len(contour_list_모음))

    for 자음_index, 자음_contour in enumerate(contour_list_자음):
        x_min = 9999
        x_max = 0
        y_min = 9999
        y_max = 0
        for i in 자음_contour:
            if i[0][0] < x_min:
                x_min = i[0][0]
            if i[0][0] > x_max:
                x_max = i[0][0]
            if i[0][1] < y_min:
                y_min = i[0][1]
            if i[0][1] > y_max:
                y_max = i[0][1]

        for 모음_index, 모음_contour in enumerate(contour_list_모음):
            for j in 모음_contour:

                # 자음의 x 최대 부터 자음의 ( 가로폭 / 2 ) 만큼 더 오른쪽으로 갔을때, 모음이 있다면?
                if j[0][0] > x_max and j[0][0] < x_max + int((x_max - x_min)):
                    연관_list_contour_index.append([자음_index, 모음_index])
                    break

    # print('qw',len(연관_list_contour_index))


    for index, contour_index in enumerate(연관_list_contour_index):
        for i in contour_list_자음[contour_index[0]]:
            # print(i[0])
            temp_자음.append(i[0])

        for i in contour_list_모음[contour_index[1]]:
            # print(i[0])
            temp_모음.append(i[0])
        new_contour_list.append([temp_자음+temp_모음])
        temp_자음.clear()
        temp_모음.clear()
    # print(len(new_contour_list))

    return new_contour_list

def my_image_thumbnail(my_image, my_image_contour, my_new_contour_list):
    # 만약 자음 옆에 모음이 없다면 값이 안들어있다.
    # 값이 안들어있다면 이전에 찾은 컨투어를 가지고 사각형을 만든다.
    # 자음 + 모음 조합이라면 자음과 연관된 모음찾기 메서드에서 반환받은 컨투어 사용
    if len(my_new_contour_list) == 0:
        print('자음+모음 조합 아님')

        for i,con in enumerate(my_image_contour):

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
            my_draw = cv2.rectangle(my_image, (x_min, y_min),(x_max, y_max), (0, 0, 255), 1)
            my_draw = cv2.putText(my_image, "No."+ str(i), (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))

            my_crop_image = my_image[y_min:y_max, x_min:x_max]

            my_crop_image = cv2.resize(my_crop_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            my_crop_gray_image = cv2.cvtColor(my_crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, my_result_crop_image = cv2.threshold(my_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    else:
        print('자음+모음 조합임')
        for i,con in enumerate(my_new_contour_list):

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
                my_draw = cv2.rectangle(my_image, (x_min, y_min),(x_max, y_max), (0, 0, 255), 1)
                my_draw = cv2.putText(my_image, "No."+ str(i), (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))

                my_crop_image = my_image[y_min:y_max, x_min:x_max]

                my_crop_image = cv2.resize(my_crop_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)

                # rgb -> gray
                my_crop_gray_image = cv2.cvtColor(my_crop_image, cv2.COLOR_BGR2GRAY)

                # 이미지 이진화 0 - 흑 , 255 백으로 나눔
                ret, my_result_crop_image = cv2.threshold(my_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return my_result_crop_image, my_draw

def op_image_thumbnail(op_image, op_image_contour, op_new_contour_list):
    result_image_list = []
    op_draw_list = []
    point_min_list = []
    point_max_list = []

    print(len(op_new_contour_list))
    # 만약 연관된 컨투어가 있다면
    if len(op_new_contour_list) != 0:

        for i,con in enumerate(op_new_contour_list):

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
            op_draw = cv2.rectangle(op_image, (x_min, y_min),(x_max, y_max), (0, 0, 255), 1)
            # op_draw = cv2.putText(op_image, "No."+ str(i), (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
            point_min_list.append([x_min,y_min])
            point_max_list.append([x_max,y_max])
            op_draw_list.append(op_draw)

            # print('\n 비교할 이미지')
            op_crop_image = op_image[y_min:y_max, x_min:x_max]

            op_crop_image = cv2.resize(op_crop_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            op_crop_gray_image = cv2.cvtColor(op_crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, op_result_crop_image = cv2.threshold(op_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            result_image_list.append(op_result_crop_image)

    else:
        for i,con in enumerate(op_image_contour):

            x_min = 9999
            x_max = 0
            y_min = 9999
            y_max = 0

            for index, j in enumerate(op_image_contour[i]):

                if x_min > j[0][0]:
                    x_min = j[0][0]

                if x_max < j[0][0]:
                    x_max = j[0][0]

                if y_min > j[0][1]:
                    y_min = j[0][1]

                if y_max < j[0][1]:
                    y_max = j[0][1]
            op_draw = cv2.rectangle(op_image, (x_min, y_min),(x_max, y_max), (0, 0, 255), 1)
            # op_draw = cv2.putText(op_image, "No."+ str(i), (x_min,y_min), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
            point_min_list.append([x_min,y_min])
            point_max_list.append([x_max,y_max])

            op_draw_list.append(op_draw)

            # print('\n 비교할 이미지')
            op_crop_image = op_image[y_min:y_max, x_min:x_max]

            op_crop_image = cv2.resize(op_crop_image, dsize=(resize_width, resize_height), interpolation=cv2.INTER_AREA)

            # rgb -> gray
            op_crop_gray_image = cv2.cvtColor(op_crop_image, cv2.COLOR_BGR2GRAY)

            # 이미지 이진화 0 - 흑 , 255 백으로 나눔
            ret, op_result_crop_image = cv2.threshold(op_crop_gray_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            result_image_list.append(op_result_crop_image)

    return result_image_list, op_draw_list, point_min_list, point_max_list


def image_comparison(image1, image2):

    image1_copy = image1.copy()
    image2_copy = image2.copy()

    # cv2.imshow('1',image1_copy)
    # cv2.imshow('2',image2_copy)
    # cv2.waitKey(0)
    count = 0

    sum = resize_width * resize_height

    image1_pixel_count = 0
    image2_pixel_count = 0

    for i in range(resize_width):
        for j in range(resize_height):
            if image1_copy[i][j] == 0:
                image1_pixel_count += 1
            if image2_copy[i][j] == 0:
                image2_pixel_count += 1
            # print("%3d"%image1_copy[i][j], end=" ")
        # print('\n')

    # print('\n')

    for i in range(resize_width):
        for j in range(resize_height):
            # print("%3d"%image2_copy[i][j], end=" ")
            if image1_copy[i][j] == image2_copy[i][j]:
                count += 1
        # print('\n')

    print(image1_pixel_count, image2_pixel_count)

    add = image1_pixel_count / 3

    pixel_range_min = int(image1_pixel_count - add)
    pixel_range_max = int(image1_pixel_count + add)

    print(pixel_range_min, pixel_range_max, (count * 100) / sum)

    if pixel_range_min > image2_pixel_count or image2_pixel_count > pixel_range_max:
        print('사이')
        return 0

    else:
        return (count * 100) / sum

if __name__ == '__main__':
    """ 코드 수행시간 측정"""
    start_time = time.time()

    """
    이미지 불러오기 : 폰트 맑은 고딕
    my_image : 비교할 이미지
    op_image : 비교대상 이미지
    op_image1 : 비교대상 이미지 ( 최종 사각형 그리는데 사용 )
    """
    my_image = cv2.imread('image/ga.png', cv2.IMREAD_COLOR)
    op_image = cv2.imread('image/asdasd.png', cv2.IMREAD_COLOR)
    op_image1 = cv2.imread('image/asdasd.png', cv2.IMREAD_COLOR)

    """ 
    이미지 이진화 처리 및 컨투어 찾기 
    컨투어 : 물체(글자)의 라인 좌표를 반환해줌 
    """
    my_image_binary, my_image_contour = 이미지_이진화_및_컨투어_찾기(my_image)
    op_image_binary, op_image_contour = 이미지_이진화_및_컨투어_찾기(op_image)

    """ 
   자음 컨투어 리스트 및 모음 컨투어 리스트를 구분해서 반환해줌 .. 
   자음과 모음은 구분은 해당 글자의 가로길이와 세로길이의 비율로 판단한다. 
   """
    my_image_자음_contour_list, my_image_모음_contour_list = 자음_모음_컨투어_구분(my_image, my_image_contour)
    op_image_자음_contour_list, op_image_모음_contour_list = 자음_모음_컨투어_구분(op_image, op_image_contour)

    """
    자음으로 부터 모음을 찾기 
    """
    my_new_contour_list = 자음과_연관된_모음찾기(my_image_자음_contour_list, my_image_모음_contour_list)
    op_new_contour_list = 자음과_연관된_모음찾기(op_image_자음_contour_list, op_image_모음_contour_list)

    """ 비교할 이미지의 썸네일을 만든다. 1장"""
    my_thumbnail, my_draw= my_image_thumbnail(my_image, my_image_contour, my_new_contour_list)

    """ 비교당할? 이미지의 썸네일을 만든다. 여러장"""
    op_thumbnail_list, op_draw_list, point_min_list, point_max_list = op_image_thumbnail(op_image,op_image_contour,op_new_contour_list)

    """ 내 이미지와 비교대상 이미지를 픽셀 단위로 비교한다. """
    for i in range(len(op_thumbnail_list)):
        percent = image_comparison(my_thumbnail, op_thumbnail_list[i])

        percent = round(percent,2)

        print('No.',str(i),'과 일치율 : ', percent)
        print('\n')
        if percent >= 90:
            draw = cv2.rectangle(op_image1, (point_min_list[i][0], point_min_list[i][1]),(point_max_list[i][0], point_max_list[i][1]), (0, 0, 255), 1)
            draw = cv2.putText(op_image1, "No."+ str(i) + " : " + str(percent) + "%", (point_min_list[i][0], point_min_list[i][1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
        else:
            draw = cv2.putText(op_image1, "not found", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0))

    cv2.imshow('my_draw', my_draw)
    cv2.imshow('op_draw', draw)
    cv2.waitKey(0)

    print("걸린 시간 : ", time.time() - start_time)
