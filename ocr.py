from PIL import Image
import cv2
import time
import numpy as np
from operator import itemgetter

"""

마지막 작업일자 : 2019년 11월 13일 - 15:40분

이 코드는 이미지에서 텍스트 영역만을 추출해내기 위해 전 처리 작업을 하는 코드이다.
목적은 이미지에서 글자의 영역 주변에 네모박스를 그린 후에, 영역이 쳐진 부분을 잘라내어 새로운 이미지를 만드는 것.

과정은 아래와 같다.

1. 전 처리할 이미지 열기 - 사용할 이미지는 basic_image에 저장

2. 이미지 사이즈 조정 - 고정된 이미지(512,512) 가 들어왔다는 가정하에 실험할거임

3. RGB 이미지 -> GrayImage로 변환

4. 물체의 외각선을 정교하게 추출하는 과정

5. 이진화 과정

6. 잡음제거 과정

7. 원본 이미지에서 글자를 찾아 영역 박스 그리기 - 3가지 과정
    7.1 글자 좌표 찾기
    7.2 글자 좌표 빨간색으로 찍기
    7.3. 글자 좌표를 가지고 네모영역(박스) 그리기

8. 영역이 그려진 부분을 고정된 크기의 이미지로 만들어서 저장



!!!!! 필독 !!!!!

코드 사용 시 수정 해야할 부분

* 1번 과정 - 전 처리할 이미지 열기 - 분석할 이미지의 경로를 수정해야함.
    변수명 - basic_image

* 8번 과정 - 분석 후 저장될 이미지의 폳더를 수정해야함.
    cv2.imwrite( 여기 수정! 폴더가 들어와야함 , cropped )
    
* 최종 분석 전 이미지는 현재 경로에 result.jpg로 저장된다.

"""

# png 파일을 jpg로 만드는 메서드임 , 전처리 과정에서 png 파일 인식못해서 우선 이렇게 만듦
def png_to_jpg(png_path, target_img):
    fill_color = None
    img = target_img
    jpg_path = png_path[:-4] + '.jpg'

    if img.mode in ('RGBA', 'LA'):
        background = Image.new(img.mode[:-1], img.size, fill_color)
        background.paste(img, img.split()[-1])
        img = background

    return img.save(jpg_path, 'JPEG')

#이미지 번호
def img_crop_after_img_save(x1,y1,x2,y2,img,index):
    cropped = img[y1:y2, x1:x2]
    cropped = cv2.resize(cropped, (150, 150), interpolation=cv2.INTER_AREA)
    cv2.imwrite('image/crop_image_'+str(index)+'.jpg', cropped)

if __name__ == '__main__':

    global crop_image_index

    # 시작 시간 저장
    start = time.time()

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """--------------------------------------------------------------1. 이미지 열기---------------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    # 이미지 불러오기
    # open()메서드 안에는 이미지의 경로가 들어간다.
    # 파일형식 jpg, png 둘중에 하나 불러옴
    try:
        basic_image = Image.open("C:\\Users\\shindonghwi\\Desktop\\텍스트이미지\\test1.jpg")
        print('jpg이미지 선택')
    except:
        print('png이미지 선택 -> jpg로 변환함')
        # png형태로는 이미지 전처리가 안된다. 이유는 잘 모르겠음. 그래서 jpg로 변환을 해준다.
        basic_image = Image.open("C:\\Users\\shindonghwi\\Desktop\\텍스트이미지\\test1.png")
        basic_image = basic_image.convert('RGB')

    # 원본이미지 저장 / 마지막에 원본이미지 위위에 글 영역 네모박스를 치기 위해 필요함.
    # original_image - 아래 7번과정에서 사용 / 글 영역 네모박스 치는 메서드를 사용할때 numpy.ndarray 자료형을 필요로 해서 일단 이렇게 만들어둠.
    original_image = np.asarray(basic_image)

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """---------------------------------------------------------2. 이미지 사이즈 조정--------------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    # 이미지의 가로, 세로 사이즈 확인
    basic_image_width, basic_image_height = basic_image.size
    original_image_height = original_image.shape[0]
    original_image_width = original_image.shape[1]

    print('image Width           : ', basic_image_width)
    print('image Height          : ', basic_image_height)
    print('original Width        : ', original_image_width)
    print('original Height       : ', original_image_height)

    min_size = 800
    mid_size = 1000
    mid2_size = 1200
    max_size = 1500
    # image 재 조정 : 가로 세로를 512로 맞출거임
    # 고정된 크기의 이미지가 들어온다고 가정하고 실험해볼거

    # 가로 또는 세로의 크기가 512보다 작을때, cv2.INTER_CUBIC는 작은 이미지를 크게 만들때 사용한다.
    if original_image_width + original_image_height <= 900 and (
            original_image_width <= 600 or original_image_height <= 600):
        print('가로 or 세로 600이하')
        original_image = cv2.resize(original_image, (min_size, min_size), interpolation=cv2.INTER_CUBIC)
        original_image_height = original_image.shape[0]
        original_image_width = original_image.shape[1]
        basic_image = basic_image.resize((min_size, min_size))
        basic_image_width, basic_image_height = basic_image.size

    # 가로 또는 세로의 크기가 900보다 작을때, cv2.INTER_AREA 작은 이미지를 크게 만들때 사용한다.
    elif original_image_width + original_image_height <= 1400 and (
            original_image_width <= 1000 or original_image_height <= 1000):
        print('가로 or 세로 1000이하')
        original_image = cv2.resize(original_image, (mid_size, mid_size), interpolation=cv2.INTER_CUBIC)
        original_image_height = original_image.shape[0]
        original_image_width = original_image.shape[1]
        basic_image = basic_image.resize((mid_size, mid_size))
        basic_image_width, basic_image_height = basic_image.size

    # 가로 또는 세로의 크기가 900이상 작을때, cv2.INTER_AREA 작은 이미지를 크게 만들때 사용한다.
    elif original_image_width + original_image_height < 2000 and (
            original_image_width <= 1300 or original_image_height <= 1300):
        print('가로 or 세로 1500이하')
        original_image = cv2.resize(original_image, (mid2_size, mid2_size), interpolation=cv2.INTER_CUBIC)
        original_image_height = original_image.shape[0]
        original_image_width = original_image.shape[1]
        basic_image = basic_image.resize((mid2_size, mid2_size))
        basic_image_width, basic_image_height = basic_image.size

    # 가로 또는 세로의 크기가 900이상 작을때, cv2.INTER_AREA 작은 이미지를 크게 만들때 사용한다.
    elif original_image_width + original_image_height <= 2300 and (
            original_image_width <= 1100 and original_image_height <= 1650):
        print('가로 or 세로 1500이하')
        original_image = cv2.resize(original_image, (max_size, max_size), interpolation=cv2.INTER_CUBIC)
        original_image_height = original_image.shape[0]
        original_image_width = original_image.shape[1]
        basic_image = basic_image.resize((max_size, max_size))
        basic_image_width, basic_image_height = basic_image.size

    # 가로 또는 세로의 크기가 1500이상 일때, cv2.INTER_AREA 큰 이미지를 작게 만들때 사용한다.
    else:
        print('가로 or 세로 1500이상')
        original_image = cv2.resize(original_image, (3000, 3000), interpolation=cv2.INTER_AREA)
        original_image_height = original_image.shape[0]
        original_image_width = original_image.shape[1]
        basic_image = basic_image.resize((3000, 3000))
        basic_image_width, basic_image_height = basic_image.size

    print('image Width           : ', basic_image_width)
    print('image Height          : ', basic_image_height)
    print('original Width        : ', original_image_width)
    print('original Height       : ', original_image_height)

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """-------------------------------------------------------3.RGB to GRAY 이미지 변환-----------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    # RGB 이미지 -> GrayScale 이미지로 변환
    # convert()메서드는 이미지를 다른 모드로 변환 시킬때 사용한다.
    # ex) RGB, CMYK, L(512단계 흑백 이미지), 1(단색 이미지)
    """ 
        문제 : gray_image를 numpy_array로 바꾸는 예제도 있는데 왜 그렇게 하는지 아직 잘 모르겠다. ----> 2019-11-11 .. 19:22시
         
        위 문제 해결 : 4번 과정에서 외각선을 추출하기위해서는 이미지의 pixel값을 조정 할 수 있어야한다. 
        그래서 numpy_array로 만드는것임 ----> 2019-11-11 .. 20:45시
    """
    gray_image = basic_image.convert('L')
    gray_np_image = np.array(gray_image, 'uint8')

    # 필터의 사이즈 조정 / 이미지 전처리 과정에서 컨벌루션 하기 위해 필요함
    # 2 x 2 행렬을 만들고 , 비교할 이미지의 왼쪽 상단부터 2x2 크기만큼 비교를 하며, 이미지에 가해지는 변형을 결정함.
    # 아래 kernel은 [ dilation 과 erosion 과정 - 이미지 변형 처리 과정 ] 에서 사용될 커널임.
    kernel = np.ones((2, 2), np.uint8)

    # 팽창과정 - 가령 A라는 글자가 있으면 이 글자를 조금 더 두껍게 만드는 과정
    dilation = cv2.dilate(gray_np_image, kernel, iterations=1)

    # 침식과정 - 가령 A라는 글자가 있으면 이 글자를 조금 더 얇게 만드는 과정
    erosion = cv2.erode(gray_np_image, kernel, iterations=1)

    # 팽창, 침식 이미지 보기
    # cv2.imshow('dilation',dilation)
    # cv2.imshow('erosion',erosion)

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """-------------------------------------------------------4. 외곽선 이미지 추출----------------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    # 외각선을 더 정교하게 추출하기 위해서 두개의 이미지 차이를 구한다.
    # 두꺼운 이미지에서 얇은 이미지를 빼면 외각선이 추출된다.
    buffer1 = np.asarray(dilation)
    buffer2 = np.asarray(erosion)
    morph_gradient_image = buffer1 - buffer2
    # cv2.imshow('morph_gradient_image',morph_gradient_image)

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """-------------------------------------------------------5. 이진화 과정-------------------------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    """
        이진화 작업 1. 기본 임계처리 방법 
        이진화란 이미지를 흑 / 백으로 분류하여 처리하는 것임
        중요한 요소는 임계값을 설정하는 것이다.
        내가 설정한 임계값 보다 크면, 백색
        내가 설정한 임계값 보다 작으면, 흑색으로 분류된다.
        
        매개 변수는 총 4개임. src, thresh, maxval, type
        src - grayscale 이미지
        thresh - 임계값
        maxval - 임계값을 넘었을 때 적용할 value
        type - thresholding type
        
        thresholding type 종류는 5가지
            cv2.THRESH_BINARY - 픽셀값이 임계값 보다 크면 maxval, 작으면 0으로 할당
            cv2.THRESH_BINARY_INV - 픽셀값이 임계값 보다 크면 0, 작으면 maxval으로 할당
            cv2.THRESH_TRUNC - 픽셀값이 임계값 보다 크면 maxval, 작으면 0으로 할당
            cv2.THRESH_TOZERO - 픽셀값이 임계값 보다 크면 maxval, 작으면 0으로 할당
            cv2.THRESH_TOZERO_INV - 픽셀값이 임계값 보다 크면 maxval, 작으면 0으로 할당
    """
    # 이진화 방법 1. -
    # _ ,Threshold = cv2.threshold(morph_gradient_image,127,255,cv2.THRESH_BINARY)

    """
        이진화 작업 2. 적응 임계처리 방법 
        위에서 처리 하였던 방법음 이미지 전체에 걸쳐 하나의 임계값으로 흑,백으로 나눈것이다. 따라서 내가 원하는 값을 얻어내긴 힘들다고함. 아직 잘 모르겠음
        
        AdaptiveThreshold 방법은 이미지의 서로 다른 작은 영역에 적용되는 문턱값을 계산하고 이미지에 적용함으로써 더 좋은 결과를 나타낸다고함.
        
        매개변수는 6개임 - img, maxValue, adaptiveMethod, thresholdType, blocksize, C
        src – grayscale image
        maxValue – 임계값
        adaptiveMethod – thresholding value를 결정하는 계산 방법
        thresholdType – 사용할 AdaptiveThreshold 알고리즘
            - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: X, Y를 중심으로 block Size * block Size 안에 있는 픽셀 값의 평균에서 C를 뺸 값을 문턱값으로 함 
            - cv2.ADAPTIVE_THRESH_MEAN_C: X, Y를 중심으로 block Size * block Size 안에 있는 Gaussian 윈도우 기반 가중치들의 합에서 C를 뺀 값을 문턱값으로 한다. 
        blockSize – block * block = thresholding을 적용할 영역 사이즈 / 단, 블록사이즈는 홀수여야한다.
        C – 보정 상수로서 adaptive에 계산된 값에서 양수면 빼주고 음수면 더해준다. 
    """
    # 이진화 방법 2. - Adaptive Threshold 이미지는 영역을 분할하고 임계값을 자동으로 조정해 얻은 흑백 이미지
    if original_image_width == 800:
        AdapThreshold_GAUSSIAN_C = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV, 15, 12)
        AdapThreshold_MEAN = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 7, 20)
    elif original_image_width == 1000:
        AdapThreshold_GAUSSIAN_C = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV, 25, 12)
        AdapThreshold_MEAN = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 7, 20)
    elif original_image_width == 1200:
        AdapThreshold_GAUSSIAN_C = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV, 25, 12)
        AdapThreshold_MEAN = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 7, 20)
    elif original_image_width == 1500:
        AdapThreshold_GAUSSIAN_C = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV, 29, 12)
        AdapThreshold_MEAN = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 7, 20)
    elif original_image_width == 3000:
        AdapThreshold_GAUSSIAN_C = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY_INV, 35, 12)
        AdapThreshold_MEAN = cv2.adaptiveThreshold(morph_gradient_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, 7, 20)



    # cv2.imshow('AdapThreshold_GAUSSIAN_C',AdapThreshold_GAUSSIAN_C)
    # cv2.imshow('AdapThreshold_MEAN',AdapThreshold_MEAN)

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """-------------------------------------------------------6. 잡음제거 과정-----------------------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    # 잡음을 제거 하는 과정임
    # Closing과정이라고도 하는데, 테두리에 팽창을 적용하고 이어서 침식을 적용하여 끊어진 점을 제거하는 과정임
    # 이걸 하게되면 테두리가 선명해짐 ㄹㅇ
    kernel = np.ones((5, 5), np.uint8)
    closing_image = cv2.morphologyEx(AdapThreshold_GAUSSIAN_C, cv2.MORPH_CLOSE, kernel, iterations=1)
    # cv2.imshow('closing_image',closing_image)

    # 주변 배경에 있는 잡음을 제거하는 과정
    # Closing과 반대로 침식 -> 팽창 하는 과정이다. 침식 작용에서 배경에있는 잡음이 제거되고, 팽창과정에서 글자를 조금 더 선명하게 해준다.
    opening = cv2.morphologyEx(closing_image, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow('opening',opening)

    """------------------------------------------------------------------------------------------------------------------------------------"""
    """---------------------------------------------7. 원본 이미지에서 글자를 찾아 네모 영역(박스) 그리기-------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    # 컨투어 - 동일한 색, 동일한 픽셀값을 가지고 있는 영역의 정보
    #   컨투어 추가 설명 - 선이라고 생각하면됨. 선은 점들의 연속으로 이루어져있다. 따라서 컨투어 객체 안에는 점 좌표들이 들어있음.
    # 물체의 윤곽선, 외형을 파악하는데 사용함
    # 매개변수 image, mode, method
    """ # image - 흑백이미지 또는 이진화된 이미지를 사용한다. """
    """ # mode - cv2.RETR_EXTERNAL - 컨투어 라인 가장 바깥쪽 라인만 찾음 """
    # mode - cv2.RETR_LIST - 모든 컨투어 라인을 찾지만, 상하구조(hierachy)관계를 구성하지 않음
    # mode - cv2.RETR_TREE - 모든 컨투어 라인을 찾고, 모든 상하구조를 구성함
    # mode - cv2.RETR_CCOMP - 모든 컨투어 라인을 찾고, 상하구조는 2 단계로 구성함
    # method - 컨투어를 찾을때 사용하는 근사화 방법
    # method - cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환
    """ # method - cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환 """
    # method - cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용하여 컨투어 포인트를 줄임
    # method - cv2.CHAIN_APPROX_TC89_KCOS: Teh_Chin 연결 근사 알고리즘 KCOS 버전을 적용하여 컨투어 포인트를 줄임

    # findContours는 이미지에 컨투어 포인트를 할당 해주는 것임
    # contours에는 점 좌표가 배열 형태로 들어있음
    image, contours, hierachy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # drawContours
    # draw_image = cv2.drawContours(original_image, contours, -1, (0,0,255), 2)

    # 사용하지 않을 컨투어 들을 담을 배열
    contours_remove_list = []

    # 글자를 찾았을때, 해당 좌표들을 여기에 담을거임
    point_list = []
    # point_index = 0

    print('전체 contours 개수 : ', len(contours))


    """------------------------------------------------------------------------------------------------------------------------------------"""
    """-----------------------------------------------------------7.1 글자 좌표 찾기--------------------------------------------------------"""
    """------------------------------------------------------------------------------------------------------------------------------------"""

    for i in range(len(contours)):
        # i번째 배열에서 첫 번째 좌표의 왼쪽 상단, 오른쪽 하단 값을 찾는다.
        X_MIN = [contours[i]][0][0][0][0]
        Y_MIN = [contours[i]][0][0][0][1]
        X_MAX = [contours[i]][0][1][0][0]
        Y_MAX = [contours[i]][0][1][0][1]

        # i번째 배열의 리스트 만큼 반복한다.
        # i번째 배열의 좌표에서 왼쪽 상단, 오른쪽 하단 좌표값을 찾을 예정
        """ 원하는 좌표값을 찾게 되면 사각형 영역을 그리는 알고리즘을 만들어야함. - 2019.11.12 - 미해결 """
        for j in range(len(contours[i])):

            # 첫번째 좌표를 제외하고
            if j <= 1:
                continue

            # i번째 배열의 x좌표,y좌표를 가져온다.
            new_X = [contours[i]][0][j][0][0]
            new_Y = [contours[i]][0][j][0][1]

            # 왼쪽 상단 좌표와, 오른쪽 상단 좌표를 찾는 식
            if new_X < X_MIN:
                X_MIN = new_X
            elif new_X >= X_MAX:
                X_MAX = new_X
            if new_Y < Y_MIN:
                Y_MIN = new_Y
            elif new_Y >= Y_MAX:
                Y_MAX = new_Y

        # x좌표 차이 또는 y좌표 차이가 거의 나지 않는 컨투어들은 제외하기 위해서 index를 기억해둔다.
        # X_MAX - X_MIN 은 글자의 가로폭을 의미한다 라고 생각하고 코드를 만들었다.
        # 따라서 word_width 는 글자의 가로폭이다.
        # len(contours[i]) --> 점의 갯수 / 너무 많은 점으로 이루어지거나 너무 적은 점으로 이루어진 것은 글자라고 판단안한다.
        """ 이 코드도 이미지 크기 별로 글자폭이 다르니 여러 Case를 두고 수정을 해야할것같다. 2019.11.13 - 11:26 """
        word_width = 10
        word_height = 10
        if X_MAX - X_MIN <= word_width or Y_MAX - Y_MIN <= word_height or len(contours[i]) <= 15 or len(contours[i]) >= original_image_width - int(original_image_width / 8):
            contours_remove_list.append(i)
        else:
            point_list.append([X_MIN,Y_MIN,X_MAX,Y_MAX])
            # point_index += 1
            # 원본 이미지 글자 위에 네모 영역을 치는 코드
            # draw_image = cv2.rectangle(original_image, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 0, 255), 1)
            # img_crop(X_MIN,Y_MIN,X_MAX,Y_MAX,original_image,point_index)
            # 이미지 위에 번호 찍는 코드
            # cv2.putText(original_image, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

            """------------------------------------------------------------------------------------------------------------------------------------"""
            """---------------------------------------------------7.2 글자 좌표 빨간색으로 찍기------------------------------------------------------"""
            """------------------------------------------------------------------------------------------------------------------------------------"""
            # 원본 이미지 글자 위에 왼쪽 상단의 좌표와, 오른쪽 하단의 좌표를 빨간색으로 찍는 코드
            draw_image = cv2.circle(original_image,(X_MIN,Y_MIN), 3, (0,0,255), -1)
            draw_image = cv2.circle(original_image,(X_MAX,Y_MAX), 3, (0,0,255), -1)

    print('제외할 contours 개수 : ', len(contours_remove_list))



    # 기억해둔 컨투어를 제외시킨다.
    for i in reversed(contours_remove_list):
        contours.pop(i)

    # 가장 작은 y축 값을 기준으로 재 정렬한다.
    """ 원래는 가장 작은 x,y 축을 기준으로 정렬을 하려했는데 python에서 어떻게 하는지 잘 모르겠음 - 2019 . 11.13 - 15:10시"""
    point_list = sorted(point_list,key=lambda x: x[1], reverse=False)


    # 재 정렬한 좌표를 가지고 글자위에 네모영역을 치는 코드
    # img_crop은 글자의 왼쪽위 좌표와, 오른쪽아래 좌표를 가지고 고정된 크기만큼으로 이미지를 잘라서 저장하는 메서드임
    # crop_image_index는 저장될 이미지의 번호
    crop_image_index = 0

    for x1,y1,x2,y2 in point_list:

        """------------------------------------------------------------------------------------------------------------------------------------"""
        """-------------------------------------------7.3. 글자 좌표를 가지고 네모영역(박스) 그리기-----------------------------------------------"""
        """------------------------------------------------------------------------------------------------------------------------------------"""
        draw_image = cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        """------------------------------------------------------------------------------------------------------------------------------------"""
        """-----------------------------------------8. 영역이 그려진 부분을 고정된 크기의 이미지로 만들어서 저장------------------------------------"""
        """------------------------------------------------------------------------------------------------------------------------------------"""
        # 이미지가 저장될 경로는 메서드 안에서 수정한다.
        img_crop_after_img_save(x1,y1,x2,y2,original_image,crop_image_index)
        crop_image_index += 1

    print('사용할 contours 개수 : ', len(contours))

    # 원본 이미지의 텍스트를 따라 파란색으로 그려주는 코드
    draw_image = cv2.drawContours(original_image, contours, -1, (255,0,0), 1)

    # 그려진 이미지 저장
    cv2.imwrite('result.jpg', draw_image)

    # 현재시각 - 시작시간 = 실행 시간
    print("time :", time.time() - start)
    cv2.waitKey(0)
