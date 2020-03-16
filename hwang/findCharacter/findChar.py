import cv2 as cv
import numpy as np

#################################################################################
# 목표
# learnChar.py 에서 찾아낸 특징점들을 이용하여 학습한 글씨를 찾는다.
# 이미지를 부분적으로 잘라내서 검색해서, 유사한 특징점이 있는지 검색한다.

# hwang/imgSet/test_image/14.jpg
# 현재 테스트 값
# 0번쨰 contour의 
#  area = 156.0
#  perimeter = 108.56
#  aspect_ratio = 0.7
#  extent = 0.24

# 1번째 contour의
#  area = 144.5
#  perimeter = 97.89
#  aspect_ratio = 0.26
#  extent = 0.31

# 전체 이미지의
#  width = 45
#  height = 50

# 이다.
#################################################################################

## main 시작 ##
characterPoint = ([156.0, 108.56, 0.7, 0.24], [144.5, 97.89, 0.26, 0.31])

for i in range(14, 15):
    # 이미지 경로
    image_path = 'hwang/imgSet/test_image/'+ str(i) + '.jpg'
    
    # 테스트값 width, height의 1.1배를 적용
    crop_width = int(45 * 1.1)
    crop_height = int(50 * 1.1)

    savePoint = []

    # bgr 이미지 불러오기
    bgr_image = cv.imread(image_path)
    height, width = bgr_image.shape[:2]

    copy_image = bgr_image.copy()
    black_image = np.zeros_like(bgr_image)
    black_image = cv.cvtColor(black_image, cv.COLOR_BGR2GRAY)
    ret, black_image = cv.threshold(black_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    for ch in range(0, height - crop_height):
        for cw in range(0, width - crop_width):
        # for cw in range(10, 20):
            # print("cw for문 시작 cw = ", cw, " ch = ", ch)
            crop_bgr_image = bgr_image[ch : ch + crop_height, cw : cw + crop_width]
            
            # crop 이미지 특징점 추출
            crop_gray_image = cv.cvtColor(crop_bgr_image, cv.COLOR_BGR2GRAY)
            ret, crop_binary_image = cv.threshold(crop_gray_image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            crop_contours, crop_hierarchy = cv.findContours(crop_binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            flag = False

            # 조건1. len(contours) == 2
            if len(crop_contours) == 2:
                # 조건2. 각 contour별 특징점 비교
                for j in range(len(crop_contours)):
                    # print("crop_contour for문 시작 j = ", j)
                    cnt = crop_contours[j]
                
                    # 면적
                    area = cv.contourArea(cnt)

                    if area > characterPoint[j][0] * 0.9 and area < characterPoint[j][0] * 1.1:
                        # 둘레
                        perimeter = cv.arcLength(cnt, True)

                        if perimeter > characterPoint[j][1] * 0.9 and perimeter < characterPoint[j][1] * 1.1:
                            # 종횡비
                            x, y, w, h = cv.boundingRect(cnt)
                            aspect_ratio = float(w) / h

                            if aspect_ratio > characterPoint[j][2] * 0.9 and aspect_ratio < characterPoint[j][2] * 1.1:    
                                # 크기
                                rect_area = w * h
                                extent = float(area) / rect_area

                                # 크기가 learn 포인트의 +- 10% 오차내에 있는가
                                if extent > characterPoint[j][3] * 0.9 and extent < characterPoint[j][3] * 1.1:
                                    # print("모든 조건 통과")
                                    
                                    if j != len(crop_contours) - 1:
                                        flag = True
                                        # print("아직 다른 contour 비교해야 함")
                                    
                                    if j == len(crop_contours) - 1 and flag == True:
                                        # print("모든 조건 만족, 테두리 칠함")
                                        # print("cw = ", cw, " ch = ", ch, " crop_width = ", crop_width, " crop_height = ", crop_height)
                                        # cv.rectangle(copy_image, (cw, ch), (cw + crop_width, ch + crop_height), (0, 0, 255), 1)
                                        savePoint.append([cw, ch, cw + crop_width, ch + crop_height])

                                else:
                                    # print("extent 틀림")
                                    break
                            else:
                                # print("aspect_ratio 틀림")
                                break
                        else:
                            # print("perimeter 틀림")
                            break
                    else:
                        # print("area 틀림")
                        break

            else:
                # print("조건 1이 틀림")
                continue

    print("모든 조건 종료")
    for scnt in range(len(savePoint)):
        rect = cv.rectangle(black_image, (savePoint[scnt][0], savePoint[scnt][1]), (savePoint[scnt][2], savePoint[scnt][3]), 255, 1)

    rect_contours, rect_hierarchy = cv.findContours(black_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for rect_i, rect_con in enumerate(rect_contours):
        if rect_hierarchy[0][rect_i][3] != -1:
            # cv.drawContours(copy_image, rect_contours, rect_i, (0, 0, 255), 1)
            x, y, w, h = cv.boundingRect(rect_con)
            cv.rectangle(copy_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            print(x, y, w, h)

            crop_image = bgr_image[y : y + h, x : x + w]
            cv.imshow("crop" + str(rect_i), crop_image)
            cv.imwrite('hwang/resultFolder/' + str(rect_i) + '_crop.jpg', crop_image)

    cv.imshow("test", copy_image)
    cv.waitKey(0)

    
