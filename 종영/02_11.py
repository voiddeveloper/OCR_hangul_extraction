import cv2
import numpy as np
import  time
"""
이미지를 16가지 색으로 묶고 해당 색 별로 추출해 skeleton을 실행 후 
컨투어를 따서 길이가 3이하인건 삭제하고  가로 or 세로 직선을 포함하는 컨투어를 표시해보자 
"""

start=time.time()
def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

#이미지 불러오기
# path="color_16.jpg"
path="../image/2.jpg"
# path="../test1.png"

img=cv2.imread(path,0)
#마지막 컨투어를 그릴 이미지 (원본이미지를 복사)
result_contour_img=cv2.imread(path)
result_contour_img1=cv2.imread(path)

print(img.shape)
# 이진화
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV )
#아직 글씨가 흰색이냐 검은색이냐에 따라 이미지 반전을 해줘야함
#나중에 색별로 뽑아서 흰색으로 그리거나 하면 값을 고정할 수 있음
img=cv2.bitwise_not(img)
#정확히 이해하진 못했지만 이진화 처리된 이미지를 축소함
img=skeletonize(img)
kernel = np.ones((3, 3), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#컨투어 box의 좌표를 저장한다
contour_box_list=[]
for i,con in enumerate(contours):
    x_min = 9999
    x_max = 0
    y_min = 9999
    y_max = 0
    cv2.drawContours(result_contour_img1, contours, i, (0,0,255), 1)

    cimg = np.zeros_like(img)
    # print(cimg.shape)
    cv2.drawContours(cimg, contours, i, color=(255, 255, 255), thickness=-1)
    check = 0



    for index, j in enumerate(contours[i]):

        if x_min > j[0][0]:
            x_min = j[0][0]

        if x_max < j[0][0]:
            x_max = j[0][0]

        if y_min > j[0][1]:
            y_min = j[0][1]

        if y_max < j[0][1]:
            y_max = j[0][1]
    #가로 or 세로 직선이 있는지 판단

    for k in range(y_min,y_max+1):
        for h in range(x_min,x_max+1):
            try:
                if cimg[k][h]==255 and (cimg[k][h]==cimg[k][h+1] and cimg[k][h]==cimg[k][h+2] and cimg[k][h]==cimg[k][h+3]):
                    check=1
                    break
                if cimg[k][h]==255 and (cimg[k][h]==cimg[k+1][h] and cimg[k][h]==cimg[k+2][h]and cimg[k][h]==cimg[k+3][h]):
                    check = 1
                    break
            except IndexError:
                pass
        if check==1:
            break
    if check==1:
        # print(len(np.where(cimg==255)[0]))
        # cv2.imshow("qweqwe",cimg)
        # cv2.waitKey(0)
        contour_box_list.append([x_min,y_min,x_max,y_max])
# pixel_list=np.where(img==255)
# print(pixel_list)
# print(img[220][399])
# result_box_list=[]
#
# for i in contour_box_list:
#     check=0
#     # print("컨투어리스트"+str(i))
#     for j in range(i[1],i[3]+1):
#         # print(j)
#         for k in range(i[0],i[2]+1):
#             # print(k)
#             # exit()
#             try:
#                 if img[j][k]==255 and (img[j][k]==img[j][k+1]==img[j][k+2]):
#                     check=1
#                     break
#                 if img[j][k]==255 and (img[j][k]==img[j+1][k]==img[j+2][k]):
#                     check = 1
#                     break
#             except IndexError :
#                 pass
#         if check==1:
#             break
#     if check==1:
#         result_box_list.append(i)
for i in contour_box_list:
    cv2.rectangle(result_contour_img,(i[0],i[1]),(i[2],i[3]),(0,0,255),1)
print(len(contour_box_list))
# print(len(contour_box_list))
print(time.time()-start)
cv2.imshow("qwe",img)
cv2.imshow("qwe123",result_contour_img)
cv2.imshow("qwe1",result_contour_img1)

cv2.waitKey(0)
