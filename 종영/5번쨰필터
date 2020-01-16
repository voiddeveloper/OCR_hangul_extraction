import cv2
import numpy as np
img=cv2.imread("test1.png")
img2=img.copy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#컨투어 내부에 있는 픽셀 색
for v,i in enumerate(range(len(contours))):
    lst_intensities = []
    #이미지와 동일한 크기의 검은색 이미지를 만든다
    cimg = np.zeros_like(img)
    #검은색 이미지에 컨투어 크기만큼 흰색으로 그림
    #컨투어 안에도 흰색으로 차있음
    print(i)
    cv2.drawContours(cimg,contours, i, color=255, thickness=-1)

    #흰색 좌표를 저장함
    pts = np.where(cimg == 255)
    #원본 이미지에서 해당 좌표에 어떤 색이 채워져 있는지 저장
    lst_intensities.append(img2[pts[0], pts[1]])
    # lst_intensities=list(set(map(tuple,lst_intensities)))
    print(lst_intensities[0])
    #중복 제거해주는 부분
    a=list(set([tuple(set(lst_intensities[0])) for lst_intensities[0] in lst_intensities[0]]))
    print(a)
    print(len(a))
    cv2.imshow(""+str(v),cimg)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
