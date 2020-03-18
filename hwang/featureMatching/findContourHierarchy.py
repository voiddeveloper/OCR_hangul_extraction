import cv2
import numpy as np

############################################################################
# 2020-01-27 종영이 코드를 이어 받아서 작업
# contour 내부의 contour 정보를 얻어내기 위한 코드
#
# ## 목표 ##
# 글씨는 'ㅁ' 처럼 내부에 구멍이 뚫린 글씨들이 많다.
# contour의 색상 값을 구할 때, 이런 구멍 부분을 제거해야 원하는 값을 구할 수 있다.
# 따라서 contour의 자식 영역 부분을 삭제할 수 있어야 한다.
#
# ## 방법 ##
# contour의 부모/자식 관계를 구한다.
# 자식 영역의 contour 부분에 검은색을 칠한다. (binary 이미지이기 때문에 검은색을 칠하면 없는 영역으로 인식할 것이다.)
# 칠하는 과정이 끝나면, 부모 contour의 영역을 다시 계산한다.
############################################################################

# 이미지 읽기 및 contour, hierarchy 구하기
img=cv2.imread("hwang/imgSet/test1_1.png")
img2=img.copy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# count: 자신의 position값을 확인하기 위한 용도
# parentList: 자식이 있는 contour 정보
count = 0
parentList = []
for i in hierarchy[0]:
    # 부모 contour가 존재한다면
    if i[3] > -1:
        print(count, "번째 :",i)
        # 자기 자신의 index, 부모의 index를 저장한다.
        parentList.append((count, i[3]))
    count += 1

print("결과 :", parentList)

# listCount: 부모 contour인지 확인하기 위한 용도
listCount = 0
#컨투어 내부에 있는 픽셀 색
for v,i in enumerate(range(len(contours))):
    lst_intensities = []
    #이미지와 동일한 크기의 검은색 이미지를 만든다
    cimg = np.zeros_like(img)
    #검은색 이미지에 컨투어 크기만큼 흰색으로 그림
    #컨투어 안에도 흰색으로 차있음
    # print(i)
    cv2.drawContours(cimg,contours, i, color=255, thickness=-1)
    
    for cnt in parentList:
        # 만약 지금 그리는 contour가 부모 contour 라면
        if listCount == cnt[1]:
            # 자식 contour 영역은 검은색으로 그린다.
            cv2.drawContours(cimg,contours, cnt[0], color=0, thickness=-1)

    #흰색 좌표를 저장함
    pts = np.where(cimg == 255)
    #원본 이미지에서 해당 좌표에 어떤 색이 채워져 있는지 저장
    lst_intensities.append(img2[pts[0], pts[1]])
    # lst_intensities=list(set(map(tuple,lst_intensities)))
    # print(lst_intensities[0])
    #중복 제거해주는 부분
    a=list(set([tuple(set(lst_intensities[0])) for lst_intensities[0] in lst_intensities[0]]))
    # print(a)
    # print(len(a))
    cv2.imshow(""+str(v),cimg)

    # cv2.destroyAllWindows()
    listCount += 1

cv2.waitKey(0)