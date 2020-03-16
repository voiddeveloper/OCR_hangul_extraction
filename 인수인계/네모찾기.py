"""
네모찾기 알고리즘
흰색 이미지위에 있는 네모를 찾는 알고리즘

완벽한 알고리즘 아님
시간 오래걸림
픽셀이 깨지는것을 대비해서 군데군데 오차값을 둠


1.  먼저 0,0 부터 이미지의 끝까지 전부 픽셀을 검사함
    현재픽셀기준 오른쪽 픽셀과 아래 픽셀을 비교한다
    현재 픽셀과 오른쪽 혹은 아래 픽셀의 색이 다르다면 해당 픽셀좌표를 저장한다
    (픽셀의 차이 값은 유클라디안 거리값 구하기)
2.  1번에서 구한 픽셀들을 가지고 가로선인지 세로선인지 판단함
    (픽셀좌표들은 무작위로 리스트에 저장되어있음)
    판단하는 기준은 반복문을 돌면서 색이 다른 픽셀의 좌표가 양옆에 있는지
    혹은 위,아래로 있는지 판단한다
    결과는 가로선리스트 , 세로선 리스트 따로 저장된다

3.  픽셀의 색마다 구분짓지 않고 바로 옆의 픽셀의 색상 값으로만 비교하다보니
    2번,4번 이미지의 경우 결과가 잘나오지만 (네모만있는경우)
    1번,3번의경우 내부에있는 네모가 나오지않는다 (네모들끼리 겹쳐있는경우)
    그래서 가로선에 세로의 끝점들이 포함되어있다면
    가로선을 2개로 나눠서 저장하는 부분을 따로 만들었다
    (가로선을 1개로만 판단하다보니 겹쳐져있는 네모는 나오지않음)
    (네모를 구분하는 알고리즘 = 가로선의 양 끝점은 세로선의 양 끝점, 2개중에 1개는 만난다
    그렇게 가로선 1개와 연결되는 세로선 2개를 구하고 세로선 2개와 연결되는 가로선1개를 구해 네모를 구한다)

    이부분을 2번 실행한다
    (겹치는 네모가 많으면 많을수록 많이 실행해야하지만 임의로 2번만 실행)

4.  위에서 만들어놓은 가로선들과 세로선들을 가지고 네모를 찾는다

    (네모를 구분하는 알고리즘 = 가로선의 양 끝점은 세로선의 양 끝점, 2개중에 1개는 만난다
    그렇게 가로선 1개와 연결되는 세로선 2개를 구하고 세로선 2개와 연결되는 가로선1개를 구해 네모를 구한다)

"""
import cv2
import time
import math
from PIL import Image
from operator import eq
start=time.time()
image_s='ppt/p1.jpg'
img = cv2.imread(image_s, cv2.IMREAD_COLOR)



image = Image.open(image_s)
y = img .shape[0]
x = img.shape[1]
print(x)
print(y)

"""
1번 
"""
width_pixels=[]
hight_pixels=[]

#픽셀의 다른부분 캐치
for l in range(y):
    # print(l)
    for f in range(x):
        if f <= x-2 and l <= y-2:
            w1,w2,w3=image.getpixel((f, l))
            r1,r2,r3=image.getpixel((f, l+1))
            d1,d2,d3=image.getpixel((f+1, l))


            right_range = math.sqrt((w1 - r1) ** 2 + (w2 - r2) ** 2 + (w3 - r3) ** 2)
            down_range = math.sqrt((w1 - d1) ** 2 + (w2 - d2) ** 2 + (w3 - d3) ** 2)

            if right_range>30:
              width_pixels.append([f, l])
            if down_range>30:
              hight_pixels.append([f, l])

width_pixels.sort(key=lambda x:x[1])
hight_pixels.sort(key=lambda x:x[0])



"""
2번
"""
width_line={}

#가로선 리스트 찾기
#가로로 바로 옆에 연결되어있는 픽셀들은 가로선이라고 정함
for i in width_pixels :
    #r의 길이가 0이 아니라면 비교
    if len(width_line) !=0:
        check=0
        for j in width_line.keys():
            check=0
            # print("a"+str(r[j]))
            for k,v in enumerate(width_line[j]):
                # print("b"+str(v))
                # if (v[0]-3<=i[0] and v[0]+3>=i[0]) and (v[1]-3<=i[1] and v[1]+3>=i[1]):
                if (v[0]+1==i[0] and v[1]==i[1]) or (v[0]+2==i[0] and v[1]==i[1])\
                        or (v[0]-1==i[0] and v[1]==i[1]) or (v[0]-2==i[0] and v[1]==i[1]):
                    rr=list(width_line[j])
                    rr.append([i[0],i[1]])
                    width_line[j]=rr
                    try:
                        width_pixels.remove(i)
                    except ValueError :
                        pass
                    break
                if k==len(width_line[j])-1:
                    check=1
        if check==1:
            # print("asd" + str(i))
            width_line[len(width_line)] = [i]
    #처음 r의 길이가 0이라면
    else:
        width_line[len(width_line)]=[i]


#길이가 10픽셀 이하인 선은 제거함
for i in list(width_line):
    if len(width_line[i]) <=10:
        del(width_line[i])

#알아보기 편하게 정렬함
sort=[]
for i in width_line.keys():
    sort.append(width_line[i])
width_line={}
for i,v in enumerate(sort):
    width_line[i]=v

print("1",time.time() - start)




#세로선 리스트 찾기
#세로로 바로 위,아래에 연결되어있는 픽셀들은 세로선이라고 정함
#픽셀의 깨짐에 따라 완벽한 값이 나오지 않기때문에 반복문 내에 조건문으로 어느정도 오차를줬음
#height_line = 세로선들의 dic
height_line={}
for i in hight_pixels :
    #r의 길이가 0이 아니라면 비교
    if len(height_line) !=0:
        check=0
        for j in height_line.keys():
            check=0
            # print("a"+str(r[j]))
            for k,v in enumerate(height_line[j]):
                if (v[1]+1==i[1] and v[0]==i[0]) or (v[1]+2==i[1] and v[0]==i[0])\
                        or (v[1]-1==i[1] and v[0]==i[0]) or (v[1]-2==i[1] and v[0]==i[0]):
                    rr=list(height_line[j])
                    rr.append([i[0],i[1]])
                    height_line[j]=rr
                    try:
                        hight_pixels.remove(i)
                    except ValueError :
                        pass
                    break
                if k==len(height_line[j])-1:
                    check=1
        if check==1:
            # print("asd" + str(i))
            height_line[len(height_line)] = [i]
    #처음 s의 길이가 0이라면
    else:
        height_line[len(height_line)]=[i]



#길이가 10픽셀 이하인 선은 제거
for i in list(height_line):
    if len(height_line[i]) <=10:
        del(height_line[i])



#알아보기 편하게 정렬함
sort=[]
for i in height_line.keys():
    sort.append(height_line[i])
height_line={}
for i,v in enumerate(sort):
    height_line[i]=v

print("2",time.time() - start)




print("세로선의 개수", len(height_line))
print("가로선의 개수", len(width_line))

#테두리도 픽셀로 저장
#네모가 테두리에 딱 붙어있을경우 현재픽셀과 다음픽셀을 비교할수 없기떄문에 따로 리스트를 만들어 넣어줌
height_right=[]
for i in range(0,y):
    height_right.append([0, i])
height_line[len(height_line)]=height_right
height_left=[]
for i in range(0,y):
    height_left.append([x - 1, i])
height_line[len(height_line)]=height_left

width_top=[]
for i in range(0,x):
    width_top.append([i, 0])
width_line[len(width_line)]=width_top
width_bottom=[]
for i in range(0,x):
    width_bottom.append([i, y - 1])
width_line[len(width_line)]=width_bottom

print("세로선의 개수", len(height_line))
print("가로선의 개수", len(width_line))




"""
3번
"""
width_list=[]
height_list=[]

for i in height_line.keys():
    for s_,j in enumerate(height_line[i]):
        for k in width_line.keys():
            for r_,h in enumerate(width_line[k]):
                height_top=[]
                height_bottom=[]
                width_left=[]
                width_right=[]
                # print(s[i][s_])
                # print(r[k][r_])

                if height_line[i][s_][0]-3<=width_line[k][r_][0] and height_line[i][s_][0]+3>=width_line[k][r_][0] and height_line[i][s_][1]-3<=width_line[k][r_][1] and height_line[i][s_][1]+3 >= width_line[k][r_][1]:

                    width_left=[]
                    width_right=[]
                    height_top=[]
                    height_bottom=[]
                    for q in range(0,s_+1):
                        height_top.append([height_line[i][q][0], height_line[i][q][1]])
                    for q in range(s_, len(height_line[i])):
                        height_bottom.append([height_line[i][q][0], height_line[i][q][1]])
                    for q in range(0,r_+1):
                        # print(r[k][q][0])
                        width_left.append([width_line[k][q][0], width_line[k][q][1]])
                    for q in range(r_, len(width_line[k])):
                        width_right.append([width_line[k][q][0], width_line[k][q][1]])

                    if len(width_left)>=10:
                        width_list.append(width_left)
                    if len(width_right)>=10:
                        width_list.append(width_right)
                    if len(height_top)>=10:
                        height_list.append(height_top)
                    if len(height_bottom) >= 10:
                        height_list.append(height_bottom)





#위에서 나눠진 가로,세로선의 중복을 제거하는과정
def tuples(A):
    try: return tuple(tuples(a) for a in A)
    except TypeError: return A
height_list=set(tuples(height_list))
width_list=set(tuples(width_list))

#중복이 제거된 새로운 가로,세로선을 원래 가로,세로선 dic 에 추가함
for i in width_list :
    list_qr=[]
    for j in i :
        list_qr.append([j[0],j[1]])
    width_line[len(width_line)]=list_qr
for i in height_list:
    list_qr = []
    for j in i:
        list_qr.append([j[0], j[1]])
    height_line[len(height_line)] = list_qr

print("11처리후s의 개수", len(height_line))
print("11처리후r의 개수", len(width_line))

#세로운 가로,세로선을 추가하고 중복제거
over_r=[]
over_s=[]
for i in width_line.keys():
    over_r.append(width_line[i])
for i in height_line.keys():
    over_s.append(height_line[i])
over_s=set(tuples(over_s))
over_r=set(tuples(over_r))
width_line={}
height_line={}
for i,v in enumerate(over_r):
    width_line[i]=v
for i,v in enumerate(over_s):
    height_line[i]=v


print("22처리후s의 개수", len(height_line))
print("22처리후r의 개수", len(width_line))



"""
3번을 다시 반복함
"""

width_list=[]
height_list=[]
# #s 는 세로선
for i in width_line.keys():
    for r_,j in enumerate(width_line[i]):
        for k in height_line.keys():
            for s_,h in enumerate(height_line[k]):
                width_left=[]
                width_right=[]
                height_top=[]
                height_bottom=[]
                # print(s[i][s_])
                # print(r[k][r_])

                if width_line[i][r_][0]-3<=height_line[k][s_][0] and width_line[i][r_][0]+3>=height_line[k][s_][0] and width_line[i][r_][1]-3<=height_line[k][s_][1] and width_line[i][r_][1]+3 >= height_line[k][s_][1]:

                    width_left=[]
                    width_right=[]
                    height_top=[]
                    height_bottom=[]
                    for q in range(0,r_+1):
                        width_left.append([width_line[i][q][0], width_line[i][q][1]])
                    for q in range(r_, len(width_line[i])):
                        width_right.append([width_line[i][q][0], width_line[i][q][1]])
                    for q in range(0,s_+1):
                        # print(r[k][q][0])
                        height_top.append([height_line[k][q][0], height_line[k][q][1]])
                    for q in range(s_, len(height_line[k])):
                        height_bottom.append([height_line[k][q][0], height_line[k][q][1]])
                    # print(list_s1)
                    # print(list_s2)
                    # print(list_r1)
                    # print(list_r2)
                    if len(width_left)>=10:
                        width_list.append(width_left)
                    if len(width_right)>=10:
                        width_list.append(width_right)
                    if len(height_top)>=10:
                        height_list.append(height_top)
                        # s_list[len(s_list)]=list_s1
                    if len(height_bottom) >= 10:
                        height_list.append(height_bottom)
                        # s_list[len(s_list)]=list_s2










#위에서 나눠진 가로,세로선의 중복을 제거하는과정
def tuples(A):
    try: return tuple(tuples(a) for a in A)
    except TypeError: return A
height_list=set(tuples(height_list))
width_list=set(tuples(width_list))

#중복이 제거된 새로운 가로,세로선을 원래 가로,세로선 dic 에 추가함
for i in width_list :
    list_qr=[]
    for j in i :
        list_qr.append([j[0],j[1]])
    width_line[len(width_line)]=list_qr
for i in height_list:
    list_qr = []
    for j in i:
        list_qr.append([j[0], j[1]])
    height_line[len(height_line)] = list_qr

print("11처리후s의 개수", len(height_line))
print("11처리후r의 개수", len(width_line))

#세로운 가로,세로선을 추가하고 중복제거
over_r=[]
over_s=[]
for i in width_line.keys():
    over_r.append(width_line[i])
for i in height_line.keys():
    over_s.append(height_line[i])
over_s=set(tuples(over_s))
over_r=set(tuples(over_r))
width_line={}
height_line={}
for i,v in enumerate(over_r):
    width_line[i]=v
for i,v in enumerate(over_s):
    height_line[i]=v


print("22처리후s의 개수", len(height_line))
print("22처리후r의 개수", len(width_line))




"""
4번
"""
last=[]
total=1
#겹치지 않은 네모 찾기
#가로 리스트 반복문
for i in width_line.keys():
    #세로 리스트 반복문
    for j in height_line.keys():
        #가로의 처음과 왼쪽 세로의 첫번째가 같다면
        if abs(width_line[i][0][0] - height_line[j][0][0])<=3 and abs(width_line[i][0][1] - height_line[j][0][1])<=3 :
            for jj in height_line.keys():
                if eq(j,jj):
                    #같은 리스트인거 넘기기
                    continue
                else:
                    #왼쪽세로가 같으니 가로의 끝과 오른쪽 세로의 첫번째 점과 비교해야함
                    #다시 세로 리스트를 반복
                    if abs(width_line[i][-1][0] - height_line[jj][0][0])<=3 and abs(width_line[i][-1][1] - height_line[jj][0][1])<=3:
                        #오른쪽 세로 까지 맞다면 이제 가로를 다시 비교
                        for ii in width_line.keys():
                            if eq(i,jj):
                                continue
                            else:
                                #아래 가로의 첫번쨰와 마지막 점을 양쪽 세로 마지막 점과 비교
                                #r[ii] 아래가로
                                #s[jj] 오른쪽 세로
                                # print(r[ii])
                                if (abs(width_line[ii][0][0] - height_line[j][-1][0]) <= 3 and abs(width_line[ii][0][1] - height_line[j][-1][1]) <= 3) and (abs(width_line[ii][-1][0] - height_line[jj][-1][0]) <= 3 and abs(width_line[ii][-1][1] - height_line[jj][-1][1]) <= 3):
                                    # print("ㅋ")
                                    last.append([height_line[j][0][0], height_line[j][0][1], height_line[jj][-1][0], height_line[jj][-1][1]])
print("3",time.time() - start)






#중복 삭제하는 부분
from  pandas import *
import numpy as np
last=np.array(last)
last=DataFrame(last).drop_duplicates().values
last=last.tolist()
last=list(last)
result=[]
for i in last:
    if len(result)!=0:
        for v,j in enumerate(result):
            #원래는 3이어야함
            if ((j[0]-10<=i[0] and j[0]+10>=i[0]) and (j[1]-10<=i[1] and j[1]+10>=i[1]))  and ((j[2]-10<=i[2] and j[2]+10>=i[2]) and (j[3]-10<=i[3] and j[3]+10>=i[3])):
                break
            if v==len(result)-1:
                result.append(i)
    else:
        result.append(i)


print(len(result))

for i in result:
    print(i)


    cv2.rectangle(img, (i[0], i[1]), (i[2],i[3]), (0, 0, 255), 2)
    cv2.putText(img, str(total), (i[0],i[1]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,
                (0, 0, 0), 1)
    total = total + 1
print("5",time.time()-start)
cv2.imshow("a",img)
cv2.waitKey(0)





#겹쳐져있는 이미지 찾는 부분
# for i in r.keys():
#     r1,r2=r[i]
#     for j in s.keys():




