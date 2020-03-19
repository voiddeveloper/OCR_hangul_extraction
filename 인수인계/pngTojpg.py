""" 
폴더안에 있는 이미지의 확장자를 jpg로 바꾸는 코드

ex ) 폴더안에 image1.png , image2.png 파일이 있다면 image1.jpg , image2.jpg로 바꾸는 코드임
"""

from os import rename, listdir

# 폰트파일이 있는 폴더의 경로
folder_path = 'C:/ttf_result/malgun/'

for i in listdir(folder_path):
    new_file = i.split(".")[0]
    rename(folder_path + i, folder_path + new_file + ".jpg")
  
  
 
