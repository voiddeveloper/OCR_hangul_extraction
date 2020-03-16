""" 폴더안에 있는 이미지의 확장자를 jpg로 바꾸는 코드"""

folder_path = 'C:/ttf_result/malgun/'

from os import rename, listdir
for i in listdir('C:/ttf_result/malgun/'):
    new_file = i.split(".")[0]
    rename(folder_path + i, folder_path + new_file + ".jpg")
