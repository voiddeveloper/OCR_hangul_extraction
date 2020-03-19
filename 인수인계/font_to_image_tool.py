"""

폰트 파일을 가지고 글자의 이미지를 추출하는 툴
F2를 누르면 실행된다

c드라이브 밑에 ttf 폴더를 만들고 그안에 폰트 파일을 넣어놓는다 (ttf파일은 여러개 있어도 상관없음)
(폴더에 들어있는 ttf파일을 전부 실행함)
ex) 나눔고딕.ttf

F2로 실행을 하기되면 툴 하단에 (나눔고딕.ttf 진행중) 표시가되면 끝날경우 끝 이라고 표시된다
실행이 끝나면 c드라이브 밑에 ttf_result 폴더가 생기고 그안에 폰트별로 저장이된다
이미지 넓이,높이,폰트크기는 설정을 할 수 있다


11172개의 글씨를 전부 이미지화 시킬수도있고 자음,모음만 따로 추출할수도있다
running메서드 안에 주석 확인
"""

"""
exe파일로 만들고싶다면 pyinstaller 공부할 것 
"""
 
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm,tqdm_notebook
import sys
import threading
from PyQt5.QtWidgets import  *
check=False


"""
툴의  ui나 행동 전부가 들어있는 클래스
"""
class back_window(QMainWindow):

    def keyPressEvent(self, e):
        global check

        from PyQt5.QtCore import Qt


# 키보드 입력을 받는부분
# F2를 누를시에 실행이된다
        if e.key() == Qt.Key_F2:
            if check == False:
                # running 메서드를 실행하는 
                t = threading.Thread(target=self.running)
                t.start()
                check=True
                self.info7.setText("진행중")
            else :
                check=False
                self.info7.setText("대기중")



    def __init__(self):
        super().__init__()
        self.initUI()
    
    
    #툴의 ui를 그리는부분
    def initUI(self):
        self.setGeometry(300,300,500,400)
        self.setWindowTitle('폰트 => 이미지 변환')
        self.info = QLabel("사용법", self)
        self.info.move(10, 20)
        self.info2=QLabel('1. c드라이브 바로 밑에 ttf 폴더를 만들기',self)
        self.info2.move(10,50)
        self.info2.resize(300,50)
        self.info3 = QLabel('2. 폴더안에 폰트파일 넣기', self)
        self.info3.move(10,70)
        self.info3.resize(250,50)
        self.info3 = QLabel('3. F2버튼 실행', self)
        self.info3.move(10,90)
        self.info3.resize(250,50)

        self.info4 = QLabel('이미지 넓이', self)
        self.info4.move(10,150)
        self.info4.resize(250,50)
        self.wwidth=QLineEdit(self)
        self.wwidth.setText("50")
        self.wwidth.move(10,200)

        self.info5 = QLabel('이미지 높이', self)
        self.info5.move(150, 150)
        self.info5.resize(250, 50)
        self.hhight = QLineEdit(self)
        self.hhight.setText("50")
        self.hhight.move(150, 200)

        self.info6 = QLabel('폰트 크기', self)
        self.info6.move(300, 150)
        self.info6.resize(250, 50)
        self.ffont = QLineEdit(self)
        self.ffont.setText("20")
        self.ffont.move(300, 200)

        self.info7 = QLabel('대기중', self)
        self.info7.move(10, 350)
        self.info7.resize(250, 50)

        self.show()

    # 폰트를 이미지화 시키는 메서드
    # F2를 누를시 스레드로 실행된다 
    def running(self):
        global check

        # 1. 11172 글자 전부 이미지화 시키는 범위
        #11172글자 전부를 이미지화 시키고싶다면 하단 start,end,name 주석 풀기
        #start 와 end 는 글자 유니코드의 범위이다
        start = "AC00"
        end = "D7A3"
        name= 'full'
 
 
        # 2. 자음과 모음만 이미지화 시키고싶다면 하단 start,end,name 주석 풀기
        #start 와 end 는 글자 유니코드의 범위이다
        # start = "3131"
        # end = "318E"
        # name='mini'

      
        # 유니코드 배열을 만들기 위한 문자의 범위 
        # 하단에 Hangul_Syllables 범위를 만들기 위해 사용됨
        co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
        co = co.split(" ")
        """
        유니코드 범위 0000 ~ FFFF 까지의 배열을 만드는 부분
        Hangul_Syllables 폰트 파일에서 한글을 추출하기위해 시작과 끝 유니코드 값이 저장되어있고 
        반복문에서 이 범위에 해당하는 이미지를 추출함
        """
        hangul_syllables = [a + b + c + d
                            for a in co
                            for b in co
                            for c in co
                            for d in co]

        hangul_syllables = np.array(hangul_syllables)
        print(hangul_syllables)
        
        # s = 출력하고자 하는 문자의 시작 유니코드값
        # e = 출력하고자 하는 문자의 끝나는 유니코드값
        # s~e 에 해당하는 유니코드 값을 추출한다 
        start_unicode = np.where(start == hangul_syllables)[0][0]
        end_unicode = np.where(end == hangul_syllables)[0][0]

        #start 부터 end 까지 범위 저장
        hangul_syllables = hangul_syllables[start_unicode: end_unicode + 1]
        # print(hangul_syllables)


        fonts = os.listdir("C:/ttf/")
        font_path = "C:/ttf/"

        #ttf폴더안에들어있는 폰트의 갯수만큼 반복문
        for ttf in tqdm(fonts):
            if not os.path.exists('C:/ttf_result'):
                os.mkdir('C:/ttf_result' )

            if ttf.endswith(".ttf"):
                print(ttf)
                self.info7.setText(str(ttf)+" 폰트 진행중")

                tt = ttf.split(".ttf")[0]
                if not os.path.exists('C:/ttf_result/' + tt+"_"+name):
                    os.mkdir('C:/ttf_result/' + tt+"_"+name)

                font = ImageFont.truetype(str(font_path) + ttf, int(self.ffont.text()))

                # 이미지 사이즈 지정
                # 사용자가 입력한 사이즈 대로 지정함
                text_width = int(self.wwidth.text())
                text_height = int(self.hhight.text())

                """
                사용자가 입력한 이미지 크기 만큼 흰색 도화지를 만든다음
                사용자가 입력한 폰트의 크기에 맞게 그린다 그리고 그 결과를 저장한다 
                
                """
                for uni in tqdm(hangul_syllables):

                    canvas = Image.new('RGB', (text_width, text_height), "white")

                    # 가운데에 그리기 (폰트 색: 검)
                    draw = ImageDraw.Draw(canvas)
                    draw_text = chr(int(uni, 16))
                    w, h = font.getsize(draw_text)
                    draw.text(((text_width - w) / 2.0, (text_height - h) / 2.0), draw_text, 'black', font)
                    canvas.save('C:/ttf_result/' + ttf.split(".ttf")[0] +"_"+name+ "/" + draw_text + '.png', "PNG")

                    if check == False :
                        break
                if check == False:

                    break
        if check==True:
            check=False
            self.info7.setText("끝")
        else :
            self.info7.setText("대기중")


if __name__=='__main__':

    app=QApplication(sys.argv)
    ex=back_window()
    sys.exit(app.exec_())

