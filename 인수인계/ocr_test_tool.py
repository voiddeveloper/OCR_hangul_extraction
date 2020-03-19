# -*- encoding: utf8 -*-


"""
 

구글api , 카카오api, 테서렉트 엔진을 사용하는 툴
테스트를 편리하게 하기위해 만든 툴로 코드 최적화는 하지않음
테서렉트 엔진을 사용하기위해서는 테서렉트 엔진이 미리 깔려있어야함
구글api와 카카오api 는 키를 직접 발급 받아서 kakao_key,google_json 전역변수에 넣어줘야함
카카오api는 키값을 넣고 구글api 는 json 파일의 경로를 넣어야함

ex)
카카오api = kakao123123aa1111aaa2222
구글api = ../../google.json

이미지에 글씨를 그리기위해 폰트 파일이 있어야함
전역변수 fontpath 에 폰트의 경로가 들어감
ex) fontpath= ../../나눔고딕.ttf
"""
"""
엔진에 따라서 이미지 이름 or 이미지 경로에 한글이 들어가면 안되는 경우가있으니
이미지 경로와 이름은 영어로 해야함
"""
"""
툴을 사용할땐 이미지를 드래그앤 드랍으로 올리고 실행 or 저장 버튼을 이용해 사용가능
이미지는 현재 jpg , png 2가지의 경우만 가능
저장을 클릭하면 현재 경로에 result 폴더가 생성되며 결과가 다 저장된다
"""
"""
테서렉트의 경우 한글,영어,기타를 선택하는 부분이있음 
테서렉트 엔진에서 사용하는 학습 데이터를 사용하며
한글의 경우 kor.traineddata 영어의 경우 eng.traineddate 파일이 테서렉트 파일 경로 내부에 있어야함
만약 다른 학습 데이터를 사용할경우 기타를 선택해 이름을 적어주면된다 (이것도 테서렉트 파일 경로 내부에있어야함)
nova.traineddata 를 사용하고싶을 경우  기타에 nova를 입력하면됨
"""


"""
exe 파일로 만들고 싶은경우
pyinstaller를 사용하면 exe파일로 만들 수 있다
"""
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
from pytesseract import Output
from google.cloud import vision
from google.cloud.vision import types
import time
import io
import os
import json
from PIL import Image, ImageDraw
import cv2
import requests
import sys
import numpy as np
from PIL import ImageFont
import threading
from PyQt5.QtWidgets import *

image = ''
fileList_Path = []
select_image = ''
fontpath = "나눔고딕L.ttf"
kakao_key='kakaoKey'
google_json='google.json'




class Mainwindow(QMainWindow):



    def __init__(self):
        super().__init__()
        self.initUI()

    # 툴의 ui를 그리는 부분

    def initUI(self):
        self.window = QVBoxLayout(self)
        self.label1 = QLabel("", self)
        self.combo1 = QComboBox(self)
        self.combo1.addItem("--선택--")
        self.combo1.addItem("테서렉트")
        self.combo1.addItem("카카오api")
        self.combo1.addItem("구글 api")
        self.combo1.addItem("미  정")
        self.combo1.move(30, 70)

        self.window.addWidget(self.combo1)
        self.window.addWidget(self.label1)

        self.combo1.activated[str].connect(self.ComboBoxEvent)

        self.textBox = QTextEdit(self)
        self.textBox.move(30, 100)
        self.textBox.resize(300, 400)
        self.textBox.setReadOnly(True)

        self.textBox1 = QTextEdit(self)
        self.textBox1.move(350, 100)
        self.textBox1.resize(300, 400)
        self.textBox1.setReadOnly(True)

        # 실행 저장 버튼
        self.running = QPushButton('실행', self)
        self.running.move(570, 650)
        self.running.clicked.connect(self.RunningClick)

        self.running1 = QPushButton('저장', self)
        self.running1.move(450, 650)
        self.running1.clicked.connect(self.SaveClick)

        # 라디오버튼
        self.radio1 = QRadioButton("한국어", self)
        self.radio1.move(200, 30)
        self.radio1.setChecked(True)
        self.radio1.clicked.connect(self.radioButtonClicked)

        self.radio2 = QRadioButton("영어", self)
        self.radio2.move(280, 30)
        self.radio2.clicked.connect(self.radioButtonClicked)

        self.radio3 = QRadioButton("", self)
        self.radio3.move(340, 30)
        self.radio3.clicked.connect(self.radioButtonClicked)

        self.fonttext = QTextEdit("", self)
        self.fonttext.move(370, 30)

        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)

        self.radio1.setVisible(False)
        self.radio2.setVisible(False)
        self.radio3.setVisible(False)
        self.fonttext.setVisible(False)


        # 드래그앤드롭
        self.setAcceptDrops(True)

        self.pagebutton0 = QButtonGroup

        self.setLayout(self.window)
        self.setGeometry(0, 0, 700, 700)
        self.setWindowTitle('Test Tool')
        self.show()

    """
    실행버튼 클릭 , 저장버튼 클릭 이벤트는 모두 스레드로 동작한다
    스레드를 사용하지않아도되는데 스레드를 사용하지않으면 결과가 나오기 전까지 툴의 모든게 멈춰버림
    """


    # 실행 버튼 클릭
    def RunningClick(self):
        print("실행버튼 클릭")
        if '미  정' in self.combo1.currentText() or '--선택--' in self.combo1.currentText():
            pass
        elif '구글 api' in self.combo1.currentText():
            self.textBox1.setText('')
            tt = threading.Thread(target=self.google_, args=('run',))
            tt.start()
        elif '카카오api' in self.combo1.currentText():
            self.textBox1.setText('')
            tt = threading.Thread(target=self.kakao, args=('run',))
            tt.start()
        elif '테서렉트' in self.combo1.currentText():
            self.textBox1.setText("")
            if self.radio1.isChecked():
                print('한국어')
                tt = threading.Thread(target=self.tesseract, args=('kor', 'run',))
                tt.start()
            elif self.radio2.isChecked():
                print("영어")
                tt = threading.Thread(target=self.tesseract, args=('eng', 'run',))
                tt.start()
            else:
                tt = threading.Thread(target=self.tesseract, args=(self.fonttext.toPlainText(), 'run',))
                tt.start()

    # 저장 버튼 클릭
    def SaveClick(self):
        if '미  정' in self.combo1.currentText() or '--선택--' in self.combo1.currentText():
            pass
        elif '구글 api' in self.combo1.currentText():
            self.textBox1.setText('')
            tt = threading.Thread(target=self.google_, args=('save',))
            tt.start()
        elif '카카오api' in self.combo1.currentText():
            self.textBox1.setText('')
            tt = threading.Thread(target=self.kakao, args=('save',))
            tt.start()

        elif '테서렉트' in self.combo1.currentText():
            self.textBox1.setText("")
            if self.radio1.isChecked():
                print('한국어')
                tt = threading.Thread(target=self.tesseract, args=('kor', 'save',))
                tt.start()
            elif self.radio2.isChecked():
                print("영어")
                tt = threading.Thread(target=self.tesseract, args=('eng', 'save',))
                tt.start()
            else:
                tt = threading.Thread(target=self.tesseract, args=(self.fonttext.toPlainText(), 'save',))
                tt.start()

    # 드래그앤드랍을 할 수 있게 해주는 메소드
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            print("1")

        else:
            event.ignore()
            print("2")
    #드래그앤드랍 이벤트
    def dropEvent(self, event):
        global fileList_Path
        fileList_Path = []
        self.textBox.setText("")

        #이미지를 올렸을때 이미지들의 경로를 저장하는 부분

        files = [np.unicode(u.toLocalFile()) for u in event.mimeData().urls()]
        for f in files:
            if '.jpg' in f or '.png' in f or '.JPG' in f or '.PNG' in f:
                fileList_Path.append(f)
                self.textBox.append(str(f).split("/")[-1])
        if (len(fileList_Path) == 0):
            self.textBox.setText("없음")

    # 콤보박스 드롭다운 이벤트
    def ComboBoxEvent(self, text):
        if '테서렉트' in text:
            self.radio1.setVisible(True)
            self.radio2.setVisible(True)
            self.radio3.setVisible(True)
            self.fonttext.setVisible(True)

        else:
            self.radio1.setVisible(False)
            self.radio2.setVisible(False)
            self.radio3.setVisible(False)
            self.fonttext.setVisible(False)

        if '카카오api' in text or '구글 api' in text:
            self.textBox1.setText('네트워크 통신으로 시간 체크 어려움')
        else:
            self.textBox1.setText('')

    # 라디오버튼 이벤트
    def radioButtonClicked(self):
        msg = ""
        if self.radio1.isChecked():
            msg = "한국어"
        elif self.radio2.isChecked():
            msg = "영 어"

    def google_(self, active):
        print('구글')

        client = vision.ImageAnnotatorClient()

        for va in fileList_Path:

            # 이미지 읽기
            with io.open(va, 'rb') as image_file:
                content = image_file.read()

            image = types.Image(content=content)

            start = time.time()

            #실질적으로 구굴 api를 실행하는 부분
            response = client.document_text_detection(image=image)
            #labels 안엔 반환값들이 들어있음 (글씨의 좌표 , ocr인식후 결과 등등)
            labels = response.text_annotations
            end = time.time()
            self.textBox1.append(str(end - start))
            print(labels)
            #이미지 1장에는 글자의 좌표를 받아 박스를 그리고
            #나머지 1장에는 ocr 인식후 결과를 이미지에 그린다
            img = cv2.imdecode(np.fromfile(str(va), dtype=np.uint8), cv2.IMREAD_COLOR)
            img1 = cv2.imdecode(np.fromfile(str(va), dtype=np.uint8), cv2.IMREAD_COLOR)

            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            text = ''
            for i, d in enumerate(labels):
                line = d.bounding_poly.vertices
                if i == 0:
                    text = d.description

                """구글 api의 결과값을 토대로 글씨라고 추측되는 영역을 네모박스 치는 부분"""
                cv2.line(img, (line[0].x, line[0].y), (line[1].x, line[1].y), (0, 0, 255), 2)
                cv2.line(img, (line[0].x, line[0].y), (line[3].x, line[3].y), (0, 0, 255), 2)
                cv2.line(img, (line[1].x, line[1].y), (line[2].x, line[2].y), (0, 0, 255), 2)
                cv2.line(img, (line[2].x, line[2].y), (line[3].x, line[3].y), (0, 0, 255), 2)
                (x, y) = (line[3].x, line[3].y)
                if i != 0:
                    # fontpath  = 전역변수로 선언해놓은 폰트의 경로
                    # 이미지위에 글씨를 쓰는 부분
                    font = ImageFont.truetype(fontpath, 20)
                    draw.text((x, y), str(d.description), font=font, fill=(0, 0, 255, 255))

            img1 = np.array(img_pil)


            """
            run 클릭시 박스를 친 이미지 , 결과를 그린 이미지 2장이 순서대로 출력되고
            save를 클릭할시 현재 경로에 result 파일을 만들고  이미지와 text로 결과를 저장함 
                    result 파일이 이미 있다면 만들지 않음
            """
            if 'run' in active:
                cv2.imshow("google", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("google", img1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                if not (os.path.isdir('google_result')):
                    os.makedirs(os.path.join('google_result'))
                result, n = cv2.imencode(
                    'google_result/' + str(va).split("/")[-1].split(".")[0] + "_local" + ".png", img,
                    params=None)
                #이미지 2장을 저장하는부분
                if result:
                    with open(
                            'google_result/' + str(va).split("/")[-1].split(".")[0] + "_local" + ".png",
                            mode='w+b') as f:
                        n.tofile(f)
                result, n = cv2.imencode(
                    'google_result/' + str(va).split("/")[-1].split(".")[0] + "_reco" + ".png", img1,
                    params=None)
                if result:
                    with open(
                            'google_result/' + str(va).split("/")[-1].split(".")[0] + "_reco" + ".png",
                            mode='w+b') as f:
                        n.tofile(f)

                #결과 텍스트를 저장하는 부분
                text_file = open(
                    'google_result/' + str(va).split("/")[-1].split(".")[0] + "_result.txt", 'w',
                    encoding='utf8')
                text_file.write(str(end - start) + "\n" + text)
                text_file.close()

    def kakao(self, active):
        print("카카오")
        LIMIT_BOX = 1024

        for va in fileList_Path:
            image_path, appkey = va, kakao_key
            start = time.time()
            #이미지가 너무 큰 경우 이미지 리사이즈 하는 부분
            resize_impath = self.kakao_ocr_resize(image_path)

            if resize_impath is not None:
                image_path = resize_impath
                print("원본 대신 리사이즈된 이미지를 사용합니다.")

            #카카오api를 이용한 결과값 반환
            #카카오api를 사용해 글자의 좌표를 얻는부분
            output = self.kakao_ocr_detect(image_path, appkey).json()

            boxes = output["result"]["boxes"]
            boxes = boxes[:min(len(boxes), LIMIT_BOX)]

            #카카오api를 이용한 결과값 반환
            #ocr의 결과 값 반환 (이미지내에 있는  글씨)
            output1 = self.kakao_ocr_recognize(image_path, boxes, appkey).json()
            end = time.time()
            self.textBox1.append(str(end - start))

            output_xy = []
            print(output1)

            print("-------------------------")
            print(output)
            print(output['result']['boxes'])

            img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)

            img_re = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            ex_list = []
            print(len(output['result']['boxes']))
            #글자의 좌표를 가지고 이미지에 박스를 그리는 부분
            for f, i in enumerate(output['result']['boxes']):
                cv2.line(img, (i[0][0], i[0][1]), (i[1][0], i[1][1]), (0, 0, 255), 2)
                cv2.line(img, (i[0][0], i[0][1]), (i[3][0], i[3][1]), (0, 0, 255), 2)
                cv2.line(img, (i[1][0], i[1][1]), (i[2][0], i[2][1]), (0, 0, 255), 2)
                cv2.line(img, (i[2][0], i[2][1]), (i[3][0], i[3][1]), (0, 0, 255), 2)
                output_xy.append([i[3][0], i[3][1]])

            #실행을 클릭했을때 이미지를 나타냄
            if 'run' in active:
                print("run")
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            #저장을 클릭했을때 이미지를 저장
            #result 폴더를 만들고 이미지 저장
            #result 폴더가 이미 있다면 그냥 이미지 저장
            else:
                print("save")

                if not (os.path.isdir('kakao_result')):
                    os.makedirs(os.path.join('kakao_result'))
                result, n = cv2.imencode(
                    'kakao_result/' + str(va).split("/")[-1].split(".")[0] + "_local" + ".png", img,
                    params=None)
                if result:
                    with open(
                            'kakao_result/' + str(va).split("/")[-1].split(".")[0] + "_local" + ".png",
                            mode='w+b') as f:
                        n.tofile(f)

            print(output1)
            print(len(output1['result']['recognition_words']))

            #ocr의 결과 를 가지고 이미지에 글씨를 쓰는 부분
            font = ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(img_re)
            draw = ImageDraw.Draw(img_pil)
            print(output1['result']['recognition_words'])
            ex_list = output1['result']['recognition_words']
            ex_list = list(reversed(ex_list))
            print(ex_list)
            for i, v in enumerate(output_xy):
                (x, y) = (v[0], v[1])
                try:
                    draw.text((x, y), str(output1['result']['recognition_words'][i]), font=font, fill=(0, 0, 255, 255))
                except IndexError:
                    pass

            img1 = np.array(img_pil)

            #실행을 클릭했을때는 그냥 보여주기만함
            if 'run' in active:
                cv2.imshow('img', img1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #저장을 클릭했을때 이미지의 결과 (txt) 파일과 글씨가 그려진 이미지를 저장
            else:
                result, n = cv2.imencode(
                    'kakao_result/' + str(va).split("/")[-1].split(".")[0] + "_reco" + ".png", img1,
                    params=None)
                if result:
                    #
                    with open(
                            'kakao_result/' + str(va).split("/")[-1].split(".")[0] + "_reco" + ".png",
                            mode='w+b') as f:
                        n.tofile(f)
                text = ''
                for v in ex_list:
                    text = text + " " + v
                text_file = open(
                    'kakao_result/' + str(va).split("/")[-1].split(".")[0] + "_result.txt", 'w',
                    encoding='utf8')
                text_file.write(str(end - start) + "\n" + text)
                text_file.close()

    def tesseract(self, lang, active):
        print('테서렉트')
        print(fileList_Path)
        for va in fileList_Path:
            print(u"" + str(va))
            img = cv2.imdecode(np.fromfile(str(va), dtype=np.uint8), cv2.IMREAD_COLOR)
            img1 = cv2.imdecode(np.fromfile(str(va), dtype=np.uint8), cv2.IMREAD_COLOR)
            start = time.time()


            #실질적으로 테서렉트를 실행하는 부분
            d = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)
            print(d)
            end = time.time()
            self.textBox1.append(str(end - start))

            #테서렉트 결과 중에 데이터가 총 몇개가 출력됐는지 확인하는부분
            n_boxes = len(d['level'])

            text = ""

            #출력된 갯수만큼 반복문을 돌며 이미지위에 네모박스를 그리는 부분
            for i in (range(n_boxes)):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 실행하는부분
            if 'run' in active:
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                # 저장하는 부분
                #result폴더가 있는지 확인한후에 있으면 그냥 저장하고
                #없으면 result폴더를 만들고 저장함
                if not (os.path.isdir('tesseract_result')):
                    os.makedirs(os.path.join('tesseract_result'))
                result, n = cv2.imencode(
                    'tesseract_result/' + str(va).split("/")[-1].split(".")[0] + "_local" + "_" + lang + ".png", img,
                )

                if result:
                    with open(
                            'tesseract_result/' + str(va).split("/")[-1].split(".")[0] + "_local" + "_" + lang + ".png",
                            mode='w') as f:
                        n.tofile(f)

                text = pytesseract.image_to_string(img1, lang=lang)

            # 이미지에 결과 글씨 쓰기
            font = ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(img1)
            draw = ImageDraw.Draw(img_pil)
            for i in range(n_boxes):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                try:

                    draw.text((x, y + h), d['text'][i], font=font, fill=(0, 0, 255, 255))
                except TypeError:
                    pass
            img1 = np.array(img_pil)
            #run을 클릭해서 실행하는 경우
            #글자가 그려진 이미지도 보여줌
            if 'run' in active:
                cv2.imshow('img', img1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            else:
            #save를 클릭해서 실행하는 경우 이미지와 결과 텍스트 저장
                result, n = cv2.imencode(
                    'tesseract_result/' + str(va).split("/")[-1].split(".")[0] + "_reco" + "_" + lang + ".png", img1,
                    params=None)
                if result:
                    with open(
                            'tesseract_result/' + str(va).split("/")[-1].split(".")[0] + "_reco" + "_" + lang + ".png",
                            mode='w+b') as f:
                        n.tofile(f)
                text_file = open(
                    'tesseract_result/' + str(va).split("/")[-1].split(".")[0] + "_" + lang + "_result.txt", 'w',
                    encoding='utf8')
                text_file.write(str(end - start) + "\n" + text)
                text_file.close()

    # 카카오 api
    def kakao_ocr_resize(self, image_path: str):
        """
        ocr detect/recognize api helper
        ocr api의 제약사항이 넘어서는 이미지는 요청 이전에 전처리가 필요.
        pixel 제약사항 초과: resize
        용량 제약사항 초과  : 다른 포맷으로 압축, 이미지 분할 등의 처리 필요. (예제에서 제공하지 않음)
        :param image_path: 이미지파일 경로
        :return:
        """
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        LIMIT_PX = 1024
        if LIMIT_PX < height or LIMIT_PX < width:
            ratio = float(LIMIT_PX) / max(height, width)
            image = cv2.resize(image, None, fx=ratio, fy=ratio)
            height, width, _ = height, width, _ = image.shape

            # api 사용전에 이미지가 resize된 경우, recognize시 resize된 결과를 사용해야함.
            image_path = "{}_resized.jpg".format(image_path)
            cv2.imwrite(image_path, image)

            return image_path
        return None

    def kakao_ocr_detect(self, image_path: str, appkey: str):
        """
        detect api request example
        :param image_path: 이미지파일 경로
        :param appkey: 카카오 앱 REST API 키
        """
        API_URL = 'https://kapi.kakao.com/v1/vision/text/detect'

        headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        jpeg_image = cv2.imencode(".jpg", image)[1]
        data = jpeg_image.tobytes()

        return requests.post(API_URL, headers=headers, files={"file": data})

    def kakao_ocr_recognize(self, image_path: str, boxes: list, appkey: str):
        """
        recognize api request example
        :param boxes: 감지된 영역 리스트. Canvas 좌표계: 좌상단이 (0,0) / 우상단이 (limit,0)
                        감지된 영역중 좌상단 점을 기준으로 시계방향 순서, 좌상->우상->우하->좌하
                        ex) [[[0,0],[1,0],[1,1],[0,1]], [[1,1],[2,1],[2,2],[1,2]], ...]
        :param image_path: 이미지 파일 경로
        :param appkey: 카카오 앱 REST API 키
        """
        API_URL = 'https://kapi.kakao.com/v1/vision/text/recognize'

        headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        jpeg_image = cv2.imencode(".jpg", image)[1]
        data = jpeg_image.tobytes()

        return requests.post(API_URL, headers=headers, files={"file": data}, data={"boxes": json.dumps(boxes)})



if __name__ == "__main__":
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_json
    app = QApplication(sys.argv)
    ex = Mainwindow()
    sys.exit(app.exec_())
