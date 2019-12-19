import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from perceptron import Perceptron

style.use('seaborn-talk')

# krfont = {'family':'NanumGothic', 'weight':'bold', 'size':10}
# matplotlib.rc('font', **krfont)
# matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    style.use('seaborn-talk')

    # read_csv() 함수를 이용해서 파일을 읽고, pandas의 DataFrame 객체로 변환
    df = pd.read_csv('/Users/narun/Desktop/mylib/hwang/dataSet/iris.data', header = None)

    # iris 데이터를 저장한 Dataframe에서 0~99라인까지 5번째 컬럼의 데이터 값을 numpy 배열로 리턴 받아 y에 대입
    # 따라서 iris 데이터 파일에서 100개의 데이터를 추출하여 5번째 컬럼의 데이터 값을 numpy 배열로 리턴한 것이다.
    y = df.iloc[0:100, 4].values
    # y에 저장된 품종을 나타내는 문자열이 'Iris-setosa'인 경우 -1, 아니면 1로 바꾼 numpy배열을 y에 다시 대입한다.
    # 참고로 0~49 인덱스의 y값은 Iris-setosa, 50~99 인덱스의 y값은 Iris-versicolor
    y = np.where(y == 'Iris-setosa', -1, 1)
    # Iris.data를 저장한 Dataframe에서 0~99라인까지 1번쨰, 3번째 컬럼의 데이터 값을 numpy 배열로 리턴 받아 이를 X에 대입한다.
    # 참고로 Iris.data의 1번째는 꽃받침 길이, 3번쨰는 꽃잎길이
    # 즉 이 코드는 꽃받침 길이, 꽃잎길이에 따른 아이리스 품종을 머신러닝으로 학습하는 것이다.
    X = df.iloc[0:100, [0, 2]].values

    # X에 저장된 Iris.data의 꽃받침 길이, 꽃잎길이에 대한 데이터의 상관관계를 파악하는 공식이다.
    # Matplotlib의 산점도를 이용해서 화면에 표로 그린다.
    plt.scatter(X[:50, 0], X[:50, 1], color = 'r', marker = 'o', label = 'setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color = 'b', marker = 'x', label = 'versicolor')
    plt.xlabel('flower leaf length(cm)')
    plt.ylabel('flower cup length(cm)')
    plt.legend(loc = 4)
    plt.show()

    ppn1 = Perceptron(eta = 0.1)
    ppn1.fit(X, y)
    print(ppn1.errors_)