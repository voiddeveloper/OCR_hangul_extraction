from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    # dataset에 저장되어 있는 iris 관련 데이터를 불러온다.
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # trainset 과 testset 구분짓기, 전체의 25%를 testset으로 분류한다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # preprocessing 표준화
    sc = StandardScaler()

    # X_train의 평균과 표준편차를 구한다.
    sc.fit(X_train)

    # 트레이닝 데이터를 표준화
    X_train_std = sc.transform(X_train)

    # 테스트 데이터를 표준화
    X_test_std = sc.transform(X_test)

    # Perceptron 모델 호출, learning rate를 0.01로, 최대 40번 수행하는 객체 선언
    ml = Perceptron(eta0 = 0.01, max_iter = 40, random_state = 0)
    ml.fit(X_train_std, y_train)

    # 퍼셉트론으로 머신러닝 후 X_test_std를 이용하여 예측값 계산
    y_pred = ml.predict(X_test_std)
    
    print('총 테스트 개수: %d, 오류개수: %d' %(len(y_test), (y_test != y_pred).sum()))
    print('정확도: %.2f' %accuracy_score(y_test, y_pred))