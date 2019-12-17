from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import style
from plotdregion import plot_decion_region

style.use('seaborn-talk')

if __name__ == '__main__':
    # dataset에 저장되어 있는 iris 관련 데이터를 불러온다.
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # trainset 과 testset 구분짓기, 전체의 30%를 testset으로 분류한다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # preprocessing 표준화
    sc = StandardScaler()

    # X_train의 평균과 표준편차를 구한다.
    sc.fit(X_train)

    # 트레이닝 데이터를 표준화
    X_train_std = sc.transform(X_train)

    # 테스트 데이터를 표준화
    X_test_std = sc.transform(X_test)

    # LogisticRegression 모델 호출
    ml = LogisticRegression(C = 1000.0, random_state = 0)
    ml.fit(X_train_std, y_train)

    # 퍼셉트론으로 머신러닝 후 X_test_std를 이용하여 예측값 계산
    y_pred = ml.predict(X_test_std)
    
    print('총 테스트 개수: %d, 오류개수: %d' %(len(y_test), (y_test != y_pred).sum()))
    print('정확도: %.2f' %accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decion_region(X = X_combined_std, y = y_combined, classifier = ml, test_idx = range(105, 150), title = 'scikit-learn Perceptron')