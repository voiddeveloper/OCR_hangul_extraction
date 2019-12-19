import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

# 퍼셉트론 분류기를 이용해 머신러닝을 수행하고, 그 결과를 좌표계에서 어떻게 영역을 구분하는지 보여주는 예제
######################################################################################################
# 아이리스 트레이닝 데이터 X는 (꽃잎 길이, 꽃잎 너비) 2개의 값으로 이루어져 있다.
# 따라서 x축의 값으로 꽃잎 길이, y축의 값으로 꽃잎 너비를 설정한 좌표계를 설정한다.
# 이제 dataset에 있는 각각의 data를 좌표계에 적용하고, 해당 좌표의 값을 분류해준다. (총 3종류의 아이리스가 있기 때문에, [0,1,2] 3종류로 분류한다.

def plot_decion_region(X, y, classifier, test_idx = None, resolution = 0.02, title = ''):
    # 좌표에 표시할 마커 종류
    markers = ('s', 'x', 'o', '^', 'v')
    # 각 마커마다 표시할 색깔 종류
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    # 중복 값을 삭제하기 위해 numpy.unique()를 적용한 colormap 객체 생성
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # decision surface 그리기
    # 트레이닝 데이터 X의 첫번째 값인 꽃잎 길이의 최소값 -1 = x1_min, 최대값 +1 = x1_max
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    # x1_min, x1_max와 같은 방법으로 x2_min, x2_max 값을 구한다.
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # numpy.meshgrid()는 격자의 교차점 좌표 값을 리턴하는 함수
    # resolution 간격으로 쥬ㅘ표 격자 교차점을 표시한다.
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()을 이용해서 1차원 배열로 만든다.
    # 이 값을 이용해서 퍼셉트론 분류기 predict()에 넣고, 예측값 Z를 구한다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    
    # Z값을 구했으면 reshape()를 이용해서 원래 배열 모양으로 바꾼다.
    Z = Z.reshape(xx.shape)

    # 아래 코드는 결과값을 그래프에 그리는 코드
    plt.contourf(xx, yy, Z, alpha = 0.5, cmap = cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], c = cmap(idx), marker = markers[idx], label = cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c = '', linewidths = 1, marker = 'o', s = 80, label = 'testset')

    plt.xlabel('standardized flower length')
    plt.ylabel('standardized flower width')
    plt.legend(loc = 2)
    plt.title(title)
    plt.show()