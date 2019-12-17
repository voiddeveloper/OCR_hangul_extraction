import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from adaline import AdalineGD

if __name__ == '__main__':
    # iris.data 품종 데이터를 읽어오는 부분, perceptron_iris와 동일함
    df = pd.read_csv('/Users/narun/Desktop/mylib/iris.data', header = None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    X_std = np.copy(X) # X의 값을 복사해서 X_std에 저장
    # Iris.data에서 꽃받침 길이와 꽃잎 길이의 표준화한 값을 X_std에 할당하는 부분
    # numpy의 mean()은 numpy 배열에 있는 값들의 평균을 구한다.
    # numpy의 std()는 numpy 배열에 있는 값들의 표준편차를 구한다.
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    # learning rate를 0.01로, 반복회수를 15로 두고, X_std를 아달라인으로 머신러닝을 수행한다.
    adal = AdalineGD(eta = 0.01, n_iter = 15).fit(X_std, y)
    plt.plot(range(1, len(adal.cost_) + 1), adal.cost_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('SSE')
    plt.title('Adaline Standardized - Learning rate 0.01')
    plt.show()