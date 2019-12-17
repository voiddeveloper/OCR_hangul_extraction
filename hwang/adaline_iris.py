import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from adaline import AdalineGD

if __name__ == '__main__':
    # iris.data 품종 데이터를 읽어오는 부분, perceptron_iris와 동일함
    df = pd.read_csv('/Users/narun/Desktop/mylib/hwang/iris.data', header = None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4))

    # learning rate = 0.01로 두고 아달라인을 수행
    # 결과값을 보면 값이 점점 커져서 J(w)값이 발산해버린다. 원하는 값을 찾을 수 없다.
    adal = AdalineGD(eta = 0.01, n_iter = 10).fit(X, y)
    ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker = 'o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SQE)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    # learning rate = 0.0001로 두고 아달라인을 수행
    # 결과값을 보면 값이 점점 작아져서 J(w)값이 0에 수렴한다. 원하는 값을 찾을 수 있다.
    adal2 = AdalineGD(eta = 0.0001, n_iter = 10).fit(X, y)
    ax[1].plot(range(1, len(adal2.cost_) + 1), np.log10(adal2.cost_), marker = 'o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(SQE)')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.subplots_adjust(left = 0.1, bottom = 0.2, right = 0.95, top = 0.9, wspace = 0.5, hspace = 0.5)
    plt.show()