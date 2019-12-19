import numpy as np

class AdalineGD():
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = [] # J(w)가 어떤 값으로 수렴하는지 확인하기 위해 매 반복마다 계산되는 비용 함수값을 저장하는 용도

        # 아달라인에서 가중치를 업데이트하는 공식
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            # X.T는 행렬X의 전치행렬
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)