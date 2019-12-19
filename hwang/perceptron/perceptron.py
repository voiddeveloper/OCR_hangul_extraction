import numpy as np

class Perceptron(): # 클래스 선언
        # 생성자 선언, thresholds = 임계값, eta = learning rate, n_iter = 학습 횟수
        def __init__(self, thresholds = 0.0, eta = 0.01, n_iter = 10): # 디폴트값 선언
            self.thresholds = thresholds
            self.eta = eta
            self.n_iter = n_iter

        # 트레이닝 데이터 X, 실제 결과값 y
        def fit(self, X, y): # 클래스 선언
            # 가중치를 numpy 배열로 정의, X.shape[1]은 트레이닝 데이터의 입력값 개수를 의미함
            self.w_ = np.zeros(1 + X.shape[1])
            # 머신러닝 반복 회수에 따라 퍼셉트론의 예측값과 실제 결과값이 다른 오류 회수를 저장하기 위한 변수
            self.errors_ = []

            # n_iter 값만큼 for문을 반복한다.
            for _ in range(self.n_iter):
                errors = 0 # 초기 오류의 횟수를 0으로 선언
                # 트레이닝 데이터 세트 X와 결과값 y를 하나씩 꺼내서 xi, target 변수에 대입한다.
                for xi, target in zip(X, y):
                    # update 값은 n(y-y^)이다.
                    # 실제 결과값과 예측값에 대한 활성 함수 리턴값이 같게 되면 update는 0이 된다.
                    # 따라서 트레이닝 데이터 xi값에 곱해지는 가중치에 update * xi 값을 더함으로써 가중치를 업데이트 할 수 있다.
                    update = self.eta * (target - self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0) # update 값이 0이 아니면 eroors 값을 1 증가 시킨다.
                # 모든 트레이닝 데이터에 대해 1회 학습이 끝나면 self.errors_에 발생한 오류 횟수를 추가한다.
                # 가중치 self.w_를 화면에 출력하고 다시 이 과정을 반복한다. (n_iter 횟수 만큼)
                self.errors_.append(errors)
                print(self.w_)

            return self
            
        # numpy.dot(x,y)는 백터 x, y의 내적 또는 행렬 x, y의 곱을 리턴한다.
        # 트레이닝 데이터 X의 각 입력값과 그에 따른 가중치를 곱한 총합, 즉 순입력 함수 결과값을 retrun한다.
        def net_input(self, X):
            return np.dot(X, self.w_[1:]) + self.w_[0]

        # 순입력 함수 결과값이 임계값인 self.thresholds보다 크면 1, 아니면 -1을 리턴한다.
        def predict(self, X):
            return np.where(self.net_input(X) > self.thresholds, 1, -1)