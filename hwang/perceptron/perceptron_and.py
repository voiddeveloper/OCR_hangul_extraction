import numpy as np
from perceptron import Perceptron # perceptron.py 파일의 Perceptron 클래스를 참고한다.

# X 는 AND 연산에 대한 트레이닝 데이터를 정의한 것이다.
# y 는  실제 결과값이다.
if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    # learning rate 에타값을 0.1로 설정한다.
    ppn = Perceptron(eta = 0.1)
    # fit() 멤버 함수를 호출해서 퍼셉트론 알고리즘을 구동한다.
    ppn.fit(X, y)
    # error 결과를 출력한다.
    print(ppn.errors_)