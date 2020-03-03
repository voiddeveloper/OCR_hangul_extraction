from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

np.random.seed(1234)

img_rows = 28
img_cols = 28
training_cnt = 10
batch_size = 100
num_classes = 10

# MNIST 학습용 데이터셋과 검증용 데이터셋을 로드함.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 입력층을 Keras에 맞게 reshape하는 과정.
input_shape = (img_rows, img_cols, 1)
# 각 인자는 (이미지의 개수, width, height, channel)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

X_train = x_train.astype('float32') / 255.
X_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 분류를 위해 0~9 사이의 10개 출력값을 one hot 인코딩함.
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# Deep Learning 을 위해 Sequential 로 모델을 생성함.
# 각 레이어를 add 하는 방식으로 구현.
model = Sequential()
# Convolution Layer 생성.
# 32개의 필터개수 (입력 데이터가 32가지)
# 커널 사이즈가 (3,3) 1칸씩 이동.
# padding을 하여 같은 크기의 출력을 만들고, 활설화함수는 ReLU
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
# Pooling Layer 생성. 
# 비교영역 크기는 (2,2)이고 (2,2)만큼 이동하면서 각 항목이 한번씩만 비교되도록 설정.
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 비활성화 뉴런은 30%
model.add(Dropout(0.3))

# 두번째 레이어도 동일하게 구성.
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

# 세번째 레이어
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# 네번째 레이어가 full connected 이므로 입력 데이터를 1차원으로 재정렬.
model.add(Flatten())

# fully connected 레이어를 정의 dense() 사용
# 32가지 출력, 활성화 함수로 ReLU 사용
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

# 10가지 출력
# 분류문제이므로 활성화 함수로 softmax 사용.
model.add(Dense(num_classes, activation='softmax'))
# 생성한 모델의 요약정보 출력
model.summary()
# 오차(cost = loss)는 cross entropy
# 최적화 방법은 adam
# 모델 평가 항목(mertics)으로 Accuracy(Y값을 정확히 예측한 비율)
# Learning rate는 0.001 기본값.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 수행.
# (학습용 입력데이터, 학습용 출력데이터, 1회학습데이터사이즈, 전체학습반복횟수, 과정(0:생략, 1:보기, 2:횟수와loss만 확인)
model.fit(X_train, Y_train, batch_size=batch_size,epochs=training_cnt, verbose=1)

# 모델 평가.
# 테스트용 데이터로 학습결과 시험진행.
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])