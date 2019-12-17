import tensorflow as tf
import random
import datetime
from tensorflow.examples.tutorials.mnist import input_data

# 학습에 필요한 실제 데이터 로드, 결과값을 one_hot 인코딩 형태로 가져온다.
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

# 재실행시에도 동일한 초기값을 갖도록 랜덤 시드를 지정
tf.reset_default_graph()
tf.set_random_seed(777)

# 학습에 필요한 설정값
# 가중치가 발산하지 않도록 조절하는 값
learning_rate = 0.001

# 전체 데이터셋 반복 학습 횟수
training_cnt = 10

# 한번에 학습할 데이터 수 (메모리, 학습시간 tradeoff)
batch_size = 100

# 텐서플로우에 입력데이터를 전달할 변수 정의
# drop out을 전달할 변수 정의
keep_prob = tf.placeholder(tf.float32) 

# X는 784가지의 입력값을 갖음
X = tf.placeholder(tf.float32, [None, 784])

# X를 이미지화 하기위한 변수 정의
X_img = tf.reshape(X, [-1, 28, 28, 1])

# Y는 10가지 출력값을 갖음
Y = tf.placeholder(tf.float32, [None, 10])

# 학습 모델 정의: 3개의 컨볼루션 레이어와 2개의 풀 컨넥티드 레이어로 정의한다.
# 가중치: (3,3,1) 필터를 32채널 사용, 따라서 b도 32채널로 정의
# 초기값을 랜덤하게 생성, 정규분포의 표준편차를 사용(stddev)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev= 0.01))

# 편향: 가중치와 동일하게 32채널로 정의
b1 = tf.Variable(tf.random_normal([32], stddev= 0.01))

# 컨볼루션 연산
# X_img에 W1을 컨볼루션
# stride는 각 차원별로 1씩 이동 [batch, height, width, channel]
# padding은 길게(원본 사이즈와 동일한 shape를 출력)
L1 = tf.nn.conv2d(X_img, W1, strides= [1, 1, 1, 1] , padding= 'SAME') + b1

# 활성화 함수로 ReLU 사용
L1 = tf.nn.relu(L1)

# 풀링 레이어는 MaxPool 사용
# ksize: 커널 사이즈. 즉, 원본 데이터에서 추출할 영역 크기 [batch, height, width, channel]
# strides: 원본 데이터에서 차례마다 이동할 칸 수 [batch, height, width, channel]
# (14, 14, 32)의 output 발생
L1 = tf.nn.max_pool(L1, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')

# 학습용 데이터에 과잉학습이 되지 않도록 일부 뉴런을 drop out
L1 = tf.nn.dropout(L1, keep_prob= keep_prob)

# (3, 3, 32)의 필터를 32채널 사용
# 왜 (*, *, 32)이냐면 max pool 결과가 (*, *, 32)이기 때문이다.
W2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev= 0.01))
b2 = tf.Variable(tf.random_normal([32], stddev= 0.01))

L2 = tf.nn.conv2d(L1, W2, strides= [1, 1, 1, 1], padding= 'SAME') + b2
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')
L2 = tf.nn.dropout(L2, keep_prob= keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))

b3 = tf.Variable(tf.random_normal([32], stddev= 0.01))

L3 = tf.nn.conv2d(L2, W3, strides= [1, 1, 1, 1], padding= 'SAME') + b3
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')
L3 = tf.nn.dropout(L3, keep_prob= keep_prob)

# 다음 레이어가 Fully Connected Layer이기 때문에, L3의 output을 1차원 배열로 변경
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 32])

# 기존 DNN 모델 사용(FC)
# 입력 데이터는 전단의 출려과 동일한 4 * 4 * 32 종류이고, 출력은 32개로 설정
# 가중치 초기화에 Xavier 사용
W4 = tf.get_variable("W4", shape=[4 * 4 * 32, 32], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([32]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# 입력은 32 종류이고, 출력은 10종류 (0~9)
W5 = tf.get_variable("W5", shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

# 분류 문제이므로 softmax cross entropy 함수 사용
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

# Adam 최적화 함수 사용
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 텐서플로우 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

start = datetime.datetime.now();
print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'), '학습을 시작')

#반복 학습
for epoch in range(training_cnt):
    avg_cost = 0
    # 학습 데이터가 크므로 전체 데이터를 분할하여 학습한다.
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        # 1회에 학습할 데이터 추출 (자동으로 다음 데이터로 이어서 가져옴)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # 텐서 플로우에 변수에 학습 데이터 연결
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}

        # cost, optimizer 함수 전달, 입력데이터 전달
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch: {:04d}, cost = {:.9f}'.format(epoch + 1,avg_cost))

print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'), '학습완료!')
print('총학습시간 : ', datetime.datetime.now() - start )

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict= {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

r = random.randint(0, mnist.test.num_examples - 1)
print("Label:      ", sess.run(tf.argmax(mnist.test.labels[r:r + 10], 1)))
print("Prediction: ", sess.run( tf.argmax(logits, 1), feed_dict= {X: mnist.test.images[r:r + 10], keep_prob: 1}))









