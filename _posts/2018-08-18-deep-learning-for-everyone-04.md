---
layout: post
title: "모두의 딥러닝 03일차"
categories:
  - Deep learning
tags:
  - deep learning
last_modified_at: 2018-08-18
---


<5장> 참 거짓 판단 장치 : 로지스틱 회귀
---
**로지스틱 회귀**
참(1)과 거짓(0)사이를 구분하는 S자 형태의 선을 그어주는 작업

**시그모이드 함수**
`y = 1/(1+e^ax+b^)`
>a : 그래프의 경사도
>b : 그래프의 좌우 이동

시그모이드 함수에서 어떻게 오차를 구할까?

y값이 0과 1 사이이다.
실제 값이 1일때 -log h 그래프
실제 값이 0일때 -log(1-h)그래프

 -{ylogh + (1-y)log(1-h)}
y = 0일 때, y = 1일 때에 따라 다른 그래프 사용


예제
```python
import tensorflow as tf
import numpy as np

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))

#시그모이드 함수 방정식(넘파이 라이브러리)
y = 1/(1 + np.e**(a * x_data + b))

#오차를 구하는 함수
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))

#학습률과 경사 하강법
learning_rate = 0.5
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#텐서플로 구동 결과 출력
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(60001):
        sess.run(gradient_decent)
        if i % 6000 == 0:
            print("Epoch : %.f, loss = %.4f, 기울기 a = %.4f, y 절편 = %.4f" % (i, sess.run(loss), sess.run(a), sess.run(b)))


```

실행결과
```
Epoch : 0, loss = 1.2676, 기울기 a = 0.1849, y 절편 = -0.4334
Epoch : 6000, loss = 0.0152, 기울기 a = -2.9211, y 절편 = 20.2982
Epoch : 12000, loss = 0.0081, 기울기 a = -3.5637, y 절편 = 24.8010
Epoch : 18000, loss = 0.0055, 기울기 a = -3.9557, y 절편 = 27.5463
Epoch : 24000, loss = 0.0041, 기울기 a = -4.2380, y 절편 = 29.5231
Epoch : 30000, loss = 0.0033, 기울기 a = -4.4586, y 절편 = 31.0675
Epoch : 36000, loss = 0.0028, 기울기 a = -4.6396, y 절편 = 32.3346
Epoch : 42000, loss = 0.0024, 기울기 a = -4.7930, y 절편 = 33.4086
Epoch : 48000, loss = 0.0021, 기울기 a = -4.9261, y 절편 = 34.3406
Epoch : 54000, loss = 0.0019, 기울기 a = -5.0436, y 절편 = 35.1636
Epoch : 60000, loss = 0.0017, 기울기 a = -5.1489, y 절편 = 35.9005
```

변수가 더 많아진다면?

예제
```python
import tensorflow as tf
import numpy as np

#실행할 때마다 가튼 결과를 출력하기 위한 seed값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

#데이터의 값
x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)

#플레이스 홀더
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

#기울기 a와 바이어스 b의 값을 임의로 정함
#[2, 1]의 의미 : 들어오는 값은 2개, 나가는 값은 1개
a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

#y 시그모이드 함수의 방정식
y = tf.sigmoid(tf.matmul(X, a) + b)

#오차를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y)+(1-Y)*tf.log(1-y))

#학습률 값
learning_rate = 0.1

#오차를 최소로하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = -tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

#학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict = {X: x_data, Y: y_data})
        if ( i + 1 ) % 300 == 0:
            print("step = %d, a1 = %.4f, a2 = %.4f, b = %.4f, loss = %.4f"% (i + 1, a_[0], a_[1], b_, loss_))


    new_x = np.array([7, 6.]).reshape(1, 2)
    new_y = sess.run(y, feed_dict={X: new_x})

    print("공부한 시간 : %d, 과외 수업 횟수 : %d" % (new_x[:, 0], new_x[:, 1]))
    print("합격 가능성 : %6.2f %%" % (new_y * 100))

```

실행결과
```
step = 300, a1 = 0.8426, a2 = -0.5997, b = -2.3907, loss = 0.2694
step = 600, a1 = 0.8348, a2 = -0.3166, b = -3.8630, loss = 0.1932
step = 900, a1 = 0.7423, a2 = 0.0153, b = -4.9311, loss = 0.1510
step = 1200, a1 = 0.6372, a2 = 0.3245, b = -5.7765, loss = 0.1235
step = 1500, a1 = 0.5373, a2 = 0.5996, b = -6.4775, loss = 0.1042
step = 1800, a1 = 0.4471, a2 = 0.8421, b = -7.0768, loss = 0.0900
step = 2100, a1 = 0.3670, a2 = 1.0561, b = -7.6003, loss = 0.0791
step = 2400, a1 = 0.2962, a2 = 1.2458, b = -8.0652, loss = 0.0705
step = 2700, a1 = 0.2336, a2 = 1.4152, b = -8.4834, loss = 0.0636
step = 3000, a1 = 0.1779, a2 = 1.5675, b = -8.8635, loss = 0.0579
공부한 시간 : 7, 과외 수업 횟수 : 6
합격 가능성 :  85.66 %
```

입력값을 통해 출력값을 구하는 함수 y
`y = a1x1 + a2x2 + b`
>입력값 x1, x2가 각각 가중치 a1, a2를 만난다.
>b값을 더한 후 시그모이드 함수를 거쳐 1 or 0을 출력

이걸 퍼셉트론이라고 하는데... 그건 다음장에 게속
