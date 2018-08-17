---
layout: post
title: "모두의 딥러닝 03일차"
categories:
  - Deep learning
tags:
  - deep learning
last_modified_at: 2018-08-17
---

모두의 딥러닝 03일차

<4장> 가장 훌륭한 예측선 긋기 : 선형회귀
---

**미분**
>a에서의 순간 변화율 = 기울기

기울기가 0인 지점을 찾기
>1.a1에서 미분을 구한다.
>2.구해진 기울기의 반대방향으로 어느정도 이동시킨 ax에서 미분을 구한다.
>3.a3에서 미분을 구한다.
>3의 값이 0이 아니면 2~3번 반복

**경사하강법** : 반복적으로 기울기 a를 변화시켜서 m의 값을 찾아내는 방법

---
**학습률**
기울기를 이동시킬 때 적당한 만큼 이동해야함
이때 이동 거리를 정해주는 것이 학습률


++경사하강법은 오차의 변화에 따라 이차 함수 그래프를 만들고 적절한 학습률을 설정해 미분 값이 0인 지점을 구하는 것!
++

**경사 하강법 실습**
```python
import tensorflow as tf

#데이터
data = [[2, 81], [4, 93], [6, 91], [8,97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

#학습률
learning_rate = 0.1

#임의의 기울기 a(1~10)와 y절편 b (1~100)
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype = tf.float64, seed = 0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype = tf.float64, seed = 0))

#일차방정식 y = ax + b
y = a * x_data + b

#평균 제곱근 오차(텐서플로 RMSE함수)
rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data )))

#경사 하강법(텐서플로 라이브러리)
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

#텐서플로를 이용한 학습
with tf.Session() as sess:
    #변수 초기화
    sess.run(tf.global_variables_initializer())
    #2001번 실행(0번째 포함하므로)
    for step in range(2001):
        sess.run(gradient_decent)
        #100번마다 결과 출력
        if step %100 == 0:
            print("Epch : %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b = %.4f" %(step, sess.run(rmse), sess.run(a), sess.run(b)))
```

---
**다중선형회귀**란?

입력값이 한 개가 아닐 경우( 기울기가 여러 개일 때)

|공부한시간(x1) | 2 | 4 | 6 | 8 |
|--------|--------|
|과외수업횟수(x2)|0|4|2|3|
| 성적(실제 값, y)	|81 |93|91|97|

	y= a1x1 + a2x2 + b
   


**다중선형회귀 실습**
```python
import tensorflow as tf

# x1, x2, y의 데이터값

data =[[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]         #새로 추가된 입력값
y_data = [y_row[2] for y_row in data]

#기울기
a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

#새로운 방정식
y = a1 * x1 + a2 * x2 + b

#학습률
learning_rate = 0.1

#평균 제곱근 오차(텐서플로 RMSE함수)
rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_data )))

#경사 하강법(텐서플로 라이브러리)
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

#텐서플로를 이용한 학습
with tf.Session() as sess:
    #변수 초기화
    sess.run(tf.global_variables_initializer())
    #2001번 실행(0번째 포함하므로)
    for step in range(2001):
        sess.run(gradient_decent)
        #100번마다 결과 출력
        if step %100 == 0:
            print("Epch : %.f, RMSE = %.04f, 기울기 a1 = %.4f, 기울기 a2 = %.4f, y 절편 b = %.4f"
                  %(step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))
```

<br>
다중 선형 회귀 그래프
![image](https://github.com/sg0415/sg0415.github.io/blob/master/_images/Figure_1.png?raw=true)

단순 선형 회귀에서 1차원 직선이었던 그래프가 3차원 평면으로 바뀜!