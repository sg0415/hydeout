---
layout: post
title: "모두의 딥러닝 02일차"
categories:
  - Deep learning
tags:
  - deep learning
last_modified_at: 2018-08-16
---


<3장> 가장 훌륭한 예측선 긋기 : 선형회귀
---

**선형회귀**란?
독립 변수 x를 사용해 종속 변수 y의 움직임을 예측하고 설명

	y = ax+b

정확한 직선을 그려내는 과정
최적의 a 값과 b 값을 찾아내는 작업

---
**최소제곱법**

a = (x-x~평균~)(y-y~평균~)의 합/(x-x~평균~)^2^의 합

b = y의 평균 - (x의 평균 * 기울기 a)

```python
import numpy as np

# 데이터 값 x : 공부한 시간, y : 성적
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x와 y의 평균
mx = np.mean(x)
my = np.mean(y)

# 최소제곱근 공식 중, 'x의 평균값과 x의 각 원소들의 차를 제곱하라'
divisor = sum([(mx - i)**2 for i in x])

# (x-x평균)*(y-y평균)의 합
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - mx)
    return d

dividend = top(x, mx, y, my)

#기울기 a
a = dividend / divisor

#y절편 b = mean(y) - (mean(x)*a)
b = my - (mx*a)

#출력
print("x의 평균값 : ", mx)
print("y의 평균값 : ", my)
print("분모 : ", divisor)
print("분자 : ", dividend)
print("기울기 a = ", a)
print("기울기 b = ", b)
```

실행결과
```
x의 평균값 :  5.0
y의 평균값 :  90.5
분모 :  20.0
분자 :  46.0
기울기 a =  2.3
기울기 b =  79.0
```

---
**평균 제곱근 오차**

입력 데이터가 여러 개일 경우에는 어떻게 할까?

일단 먼저 그리고 오차를 줄이며 수정해나가기
**1. 오차를 계산 할 수 있어야함**
2. 오차가 작은 쪽으로 바꾸는 알고리즘 필요

> 오차 = 실제 값 - 예측 값

|공부한시간(x) | 2 | 4 | 6 | 8 |
|--------|--------|
| 성적(실제 값, y)	|81 |93|91|97|
|예측 값|82|88|94|100|
| 오차	| 1|-5|3|3|


**평균 제곱 오차(MSE)**


>오차의 합을 구할 때는 각 오차의 값을 제곱한 뒤 더한다
>그 값을 n으로 나눠 오차 합의 평균을 구한다
> 44/4 = 11

**평균 제곱근 오차(RMSE)**
평균 제곱 오차에 제곱근을 씌워줌


**선형 회귀란 임의의 직선을 그어 이에 대한 평균 제곱근 오차를 구하고, 이 값을 가장 작게 만들어주는 a와 b를 찾아가는 작업 **


```python
import numpy as np

#기울기 a = 3, y절편 b = 76
ab=[3, 76]

#data : 공부한 시간과 성적
data = [[2, 81], [4, 93], [6, 91],[8, 97]]

#data의 첫 번째 값을 x에 저장, 두 번째 값을 y에 저장
x = [i[0] for i in data]
y = [i[1] for i in data]

#일차방정식
def predict(x):
    return ab[0]*x + ab[1]

#평균 제곱근 공식
def rmse(p,a):
    return np.sqrt(((p - a) ** 2).mean())

# rmse() 함수에 데이터를 대입하여 최종값을 구하는 함수
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

#예측값이 들어갈 빈 리스트
predict_result = []

#모든 x 값을 대입
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f" %(x[i], y[i], predict(x[i])))

#최종 RMSE 출력
print("rmse 최종값 : " + str(rmse_val(predict_result, y)))
```

오차를 최소화하는 방법은 4장부터~
