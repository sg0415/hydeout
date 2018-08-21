---
layout: post
title: "모두의 딥러닝 06일차"
categories:
  - Deep learning
tags:
  - deep learning
last_modified_at: 2018-08-21
---


<10장> 모델 설계하기
---

분석할 코드 

```python
# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옵니다.
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옵니다.
import numpy
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분입니다.
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러들입니다.
Data_set = numpy.loadtxt("../dataset/ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장합니다.
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정합니다(모델을 설정하고 실행하는 부분입니다).
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝을 실행합니다.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=30, batch_size=10)

# 결과를 출력합니다.
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

```

<b>입력층, 은닉층, 출력층</b>

```python
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```
`Sequential()` : 퍼셉트론 위에 퍼셉트론 쌓기
`model.add()` : 새로운 층 추가


출력층을 제외한 나머지는 모두 '은닉층'
`model.add(Dense(30, input_dim=17, activation='relu'))`
>30 : 30개의 노드를 만든다
>input_dim : 입력 데이터로부터 몇 개의 값이 들어올지 정함
>keras는 첫번째 은닉층이 은닉층 + 입력층 역할
>활성화 함수로 relu 선택
>
> <i>데이터에서 17개의 값을 받아 은닉층의 30개의 노드로 보낸다</i>

맨 마지막 층은 결과를 출력하는 '출력층'
`model.add(Dense(1, activation='sigmoid'))`
> 출력값을 하나로 정해야하므로 노드 수는 1개
> 활성화 함수는 sigmoid

<b>모델 컴파일</b>
```python
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
```

>loss : 평균 제곱 오차 함수(mean_squared_error) 사용
>optimizer : 최적화를 위해 adam 사용
>metrics : 모델이 컴파일될 때 모델 수행 결가를 나타나게끔 설정
>- 과적합문제(over fitting, 13장에서 배움) 방지

<b>모델 실행</b>
```python
model.fit(X, Y, epochs=30, batch_size=10)
```
모델 실행하기
>epochs = 30 : 샘플이 처음부터 끝까지 30번 재사용될 수 있도록 실행 반복
>batch_size : 샘플을 한 번에 몇 개씩 처리할지(적당한 값 골라야함)


<b>교차 엔트로피</b>
오차 함수. 출력 값에 로그를 취해 오차가 커지면 수렴 속도가 빨라지고 오차가 작아지면 속도 감소
분류 문제에서 주로 사용됨
예측 값이 참, 거짓 둘 중 하나일 때 `binary_crossentropy` 사용

오차 함수에는 여러 종류가 있다

<i>평균 제곱 계열</i>
mean_squared_error - 평균 제곱 오차
mean_absolute_error - 평균 절대 오차
 - 실제 값과 예측 값 차이의 절댓값 평균

mean_absolute_percentage_error - 평균 절대 백분율 오차
- 절댓값 오차를 절댓값으로 나눈 후 평균

mean_squared_logarithmic_error - 평균 제곱 로그 오차
- 실제 값과 예측 값에 로그를 적용한 값의 차이를 제곱한 값의 평균

<i>교차 엔트로피 계열</i>
categorical_crossentropy - 범주형 교차 엔트로피(일반적인 분류)
binary_crossentropy - 이항 교차 엔트로피(두 개의 클래스 중에서 예측)

<11장> 데이터 다루기
---
딥러닝을 공부할 때에는 좋은 데이터를 먼저 구해놓고 이를 통해 연습하는 것이 중요
UCI 머신러닝 저장소(UCI Machine Learning Repository)
- http://archive.ics.uci.edu

위 사이트에서 해당 데이터셋을 내려받을 수 있다고 책에 쓰여있는데...
접속했더니 퍼미션이 만료되었다고 한다 -.- 이전에 내려받은 파일로 실습 진행

pima-indians-diabetes.csv
샘플 768개 속성 1개 클래스 1개

속성과 클래스를 먼저 구분해야하고, 정확도 향상을 위해 데이터의 추가 및 재가공이 필요할 수도 있음! 따라서 <u>데이터의 내용과 구조를 잘 파악하는 것</u>이 중요함



<b>panda를 활용한 데이터 조사</b>
데이터를 잘 파악하기 위해 <b>데이터 시각화</b>하기

데이터를 불러와 그래프로 표현해보기

```python
import pandas as pd
df = pd.read_csv('../dataset/pima-indians-diabetes.csv', names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])
```
csv(comma separated values file) : 콤마()로 구분된 데이터들의 모음
보통 csv파일에는 헤더가 존재하지만 이 파일에는 헤더가 없으므로
`names` 함수를 통해 키워드를 지정해줌

<br>
```python
print(df.head(5))
```
`head()` : 데이터 첫줄부터 불러오기

실행결과
```
   pregnant  plasma  pressure  thickness  ...     BMI  pedigree  age  class
0         6     148        72         35  ...    33.6     0.627   50      1
1         1      85        66         29  ...    26.6     0.351   31      0
2         8     183        64          0  ...    23.3     0.672   32      1
3         1      89        66         23  ...    28.1     0.167   21      0
4         0     137        40         35  ...    43.1     2.288   33      1

[5 rows x 9 columns]
```
<br>
```python
print(df.info())
```

실행결과
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
pregnant     768 non-null int64
plasma       768 non-null int64
pressure     768 non-null int64
thickness    768 non-null int64
insulin      768 non-null int64
BMI          768 non-null float64
pedigree     768 non-null float64
age          768 non-null int64
class        768 non-null int64
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
None
```
<br>
```python
print(df.describe())
```
실행결과

```
         pregnant      plasma     ...             age       class
count  768.000000  768.000000     ...      768.000000  768.000000
mean     3.845052  120.894531     ...       33.240885    0.348958
std      3.369578   31.972618     ...       11.760232    0.476951
min      0.000000    0.000000     ...       21.000000    0.000000
25%      1.000000   99.000000     ...       24.000000    0.000000
50%      3.000000  117.000000     ...       29.000000    0.000000
75%      6.000000  140.250000     ...       41.000000    1.000000
max     17.000000  199.000000     ...       81.000000    1.000000
```

<br>
```python
print(df[['pregnant', 'class']])
```
일부 컬럼만 보고 싶을 경우(pregnant, class)

실행결과
```
     pregnant  class
0           6      1
1           1      0
2           8      1
3           1      0
4           0      1
5           5      0
6           3      1
7          10      0
8           2      1
9           8      1
10          4      0
11         10      1
12         10      0
13          1      1
14          5      1
15          7      1
16          0      1
17          7      1
18          1      0
19          1      1
20          3      0
21          8      0
22          7      1
23          9      1
24         11      1
25         10      1
26          7      1
27          1      0
28         13      0
29          5      0
..        ...    ...
738         2      0
739         1      1
740        11      1
741         3      0
742         1      0
743         9      1
744        13      0
745        12      0
746         1      1
747         1      0
748         3      1
749         6      1
750         4      1
751         1      0
752         3      0
753         0      1
754         8      1
755         1      1
756         7      0
757         0      1
758         1      0
759         6      1
760         2      0
761         9      1
762         9      0
763        10      0
764         2      0
765         5      0
766         1      1
767         1      0

[768 rows x 2 columns]
```


임신 횟수와 당뇨병 발병 확률 계산
```python
print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))
```
`groupby` : 'pregnant' 정보를 기준으로 하는 새 그룹 생성
`as_index=False` : pregnant 정보 옆에 새로운 index 생성
`mean().sort_values(by='pregnant', ascending=True)` : mean 함수로 평균을 구하고 sort_values 함수로 pregnant 칼럼을오름차순(ascending) 정렬

실행결과
```
    pregnant     class
0          0  0.342342
1          1  0.214815
2          2  0.184466
3          3  0.360000
4          4  0.338235
5          5  0.368421
6          6  0.320000
7          7  0.555556
8          8  0.578947
9          9  0.642857
10        10  0.416667
11        11  0.636364
12        12  0.444444
13        13  0.500000
14        14  1.000000
15        15  1.000000
16        17  1.000000
```

<b>matplotlib</b>를 이용해 그래프 그리기
`matplotlib`는 파이썬에서 그래프를 그릴 때 가장 많이 사용됨!
`seaborn`은 `matplotlib`를 기반으로 더 정교한 그래프를 그리게 도와줌

```python
#그래프의 크기 결정
plt.figure(figsize=(12,12))

#heatmap : 각 항목 간의 상관관계를 나타내주는 함수
#           두 항목씩 짝 지은 뒤 각각 어떤 패턴으로 변화하는지 관찰
#           두 항목이 전혀 다른 패턴으로 변화하면 0, 비슷한 패턴으로 변할수록 1에 가까운 값 출력
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)

plt.show()
```

![image](https://github.com/sg0415/sg0415.github.io/blob/master/_images/deep06.png?raw=true)

plasma와 class의 상관관계가 가장 높다

plasma와 class의 관계
```python
grid = sns.FacetGrid(df, col='class')
grid.map(plt.hist, 'plasma', bins=10)
plt.show()```

![image](https://github.com/sg0415/sg0415.github.io/blob/master/_images/deep06_2.png?raw=true)

당뇨병 환자(class=1)일 때 plasma 항목의 수치가 150이상인 경우가 많다
<u>결과에 미치는 영향이 큰 항목을 발견하는 것이 데이터 전처리 과정의 한 예</u>

seed 값 설정
일정한 결과 값을 얻기 위해 넘ㄴ파이 seed값과 텐서플로 seed 값을 모두 설정해야함

예제
```python
from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf


#seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

#모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

#결과 출력
print("\n Accuracy: %.4f" %(model.evaluate(X, Y)[1]))
```

실행결과
```
Epoch 1/200

 10/768 [..............................] - ETA: 16s - loss: 6.0117 - acc: 0.4000
740/768 [===========================>..] - ETA: 0s - loss: 2.4714 - acc: 0.5149 
768/768 [==============================] - 0s 354us/step - loss: 2.4329 - acc: 0.5143
Epoch 2/200

 10/768 [..............................] - ETA: 0s - loss: 0.9666 - acc: 0.4000
768/768 [==============================] - 0s 65us/step - loss: 0.9142 - acc: 0.6393

(중략)
Epoch 200/200

 10/768 [..............................] - ETA: 0s - loss: 0.3972 - acc: 0.8000
670/768 [=========================>....] - ETA: 0s - loss: 0.4598 - acc: 0.7761
768/768 [==============================] - 0s 76us/step - loss: 0.4636 - acc: 0.7721

 32/768 [>.............................] - ETA: 0s
768/768 [==============================] - 0s 50us/step

 Accuracy: 0.7839
 
ㅇㄹㄴㅇㄹㄴ
ㄹ
ㄴㄹ
ㅇㄴ
ㄹ