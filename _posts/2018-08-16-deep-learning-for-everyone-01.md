---
layout: post
title: "모두의 딥러닝 01일차"
categories:
  - Deep learning
tags:
  - deep learning
last_modified_at: 2018-08-16
---

<1장> 최고급 요리를 먹을 시간

인공지능 > 머신러닝 > 딥러닝

* <b>인공지능</b>이 먹을 수 있는 모든 음식이라면
* <b>머신러닝</b>은 영양가 많은 고기 음식
* <b>딥러닝</b>은 최고급 스테이크 !
  
---

<b>딥러닝 환경 구축</b>

환경 : 64bit Windows

아나콘다

http://www.continuum.io/downloads
Anaconda 5.2 설치

Anaconda Prompt 실행

```
conda create -n tutorial python=3.6 numpy scipy matplotlib spyder pandas seaborn scikit-learn h5py
```
tutorial 은 작업환경, numpy 이하는 본 교재에서 사용되는 모든 파이썬 라이브러리의 이름


텐서플로 설치
```pip install tensorflow```

케라스 설치
```pip install keras```
 
---

파이참

https://www.jetbrains.com/pycharm/download/

Community 버전 설치!
대학교 인증 받으면 Professional 버전도 설치가 가능했던 거 같지만...? 일단은 Community 버전을 설치하기로

Create New Project, 폴더명은 deeplearning

Interpreter를 기존에 설치한 작업 환경으로 바꿔준다
Anaconda3 > envs > tutorial > python.exe

deeplearning 폴더 안에 예제 소스 복사해 넣기
나는 깃허브로 받았당
```git clone https://github.com/gilbutITbook/006958```


---
2장 처음 해 보는 딥러닝

* <b>프로젝트</b>가 여행이라면
* <b>텐서플로</b>는 목적지까지 빠르게 이동시켜주는 비행기
* <b>케라스</b>는 비행기의 이류규 및 정확한 지점까지의 도착을 책임지는 파일럿
  
  
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
  
