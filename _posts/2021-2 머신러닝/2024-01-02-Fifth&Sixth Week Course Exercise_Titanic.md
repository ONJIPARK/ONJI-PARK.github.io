---
layout: single
title:  "Fifth&Sixth Week Course Exercise"
categories: coding
tag: [python, blog, jupyter]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# 60191315 박온지 5주차 과제

## Titanic 승객 데이터로 의사결정나무 만들기

- 문제정의: 타이타닉 생존자를 예측하는 의사결정나무 모델을 만들어보자


### 1. 데이터셋 로딩

- 데이터는 캐글 사이트에 다운로드 가능(https://www.kaggle.com/c/titanic/data?select=train.csv)

- train.csv 파일을 titanic.csv 변환하여 사용

- 현재 폴더에 data라는 이름의 하위 폴더를 만든 뒤 titanic.csv를 저장



```python
#패키지 불러오기
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
```


```python
#데이터프레임으로 데이터 읽어오기
df=pd.read_csv("data/titanic.csv", index_col=["PassengerId"])
print(df.shape)
df.head()
```

<pre>
(891, 11)
</pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


### 변수 설명

- Survived: 생존 여부

- Pclass: 객실 등급

- Name: 이름

- Sex: 성별

- Age: 나이

- SibSp: 함께 탑승한 형제자매와 배우자수

- Parch: 함께 탑승한 부모의 자녀수

- Ticket: 티켓 번호

- Fare: 운임

- Cabin: 선실 번호

- Embarked: 탑승 항구(C= Cherbourg, Q=Queenstown, S=Southampton)


## 2. 데이터 전처리

- 필요없는 열 제거

- 변수값 변환

- 독립변수와 종속변수 구분



```python
#필요없는 컬럼 제거-> 생존여부와 연관없는
df=df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
```


```python
#제거 확인
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>888</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>891</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 7 columns</p>
</div>



```python
#전처리: 진위형 변수로 변경 -> 성별은 문자보다 숫자로 다루는 게 나으니까
df["Sex"] = df.Sex.map({"female":0, "male":1})
```


```python
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>888</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>891</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 7 columns</p>
</div>



```python
#결측치 확인
df.isnull().sum()
```

<pre>
Survived      0
Pclass        0
Sex           0
Age         177
SibSp         0
Parch         0
Fare          0
dtype: int64
</pre>

```python
#중간값으로 결측치 채워주기
df.Age.fillna(df.Age.median(), inplace=True)
```


```python
#결측치이던 889번 채워진 거 보이지
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
    </tr>
    <tr>
      <th>888</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>28.0</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
    </tr>
    <tr>
      <th>891</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 7 columns</p>
</div>



```python
#Input 변수와 Output 변수 구분(생존 변수만 y 변수)
X=np.array(df.iloc[:, 1:]) # 모든 행에 대하여 1번 열 데이터 전까지 가져오기
y=np.array(df['Survived'])
```


```python
X
```

<pre>
array([[ 3.    ,  1.    , 22.    ,  1.    ,  0.    ,  7.25  ],
       [ 1.    ,  0.    , 38.    ,  1.    ,  0.    , 71.2833],
       [ 3.    ,  0.    , 26.    ,  0.    ,  0.    ,  7.925 ],
       ...,
       [ 3.    ,  0.    , 28.    ,  1.    ,  2.    , 23.45  ],
       [ 1.    ,  1.    , 26.    ,  0.    ,  0.    , 30.    ],
       [ 3.    ,  1.    , 32.    ,  0.    ,  0.    ,  7.75  ]])
</pre>

```python
y
```

<pre>
array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
       1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
       1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
       1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
       0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1,
       1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
       0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
       0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
       1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0,
       0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1,
       1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
       0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0,
       1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
       0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1,
       1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], dtype=int64)
</pre>
### 3. 모델 적합



```python
#트레이닝 셋트와 테스트 셋트로 데이터 구분(여기서는 7:3으로 구분)
from sklearn.model_selection import train_test_split

#random_state는 반복적으로 같은 결과를 내기 위해서 설정 무작위 설정
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)

print("X_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)
print("X_test 크기:", X_test.shape)
print("y_test 크기:", y_test.shape)
```

<pre>
X_train 크기: (623, 6)
y_train 크기: (623,)
X_test 크기: (268, 6)
y_test 크기: (268,)
</pre>

```python
#의사결정나무모델에 데이터 적합(fitting) 피팅은 학습시키는 것 학습시키는 모델이 의사결정나무모델
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(random_state=0, max_depth=3)
tree.fit(X_train, y_train)
```

<pre>
DecisionTreeClassifier(max_depth=3, random_state=0)
</pre>
### 4. 모델 성능평가



```python
temp_y_pred=tree.predict(X_test)
#Training값을 X_test에서 넣음
print('예측값\n', temp_y_pred)
print('실제값\n', y_test)
```

<pre>
예측값
 [0 0 0 1 1 0 1 1 0 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 1 1 0 1 1 0 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 1 0 0 0
 0 1 0 0 0 0 0 1 1 0 0 1 1 1 1 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 1 1 1 0 0 1 0
 1 0 1 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 1 1 1 0 1
 1 0 0 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 0 0 0 0 0
 0 1 0 0 1 1 0 1 1 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1
 1 0 1 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 0 1 0 0 1 0 1 0 0 1 0 0 0 0 0 1 1 0 0
 0 0 0 0 0 0 0 1 0]
실제값
 [0 0 0 1 1 1 1 1 1 1 0 1 0 1 1 0 0 0 0 1 0 1 0 0 0 1 0 1 1 0 0 1 0 1 0 1 0
 0 0 0 1 0 0 0 1 0 0 1 0 0 1 1 1 0 1 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 1 0 0
 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 0 0 1 1 0 0 1 0
 1 1 0 1 1 1 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1
 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0
 0 1 0 0 1 0 0 1 1 0 0 0 1 1 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1
 1 0 1 0 0 1 1 0 0 1 1 0 0 0 1 1 1 0 0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 0 1 1 1
 0 0 0 0 0 0 0 1 0]
</pre>

```python
#정확도 계산
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

temp_acc=accuracy_score(y_test, temp_y_pred)
#y_test와 temp_y_pred로 예측한 값을 적용하여 temp_acc로 추출함.

print('정확도:', format(temp_acc))
```

<pre>
정확도: 0.8208955223880597
</pre>

```python
#오차행렬(Confusion Matrix)
print(confusion_matrix(y_test, temp_y_pred))
```

<pre>
[[146  22]
 [ 26  74]]
</pre>

```python
from sklearn.metrics import precision_score, recall_score, f1_score
 # 정확도, 정밀도, 재현율, f1-score
print('accuracy: ', accuracy_score(y_test, temp_y_pred))  
print('precision: ', precision_score(y_test, temp_y_pred))
print('recall: ', recall_score(y_test, temp_y_pred))
print('f1: ', f1_score(y_test, temp_y_pred))
```

<pre>
accuracy:  0.8208955223880597
precision:  0.7708333333333334
recall:  0.74
f1:  0.7551020408163266
</pre>

```python
#모델 성능을 보여주는 classification_report  
print(classification_report(y_test, (tree.predict(X_test))))

# 이렇게 해도 됨  > 0.5는 X_test를 정수형으로 보이게 하려는 것.?
# print(classification_report(y_test, (tree.predict(X_test) > 0.5).astype("int16")))
```

<pre>
              precision    recall  f1-score   support

           0       0.85      0.87      0.86       168
           1       0.77      0.74      0.76       100

    accuracy                           0.82       268
   macro avg       0.81      0.80      0.81       268
weighted avg       0.82      0.82      0.82       268

</pre>
# 6주차 과적합과 가지치기

- 60191315 박온지

- 의사결정나무의 깊이별 정확도 비교와 최적의 의사결정나무 깊이 구하기



```python
train_scores, test_scores = list(), list()
#Train_score와 test_score는 각각 훈련데이터와 테스트 데이터의 정확도를 저장
#Train_score는 훈련데이터에 특화될수록 증가
#Test_score는 테스트데이터에 일반화될수록 증가
```


```python
#의사결정나무 모델의 depth 조절(1부터 19까지)
for i in range(1, 20):
    DT1 = DecisionTreeClassifier(max_depth=i)
    #깊이를 증가시키면서 의사결정나무 모델 적합
    DT1.fit(X_train, y_train)
    
    #training dataset 정확도 평가
    train_att = DT1.predict(X_train)
    train_acc = accuracy_score(y_train, train_att)
    train_scores.append(train_acc)
    #5주차에는 테스트 데이터에 대해서는 정확도를 평가했었지만
    
    #test dataset 정확도 평가
    test_att = DT1.predict(X_test)
    test_acc = accuracy_score(y_test, test_att)
    test_scores.append(test_acc)
    
    #Train 정확도와 Test 정확도를 depth 개수에 따른 정확도 확인
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    
    
```

<pre>
>1, train: 0.787, test: 0.787
>2, train: 0.791, test: 0.787
>3, train: 0.836, test: 0.821
>4, train: 0.844, test: 0.821
>5, train: 0.859, test: 0.813
>6, train: 0.872, test: 0.795
>7, train: 0.881, test: 0.802
>8, train: 0.905, test: 0.813
>9, train: 0.925, test: 0.817
>10, train: 0.941, test: 0.813
>11, train: 0.957, test: 0.802
>12, train: 0.968, test: 0.817
>13, train: 0.973, test: 0.821
>14, train: 0.976, test: 0.821
>15, train: 0.978, test: 0.802
>16, train: 0.978, test: 0.806
>17, train: 0.979, test: 0.802
>18, train: 0.979, test: 0.802
>19, train: 0.979, test: 0.799
</pre>

```python
from matplotlib import pyplot

pyplot.plot(range(1,20), train_scores, '-o', label='Train_acc')
pyplot.plot(range(1,20), test_scores, '-o', label='Test_acc')
pyplot.legend()
pyplot.show()

#최적의 의사결정나무 깊이는?
#훈련데이터의 경우 15정도
#테스트 데이터의 경우 3정도
```

<img src="data:image/png;base64,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"/>


```python
#트리 시각화를 위한 라이브러리 설치
!pip install pydotplus
!pip install graphviz
```

<pre>
Requirement already satisfied: pydotplus in c:\users\administrator\anaconda3\lib\site-packages (2.0.2)
Requirement already satisfied: pyparsing>=2.0.1 in c:\users\administrator\anaconda3\lib\site-packages (from pydotplus) (2.4.7)
Requirement already satisfied: graphviz in c:\users\administrator\anaconda3\lib\site-packages (0.17)
</pre>

```python
#트리 시각화, 생존자 분류 의사결정나무
import graphviz
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(random_state=0, max_depth=3)
tree.fit(X_train, y_train)

#속성들
feature_name=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
#죽었는지 살았는지
tree_graph = export_graphviz(tree, feature_names=feature_name,
                      class_names=['Perish', "Survived"]) #한글이 아니라 영어로 지정해야 됨
graphviz.Source(tree_graph)
```

<pre>
<graphviz.files.Source at 0x1cefa26b610>
</pre>

```python
#외부 파일로 시각화 결과 내보내기
dot=graphviz.Source(tree_graph)
dot.format='gif'
dot.render(filename='titanic_tree', directory='tree', cleanup=True)
```

<pre>
'tree\\titanic_tree.gif'
</pre>

```python
# 성별이 0은 여성이고 1은 남성. true라는 건 0 여성.
#첫번째 분할한 후 true 여성인 경우에서 value(사망 53명, 생존 162) 생존이 훨씬 많기에 이 클래스의 경우 생존으로 분류
#남성인 경우 즉 false인 경우 사망이 훨씬 많아 사망으로 분류
#그리고 남성에서 두번째 분할에서 Age가 14세 이하인지 초과인지 분류. 이하인 경우 29명 같은 경우는 생존이 더 많아서 생존으로 분류 초과인 경우 379명은 사망이 훨씬 많아 사망으로 분류.
```

## 8주차 강의실습


### Accuracy, Precision, Recall, F1-score 값을 각각 구하시오.



```python
from sklearn.metrics import precision_score, recall_score, f1_score
 # 정확도, 정밀도, 재현율, f1-score
print('accuracy: ', accuracy_score(y_test, temp_y_pred))  
print('precision: ', precision_score(y_test, temp_y_pred)) # 정밀도(모델관점)
print('recall: ', recall_score(y_test, temp_y_pred))       # 재현율(정답관점)
print('f1: ', f1_score(y_test, temp_y_pred))               # f1 score(정밀도와 재현율의 조화평균)
```

<pre>
accuracy:  0.8208955223880597
precision:  0.7708333333333334
recall:  0.74
f1:  0.7551020408163266
</pre>
### ROC curve와 AUC



```python
tree.predict(X_test)
```

<pre>
array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
       1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
       0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
       1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
       0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
       0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
       0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0], dtype=int64)
</pre>

```python
# 차례대로 비율을 보면 위 X_test가 왜 0 1이 나왔는지 알 수 있음. 1에 가까우면 생존 0에 가까우면 사망
tree.predict_proba(X_test)[:,1]
```

<pre>
array([0.11904762, 0.11904762, 0.        , 0.94214876, 0.625     ,
       0.34117647, 0.94214876, 0.94214876, 0.34117647, 0.625     ,
       0.11904762, 0.94214876, 0.11904762, 0.94214876, 0.94214876,
       0.625     , 0.11904762, 0.11904762, 0.11904762, 0.94117647,
       0.11904762, 0.94214876, 0.11904762, 0.34117647, 0.625     ,
       0.94214876, 0.11904762, 0.625     , 0.94214876, 0.14285714,
       0.11904762, 0.94214876, 0.11904762, 0.34117647, 0.11904762,
       0.34117647, 0.11904762, 0.11904762, 0.11904762, 0.11904762,
       0.34117647, 0.11904762, 0.11904762, 0.        , 0.94214876,
       0.11904762, 0.11904762, 0.94214876, 0.11904762, 0.34117647,
       0.34117647, 0.34117647, 0.94214876, 0.11904762, 0.34117647,
       0.11904762, 0.34117647, 0.14285714, 0.        , 0.11904762,
       0.11904762, 0.625     , 0.94214876, 0.34117647, 0.625     ,
       0.11904762, 0.94214876, 0.11904762, 0.        , 0.94214876,
       0.94214876, 0.34117647, 0.34117647, 0.11904762, 0.11904762,
       0.94214876, 0.34117647, 0.34117647, 0.11904762, 0.11904762,
       0.11904762, 0.94117647, 0.94214876, 0.11904762, 0.11904762,
       0.94214876, 0.94214876, 0.625     , 0.94214876, 0.34117647,
       0.34117647, 0.11904762, 0.94117647, 0.94214876, 0.14285714,
       0.11904762, 0.94214876, 0.11904762, 0.34117647, 0.34117647,
       0.        , 0.11904762, 0.11904762, 0.11904762, 0.625     ,
       0.625     , 0.94214876, 0.14285714, 0.11904762, 0.625     ,
       0.11904762, 0.94214876, 0.11904762, 0.625     , 0.34117647,
       0.94214876, 0.625     , 0.94214876, 0.11904762, 0.94214876,
       0.11904762, 0.11904762, 0.11904762, 0.34117647, 0.11904762,
       0.34117647, 0.11904762, 0.11904762, 0.11904762, 0.11904762,
       0.625     , 0.11904762, 0.11904762, 0.625     , 0.34117647,
       0.11904762, 0.11904762, 0.625     , 0.11904762, 0.11904762,
       0.11904762, 0.94214876, 0.11904762, 0.625     , 0.94214876,
       0.625     , 0.11904762, 0.94214876, 0.94214876, 0.11904762,
       0.34117647, 0.625     , 0.625     , 0.11904762, 0.94214876,
       0.11904762, 0.625     , 0.11904762, 0.625     , 0.625     ,
       0.11904762, 0.11904762, 0.94214876, 0.625     , 0.11904762,
       0.11904762, 0.11904762, 0.11904762, 0.11904762, 0.11904762,
       0.11904762, 0.94214876, 0.11904762, 0.11904762, 0.625     ,
       0.11904762, 0.94214876, 0.11904762, 0.11904762, 0.625     ,
       0.11904762, 0.11904762, 0.11904762, 0.11904762, 0.34117647,
       0.11904762, 0.625     , 0.11904762, 0.11904762, 0.94214876,
       0.625     , 0.34117647, 0.625     , 0.625     , 0.11904762,
       0.34117647, 0.11904762, 0.625     , 0.34117647, 0.11904762,
       0.11904762, 0.625     , 0.11904762, 0.625     , 0.11904762,
       0.11904762, 0.625     , 0.11904762, 0.625     , 0.11904762,
       0.11904762, 0.34117647, 0.        , 0.625     , 0.11904762,
       0.11904762, 0.11904762, 0.94117647, 0.14285714, 0.94214876,
       0.11904762, 0.94214876, 0.625     , 0.11904762, 0.625     ,
       0.11904762, 0.11904762, 0.94214876, 0.11904762, 0.11904762,
       0.11904762, 0.94214876, 0.625     , 0.625     , 0.34117647,
       0.11904762, 0.94214876, 0.94214876, 0.94214876, 0.11904762,
       0.11904762, 0.94214876, 0.11904762, 0.11904762, 0.94214876,
       0.11904762, 0.625     , 0.11904762, 0.11904762, 0.625     ,
       0.11904762, 0.11904762, 0.34117647, 0.11904762, 0.14285714,
       0.625     , 0.625     , 0.11904762, 0.        , 0.11904762,
       0.11904762, 0.11904762, 0.11904762, 0.34117647, 0.11904762,
       0.11904762, 0.625     , 0.34117647])
</pre>

```python
from sklearn.metrics import roc_curve

# fp 의 비율, tp의 비율, 임계점=roc

fpr, tpr, thresholds=roc_curve(y_test, tree.predict_proba(X_test)[:, 1])
```


```python
# Roc curve 그래프 그리기

#우리가 만든 모형
plt.plot(fpr, tpr, '-', ms=2, label="Decision Tree")

#최악의 모형 노란색 쩜쩜
plt.plot([0,1], [0,1], '--', label="Random Guess")

#X축 이름
plt.xlabel('False Positive Rate')
#Y축 이름
plt.ylabel('True Positive Rate')
plt.title('Roc Curves')
plt.legend(loc="lower right")
plt.show()
```

<img src="data:image/png;base64,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"/>


```python
# AUC 계산 내가 만든 모형의 하단부위 면적. 최대값은 1이며 1에 가까울수록 우수한 모델을 나타냄. 0.5일 경우 분류능력 없음

from sklearn import metrics

roc_auc=metrics.auc(fpr, tpr)
print('Area Under Curve: %0.2f'% roc_auc)
```

<pre>
Area Under Curve: 0.87
</pre>
## 9주차 강의실습


## Random Forest



```python
from sklearn.ensemble import RandomForestClassifier

# 의사결정나무에서 중요한 파라미터가 max_depth였으면 
# 랜덤 포레스트에서는 n_estimators 즉 나무가 몇그루이냐 꼭 설정해줘야 함
RF = RandomForestClassifier(n_estimators=100, random_state=0 )
# n_estimators : 사용할 tree의 개수
```


```python
# 모델 적합
RF.fit(X_train, y_train)
```

<pre>
RandomForestClassifier(random_state=0)
</pre>

```python
temp_y_pred_rf = RF.predict(X_test)
temp_y_pred_rf
```

<pre>
array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
       1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
       1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0], dtype=int64)
</pre>

```python
# 랜덤 포레스트의 정확도 계산
temp_acc = accuracy_score(y_test, temp_y_pred_rf)

print(format(temp_acc))
```

<pre>
0.8097014925373134
</pre>

```python
# 의사결정 나무의 개수가 얼마가 적당한지 많이 하면 할수록 좋은지 살펴봄

scores=[]
for i in range(10, 500, 5):
    RF = RandomForestClassifier(n_estimators=i, random_state=0)
    RF.fit(X_train, y_train)
    
    att=RF.predict(X_test)
    acc=accuracy_score(y_test, att)
    scores.append(acc)
    
    print('>%d, acc: %.3f' % (i, acc))
```

<pre>
>10, acc: 0.802
>15, acc: 0.813
>20, acc: 0.806
>25, acc: 0.813
>30, acc: 0.806
>35, acc: 0.813
>40, acc: 0.806
>45, acc: 0.806
>50, acc: 0.806
>55, acc: 0.799
>60, acc: 0.806
>65, acc: 0.806
>70, acc: 0.810
>75, acc: 0.806
>80, acc: 0.799
>85, acc: 0.799
>90, acc: 0.806
>95, acc: 0.806
>100, acc: 0.810
>105, acc: 0.813
>110, acc: 0.813
>115, acc: 0.813
>120, acc: 0.813
>125, acc: 0.810
>130, acc: 0.817
>135, acc: 0.817
>140, acc: 0.813
>145, acc: 0.817
>150, acc: 0.821
>155, acc: 0.821
>160, acc: 0.821
>165, acc: 0.821
>170, acc: 0.821
>175, acc: 0.821
>180, acc: 0.821
>185, acc: 0.821
>190, acc: 0.821
>195, acc: 0.821
>200, acc: 0.825
>205, acc: 0.821
>210, acc: 0.825
>215, acc: 0.821
>220, acc: 0.821
>225, acc: 0.821
>230, acc: 0.821
>235, acc: 0.825
>240, acc: 0.821
>245, acc: 0.821
>250, acc: 0.817
>255, acc: 0.825
>260, acc: 0.817
>265, acc: 0.817
>270, acc: 0.821
>275, acc: 0.821
>280, acc: 0.821
>285, acc: 0.821
>290, acc: 0.825
>295, acc: 0.821
>300, acc: 0.825
>305, acc: 0.825
>310, acc: 0.828
>315, acc: 0.825
>320, acc: 0.828
>325, acc: 0.825
>330, acc: 0.825
>335, acc: 0.825
>340, acc: 0.825
>345, acc: 0.825
>350, acc: 0.821
>355, acc: 0.825
>360, acc: 0.821
>365, acc: 0.825
>370, acc: 0.828
>375, acc: 0.828
>380, acc: 0.828
>385, acc: 0.825
>390, acc: 0.828
>395, acc: 0.828
>400, acc: 0.828
>405, acc: 0.828
>410, acc: 0.828
>415, acc: 0.828
>420, acc: 0.828
>425, acc: 0.828
>430, acc: 0.828
>435, acc: 0.825
>440, acc: 0.825
>445, acc: 0.825
>450, acc: 0.825
>455, acc: 0.825
>460, acc: 0.825
>465, acc: 0.825
>470, acc: 0.825
>475, acc: 0.825
>480, acc: 0.825
>485, acc: 0.825
>490, acc: 0.825
>495, acc: 0.825
</pre>

```python
# 위처럼 0부터 500까지 숫자를 늘려가며 정확도를 봄
# 의사결정나무보다 정확도 안정성이 있음 y축을 보면 큰 변화가 없음
# 나무개수가 300개 정도 지나면 정확도가 거의 최대치도달
pyplot.plot(range(10, 500, 5), scores, 'b--', label='RF_acc')
pyplot.legend()
```

<pre>
<matplotlib.legend.Legend at 0x1ceff94c340>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYAAAAD4CAYAAADlwTGnAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA5tklEQVR4nO2deXhU1dnAfy8hbLKEsEOQRKUsioAEVNQWd0Qtaqu4fVWqj6Win2sramu1WqvW9RPUYlUQi2hF64Z1rVWpCkGQVQRJgAiyJoAoS8j5/njnOncmk2Qymcwkue/veeaZe8+95857wsN573nPu4hzDsMwDCN4NEm3AIZhGEZ6MAVgGIYRUEwBGIZhBBRTAIZhGAHFFIBhGEZAaZpuAWpCx44dXW5ubrrFMAzDaFDMmzdvs3OuU3R7g1IAubm5FBQUpFsMwzCMBoWIrI7VbiYgwzCMgGIKwDAMI6CYAjAMwwgoDWoPIBZ79+6luLiYXbt2pVuUekWLFi3IyckhMzMz3aIYhlFPafAKoLi4mDZt2pCbm4uIpFuceoFzji1btlBcXExeXl66xTEMo57S4E1Au3btokOHDjb5+xAROnToYKsiwzCqpMErAMAm/xjY38QwjOpo8CYgwzAi2bULpk+Hiy+GJlGveP/4B/zkJ9C5c+rlmj8fXnqpYvvPfw6HHlqx3Tl45BHYsCGy/aST4OijYcsWeOihiv1OOw2GDYP16+HRRyte/9nPYOBAWL0anngCMjJg7FjYf//ExtWQMQVgGI2M226Du+6CDh1g9Ohw+8aNcM45ev2WW1IvV1ERPPUUfP11ZHvfvrEVwJo1cMUVeuxf0LZurQqgpATuuKNiv27dVAF8803s6z/6kSqANWv0unOqNP/854SH1mAxBZAEMjIyGDBgAGVlZeTl5TFt2jSysrIoKiqiX79+9OnT54d758yZQ7NmzdIordHYWR2K+dy8ObK9qEi/Bw1KpTRhzjxTP9Fs26aK4eijoXfvcHuvXlBerp+MjIr9DjpIr1XG4MFVXz/mGL1+0EFQWBj/OBoTjWIPIN20bNmSBQsWsHjxYrKzs5k0adIP1w488EAWLFjww8cmf6Ou8Yr89eoV2e4pgBYtUioOoDIVFUFZWcVrO3bAL38J771X8ZpI7Mk/meTmhv82QaPRKYARIyp+HnlEr333XezrU6bo9c2bK16rKUceeSRfR69x42DOnDkMHz6cwYMHM3z4cJYvXw7Avn37uP766xkwYACHHnooDz/8MABz585l+PDhDBw4kGHDhrFjx46aC2s0Sq69FmbOhBNOiGz33nLHjEm9TJs2QV5e+P+in27dIDOz4lv444/DuHF1L9vhh0NOTt3/Tn3ETEBJZN++fbz77rtccsklP7R99dVXDAqtuY866qiI1YGfvn378sEHH9C0aVPeeecdbrrpJmbOnMnkyZMpLCxk/vz5NG3alK1bt7Jnzx7GjBnDc889x9ChQ9m+fTstW7ZMxRCNBsDQofrZu1cnVg/vLXf79orX6hpvco8VlpKRoauV6Lfwd97RjeO65k9/qvvfqK80OgXw/vuVX2vVqurrHTtWfb0yvv/+ewYNGkRRURFDhgzhxBNP/OGaZwKqjm3btnHRRRexYsUKRIS9e/cC8M477zBu3DiaNtV/quzsbBYtWkS3bt0YOnQoAG3btq250EajZN8+eP11mDwZvvxSPx4/+xl89hnMmQNr18IBB6ROLm9yryybe25uxRVAYWHl9xvJodGZgNKBtwewevVq9uzZU+lbflX8/ve/59hjj2Xx4sW8+uqrPwRxOecq+PTHajMMUA+b0aPh44910t23L3zthBPUOwhSv+np/V5lE3peXsUVQFFR7BVDsvniC+jXD956q+5/q75hCiCJtGvXjv/7v//j3nvv/eENPl62bdtGjx49AJjibUoAJ510Eo899hhlod2zrVu30rdvX9atW8fcuXMB2LFjxw/XjWDjTaLHHqtmnvXr9dw5ffPPzo68L5VydegAbdrEvn7bbfD55+Hzb7/VfYNUrACyslQJrFhR979V3zAFkGQGDx7MwIEDmTFjRo36/fa3v+XGG2/kqKOOYp/vte3SSy9l//3359BDD2XgwIFMnz6dZs2a8dxzz3HllVcycOBATjzxREv7YADhN+3jjos8/+Yb3ex8/324807dI0gl554L99xT+fVu3aBr1/D5li3Qp4/67Nc1XbqoZ1QgXUGdcw3mM2TIEBfN0qVLK7QZiv1tgscf/uCciHOLFjkHzj39tLb/9796/vrraRWvUjZtcu6225ybPz89v9+3r3M/+1l6fjsVAAUuxpxqKwDDaEQUFUGPHhrcdNVV4cAqvw3+m28gDr+EpFFeDh9+CFu3Vn5PWRn84Q/w0Uepk8tP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"/>


```python
# 의사결정나무 모델 성능
# 정확도, 정밀도, 재현율, f1-score
print('accuracy: ', accuracy_score(y_test, temp_y_pred))  
print('precision: ', precision_score(y_test, temp_y_pred))
print('recall: ', recall_score(y_test, temp_y_pred))
print('f1: ', f1_score(y_test, temp_y_pred))
```

<pre>
accuracy:  0.8208955223880597
precision:  0.7708333333333334
recall:  0.74
f1:  0.7551020408163266
</pre>

```python
# # n_estimators = 300으로 새로 적합
RF = RandomForestClassifier(n_estimators=300, random_state=0 )

# 모델 적합
RF.fit(X_train, y_train)
temp_y_pred_rf = RF.predict(X_test)
```


```python
# 렌덤 포레스트 모델 성능
# 정확도, 정밀도, 재현율, f1-score
print('accuracy: ', accuracy_score(y_test, temp_y_pred_rf))  
print('precision: ', precision_score(y_test, temp_y_pred_rf))
print('recall: ', recall_score(y_test, temp_y_pred_rf))
print('f1: ', f1_score(y_test, temp_y_pred_rf))
```

<pre>
accuracy:  0.8246268656716418
precision:  0.7912087912087912
recall:  0.72
f1:  0.7539267015706805
</pre>

```python
# ROC 커브/ AUC 비교

from sklearn.metrics import roc_curve

# fp 의 비율, tp의 비율, 임계점=roc

fpr1, tpr1, thresholds1=roc_curve(y_test, tree.predict_proba(X_test)[:, 1])
fpr2, tpr2, thresholds2=roc_curve(y_test, RF.predict_proba(X_test)[:, 1])
```


```python
# 두 모델의 Roc curve 그래프 그리기

plt.plot(fpr1, tpr1, 'b--', label="Decision Tree")
plt.plot(fpr2, tpr2, 'r--', label="Random Forest")

#최악의 모형을 비교 차원에서 그리도록 함
plt.plot([0,1], [0,1], 'g--', label="Random Guess")

# 타이틀
plt.title('Roc Curves')

#X축 이름
plt.xlabel('False Positive Rate')

#Y축 이름
plt.ylabel('True Positive Rate')

# 연혁 하단부에 위치
plt.legend(loc="lower right")
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABL60lEQVR4nO3dd3gUVffA8e8hISShhI5UAaW30EUEglRBpSiKIkXgRZqN91UQFUGxIT+liwiCCAiKiIgoCEiRDho6IiBNUHooSUi7vz/uJiQhhA1ks9ns+TzPPtmdtmcSmDNzZ+65YoxBKaWU98rm7gCUUkq5lyYCpZTycpoIlFLKy2kiUEopL6eJQCmlvJwmAqWU8nKaCJRSystpIlBZhogcFpEIEbksIv+IyAwRyeWC7ykvIl+LyBkRCRORHSIySER80vu7lMoImghUVvOQMSYXEAzUBF5Jz42LyF3AJuAYUM0YEwR0AuoAuW9he77pGZ9St0ITgcqSjDH/AEuxCQEAEXlYRHaLyAURWSUilRLNKykiC0TktIicFZEJN9j0CGC9MWaQMeak47v+MMY8aYy5ICIhInI88QqOK5XmjvfDRWS+iMwSkYvAUMdVTP5Ey9d0XG1kd3zuKSJ7ReS8iCwVkTsd00VEPhKRU4muTKqmx+9PeRdNBCpLEpESwAPAAcfn8sCXwAtAIWAJ8L2I+DmadBYDR4DSQHFg7g023RyYf5vhtXNsIy/wAbABeCTR/CeB+caYaBFpDwwFOjriXuvYD4CWQGOgvGNbjwNnbzM25YU0EaisZqGIXMI23ZwC3nBMfxz4wRjzszEmGhgNBAD3AvWAYsBLxpgrxphIY8yvN9h+AeDkbca4wRiz0BgTZ4yJAOYAT4A9ywc6O6YBPAO8a4zZa4yJAd4Bgh1XBdHY5qiKgDiWud3YlBfSRKCymvbGmNxACPYAWdAxvRj2jB8AY0wcNlkUB0oCRxwH2ps5CxS9zRiPJfs8H2ggIsWwZ/gGe+YPcCcw1tGcdQE4BwhQ3BizEpgATAT+FZEpIpLnNmNTXkgTgcqSjDGrgRnYM3+AE9iDKpBw5l0S+Bt7YC7l5I3b5SRtxknuChCY6Ht8sE06ScJLFusFYBnwGLZZ6EtzrSzwMeAZY0zeRK8AY8x6x7rjjDG1gSrYJqKXnNgHpZLQRKCysjFACxEJBr4C2opIM8dN2P8CV4H1wGZsc897IpJTRPxFpOENtvkGcK+IfCAidwCIyN2Om795gf2Av4i0dXzPa0AOJ2KdA3TDJpk5iaZPBl4RkSqO7woSkU6O93VFpL7je64AkUCsU78ZpRLRRKCyLGPMaWAm8Lox5g/gKWA8cAZ4CPuoaZQxJtbx+W7gKHAce08hpW0eBBpgbyrvFpEw4BtgK3DJGBMG9AemYq82rji2dzOLgHLAv8aY7Ym+71vgfWCu4ymjXdib4AB5gE+B89hmr7NcuwJSymmiA9MopZR30ysCpZTycpoIlFLKy2kiUEopL6eJQCmlvJzHFbwqWLCgKV26tLvDUEopj7Jt27YzxpjkfVoAD0wEpUuXZuvWre4OQymlPIqIHLnRPG0aUkopL6eJQCmlvJwmAqWU8nKaCJRSystpIlBKKS/nskQgIp85htDbdYP5IiLjROSAY4i9Wq6KRSml1I258opgBtA6lfkPYKstlgP6AB+7MBallFI34LJ+BMaYNSJSOpVF2gEzHQNwbBSRvCJSVIfaU0p5rehoiI0Ff38ID4fVq+HcOcKOn2FrZBTN3nDNuEPu7FBWnKRD9h13TLsuEYhIH+xVA6VKlcqQ4JRS6rYYA7//DufPw7lz134GB0Pr1nD5Mjz8cNJ5ly/Dm2/C66/baW3a8Psd0LMd/J0rOwci+pMnIGe6h+rORCApTEtxcARjzBRgCkCdOnV0AAWlMsratfDDD9dPf+UVCAqC5cvtK7nhw+1Z7Q8/2G0k9+67IALffgubNiWd5+dnD4YAc+dCaGjS+blywWuv2feffw579yadX6AAvOQ4c54yBQ4dSjq/WDF47jn7fvx4+PvvpPPLlIFnnrHv/+//4PTppPMrVoQePez7//wHjh1LejDv2BE+/dTOv+cee5af2IABNhEEBNh5pUpBjRqQPz/kywchIQC88EEQXzXrysmGcwiMy89b9Sa4JAkAYIxx2Qs7itOuG8z7BHgi0ec/gKI322bt2rWNUsqFoqKMCQ+378eMMSZHjutfx4/b+W+9lfL8sDA7f/DglOfHxdn5AwZcPy9//muxdO9+/fxSpa7N79jx+vmVK1+b36LF9fPr1782v0GD6+c3b35tfrVq189v3/7a/CZNjKlXz5hWrYx54glj+vc3Ztasa/OXLDFm9Wpjdu60v7P432sKrlwx5pNPjDl/3n4OHt3KMBzT+cunzbnwc6n9xZwCbDU3OK66dIQyxz2CxcaYqinMawsMBNoA9YFxxph6N9tmnTp1jNYaUspF/voLnnwSKlWCzz5zdzRZkjH2YujqVVi82F6QHDgAs2fDucuXmDwpO8/08mfV4VVEx0bT4q4W6fK9IrLNGFMnpXkuaxoSkS+BEKCgiBzHDvqdHcAYMxlYgk0CB4Bw4GlXxaKUcsK8edCnj33//PPujcUDGQNhYfYAX6SI/fzOO3D8uD3Yx7+eegpGj4a4OHj0UbtujhxQ5/GlHKjUhyOlnwLeJqR0SIbF7sqnhp64yXwDDHDV9yulnBQTA/3723bte+6BOXNsO7lKEB0NJ0/aA7mI/TUBDBwIu3dfO8iHh0OnTvDVV3a5jz6yyxUvbl+1akH9+nZaQABs3w4B+c/x9tZBfL79cyrmrsiD5dtm+P55XBlqpVQ6O3QI5s+HwYPhrbcge3Z3R5Rh4s/iE5+xGwM9e9r5XbrAihVw6pSdDjYJbNhg3x8+bJNEzZrw4IP2YF+jxrXtnzhh733fyOlcK+gytwtnI87yaqNXea3xa/j7+rtkX1OjiUCprCgsDCZMgPvugyZN4OhRePvt65fr2dOeov7xBxRKccwSj3bqFBw8mPRAf+UKTJxo53fqBN98k3Sd4sWvJYLy5SEw8NoZffHikHhcrMWLU//+1JIAQOGchSmTrww/PfUTwXcEp2XX0pUmAqWykqgo+Phje2Z/9qy94dukiU0MixZdv3yLFjYReFASiD8zF7FPjm7cmPRAf/IkrF9vL2xGjIBJk66t6+dnn9aMi4Ns2aBzZ2jQIOmBvlixa8u/8UZ6x274fPvn/HbyN8Y9MI5qRaqxvud6RFJ6mj7jaCJQKjM6cwZGjrQH8KlTwccHZs6EX35Julz27PZZebA/33/fNvU0awajRtlGaYBq1ewRMpOLjoZ//rF5yd8ftm617e0nTiQ92O/daw/oCxfC0KF23QIFrh3Mr1yBvHntY/5t216bXqCATSDx4m/WZoS/zv/FM4uf4edDP9OoVCMioiMIyB7g9iQAmgiUynwuX7ZHr99/h6JFr50C798PK1cmXTZHjmvv9+yxj6tMmgQtWyY94rlZ8rb46tXhjjtgyxab7+Kn//uvXXbdOrj3Xti3D8aOtWfpxYtfa4uPv43Rqxc8/rid759C03pwsH25U2xcLBO3TOSVFa+QTbIxqc0knqnzDNkk8xR/dmk/AlfQfgQqyzl9+lrTzNixMGMG7NwJCxbYEgSZXPxZfOIz9saN7QF4+3bbDh//RE28uXPtAXz9eujXL2mzTPHi8NBDNgfGxNiLoUyU09Ls38v/UmFCBe4teS+TH5xMqSD3lMlxSz8CpZQTFi2CJ56AJUtsW/6BAxAZaZuBMkESiI21Z+WJD/J//20vODp0gCNH7JOmyc8nR4+2iSB//qRP1MS/qjq6mN57r00WN+LroUeo6NhoZu+cTbca3SiSqwi/PfMbZfKWyRTNQCnx0F+zUh7i22/tc/lFi8K4cXba8OH24XNjbC2e6tWhdm07b/z4DAkrvndrfIjHjiU90LdoYdveY2KuHbTjFShw7cmZO+6w9dESH+SLF4eCBe38kiVtPzVvsu3ENnou6smOf3dQNFdRWt3dirL5yro7rFRpIlDKlaZMgVWroGnTa9OOHLHt+WCnz5xpC6mlg/i2+EuX7EEYbG7Zsyfpgf6++2zXAYC+fe1jln5+15pm4tvbc+SAr7+2tx7im24St8XnyGGfzFEQER3BiNUjGL1+NIVzFubbx7+l1d2t3B2WUzQRKJUeoqPtEVXEPuUD9sx/2zZ7xr9kybVlp0+/5a+I79165Qo0b26nv/EGrFmTtHdr3bqwebOd//nntuNT4huu8T1jwa6bP789i0+p5SIjn6zxZO3ntWfZwWX0rtmbD1p+QF7/vO4OyWmaCJRKD3/9ZZ/Zj68fAPYmcPHi0L59qqum1Lv13Dn473/t/P/9D2bNStq79Y47rj0N+u+/1/duLV/+2vY3bky9rb1ChbTvrrIuXr2In48f/r7+DL1vKC/f+zLNyjZzd1hppolAqds1YoStQwDX6txDQvfVs2fhz2Sdnk6csBcOOXLYA358TZp4InZT2bPDXXddf7O1RIlry06enHp4nnrDNbNb8ucS+i7uy1PVn+KdZu/QpHQTd4d0y/SfiFK3IHHv1rC/LxN1/CrhpRvz2eK6hM6zB/sFC2ynp2nTbBmfePFt8efP2zP75Af5+Lb4+Gfl+/XL+P1TN3Ym/AwvLn2RWTtmUblQZR6u4P6nu26XJgLlfUaMsA3jYOsM/Pyzff/BB/DTT8QZiLpqz6R98+biz9HfMW0a1PtxBKWPruHqVVtqOCRuJXTuzJKHv+RJx4BU+S9eO5hHRdlpHTtClSpJn6hJ3BZ///32pTK/nw/+TJcFXTgfeZ5hjYcxtNFQcvjmuPmKmZwmAuU1jIGLFyHw40+JuxpN3N3lCcjpw6FD8OKL0GJrDHVPRxHlGFmwQgUoHBjFP//YppvRATHcZaLImxNy5IPwovcR2KwZbdrYwmY36t169932pTxf0dxFKV+gPB+3/ZhqRaq5O5x0oz2LVeY3ahR8/70tE1m4sO15O21akkXiDJyY+iPHL+Qix7RJFF75JTlzQt4giIqGHTvgflnFpXAfwsjDVzzG2femMniwLcyZUvNMs2a2fT4uzp7BZ9K+QMqFjDFM+30av5/8nYltJyZMy6wdw1KjPYuVZ5s1i5jjJ/l1tXD4CuRb7EOFv/zIncsesOOMfVT/4UpwBfgPvjyOHyVKQN5CtoknR27o3RmKlYQ/d7xF/arlKNLDbr5UKZsobiRb5ikJozLQofOH+M/3/2HlXysJKR2SqYrEpTe9IlBuExcH2V59BVau5EKYrawQFQWj7/mGnedL0CPbTLpfngg7drCEB2gbuSBh3fz57ZB/Y8faz++/D/nyJT2jL1BAD+Iq7WLjYhm3aRyvrnwV32y+jG45mt61emeqInG3Qq8IVIaKfy7+7FnbtAK282zyuvElSsCWU3MgJoY9V6oTFgbZBDZtyUb2O8GvkD/45YeQEIrV6cLqFteeqAkISPqdiZ/KUep2nAk/w4jVI2hWthkft/2YEnlK3HwlD6dXBMopo0fD7NlJp4nAb7/Z92++aWvWhIdf6/lasKDtUwW2d+ovv1w7W+986VOiG91P74dPQb58HPavSM6cehav3CMqNopZO2bRI7gH2SQbhy8c5s6gO7NUM5BeEag0O3rUVkx49107Bmu+fLYtPbHE/0fy57fz/f2hTZvrOz3Nm2fLCSdo/wMcXQUNbHYp7aodUeomtvy9hZ6LerLr1C5K5ClBy7taUjpvaXeHlaE0EagkjLGVEl580b4/eNAmgl697OtGBg60rxtJkgTi7d592/EqdavCo8MZ9sswPtr4EUVzFWVR50W0vKulu8NyC00EKsHff9uh/X78EUJCbG20xAN1K5WVtJvbjuWHltOnVh9GtRhFkH+Qu0NyG22NVQmmTrWPYY4bZ0vnuCwJfP75td68SmWgsMgwImMiAXi98eus7LaSTx76xKuTAGgi8Hr//msHCAcYMsSOkPjssy64YbtiBVSubEcdL1LEVuSML6+pVAZYvH8xVSZVYcQqO4BC4zsb07RM05us5R00EXixr76yNXCefNIOSZgjx7XHPROsXWvrI9x557XXpk123sKFSafHv3btsvNnzbo2rXNnmwSuXoXWre0jSF27ZuTuKi91+sppnvzmSR768iHyB+SnY6WO7g4p09F7BF7ozBkYMMAmgrp1bUtNijdzAX7/3d4xfuIJmykA8ua1P4sUSblaWvxoW8WLJ52fN6+9KlAqgyw7uIwuC7oQFhnGiJARDLlvCH4+fu4OK9PRfgRe5q+/7OhU58/bAbRefjlRvfqePe2d4nhz59rnQX/80fbYSt6LS6lMbvep3fT7oR8ft/2YKoWruDsct9J+BMqWc8hmbwA//rh9FLRGjWQLrV5tz+bjz+ILFbJn8IlH3VIqE4szcUz9bSq/n/ydjx+0B/81T69xd1iZniYCL/Djj3a4w6VLbSevceNusGCjRjZjfPJJhsanVHo4cO4A//n+P6w6vIqmpZsmFIlTN6eJIAu7eBEGDbIVm6tUsZ9T9PPPcOGCzRbh4RkZolK3LTYuljEbx/D6L6+T3Sc7nz70Kb1q9spS5SFczaWJQERaA2MBH2CqMea9ZPODgFlAKUcso40x010Zk7dYscI2+R8/bpv3R4y4dq+XPXugaVN78Pfzg4gImym2b3dnyErdkjPhZxi5diQt7mrBpDaTKJ6nuLtD8jguSwQi4gNMBFoAx4EtIrLIGLMn0WIDgD3GmIdEpBDwh4jMNsZEuSoub/HFF/Y+77p19uZwEiLQsKFNAsUd/2lCQjI6RKVu2dWYq8zcPpNetXpRJFcRQp8JpVRQKb0KuEWuvCKoBxwwxhwCEJG5QDsgcSIwQG6xf71cwDkgxoUxZWlr19onNKtVs/cBfH0hMDDRAuHhUKmSHXdxwYIbbUapTG3T8U30WtSL3ad3c2feO2l5V0vuzHunu8PyaK7sUFYcOJbo83HHtMQmAJWAE8BO4HljTFzyDYlIHxHZKiJbT8fXNVYJIiLsvYAmTWDYMDstT55kSQDg3DlbVjS+Q5hSHuRK1BUGLR1Eg2kNCLsaxg9P/uC1ReLSmysTQUrXaMk7LbQCQoFiQDAwQUTyXLeSMVOMMXWMMXUKFSqU3nF6tE2boGZNe5Lfty/MHrDePu8vAvPn24WWLbOfS5a0nytWdF/ASt2i9vPa89HGj+hbpy+7+++mTbk27g4py3Bl09BxoGSizyWwZ/6JPQ28Z2yvtgMi8hdQEdjswriyjGVLDQ89EMMdxX34+edsNL8/Dmbut2M+Dhxom4EAypa9dqmQIwe0a+e+oJVKgwuRF8jhk4OA7AEMazyM1xu/TuM7G7s7rCzHZT2LRcQX2A80A/4GtgBPGmN2J1rmY+BfY8xwESkC/AbUMMacudF2tWexPc77+0Ns5y74zJtD+JRZBP6ni+0QFn/T99AhKFPGrXEqdTsW/bGIfj/0o2v1rrzX/L2br6BS5ZaexcaYGBEZCCzFPj76mTFmt4j0dcyfDLwFzBCRndimpMGpJQGvERdnG/6TicKPt0dlZ84XsWz9NZKg/XuhXDkCGzi6CJcuDSNH2h7BOpCA8lCnrpziuR+fY97ueVQvUp1HKz/q7pCyPK01lBnt3g1Vq143eXjJaYw41pM3Wm1k+NIGduLDD8N332VwgEq5xk8HfqLLgi5cjrrM641fZ3DDwWT3ye7usLIErTXkKS5etPWgCxeGUaMAe3GwapW937vxSl2+/Rba1ysFs+18WrVyX7xKpbOSeUpSrXA1JrWdROVCWqk2o+gVQWZgjB2k5aOP7OfXXoO33kqY1bKlHTx+0iQoWNCNcSqVzuJMHJ9s/YTQf0L55CGtceVKekWQ2b31lk0CXbtC7drE1qnP+DHw6KO2SNx336XQJ0ApD7f/7H56L+rN2qNraVG2BZExkfj7+rs7LK+kicDdtm2DN96wA798/jkHDgo9etjSEJcv24sDTQIqK4mJi+H/1v8fb6x6g4DsAUxvN53uNbpreQg30kTgboUKwSefEPfgw0ycIAwebEsAzZwJTz3l7uCUSn9nw8/y/rr3aVOuDRPbTKRo7qLuDsnr6ZjF7laqFPTpw3sz7uC556BxYzvkb9eutjOwUlnB1ZirfLL1E+JMHEVyFWF73+0seHyBJoFMQq8I3OncOThyBM6f59mn61GsWC66d9cEoLKWDcc20GtRL/ae2ctd+e+iednmlAwqefMVVYbRROAuly7ZEtCRkQDkDg2lR4/kY0cq5bkuR13mtZWvMW7TOEoGleSnLj/RvGxzd4elUqCJwNUOH0442AP24J87N/zzD0RG8lVgdw5U7cDQ6tXdFqJSrtB+bntW/LWCgXUH8k6zd8idI7e7Q1I3oInAlX7+2XYCSGzhQmjXjpiNW/EFlsc0ZcCUdinXalXKw5yPOI+/rz8B2QMYHjKc4SHDua/Ufe4OS92E04lARHIaY664Mpgs5+xZ+3P06GsjgdWuDcDry5vwB9/QZdoD1NAWIZUFLNi7gAFLBtCtejfeb/G+JgAPctNEICL3AlOxI4iVEpEawDPGmP6uDs7jNWsGa9bYg3+izgDTp8N7M4vx8ssdeUQfEVUe7p/L/zBwyUC+2fsNwXcE07lqZ3eHpNLImSuCj7ADyCwCMMZsFxEtCO6MQoXsK5mQEHjuOXjnnYwPSan09OOfP9JlQRfCo8N55/53+N+9/9MicR7IqaYhY8yxZL3+Yl0TThZz+DD8+is89BAEBXHxor1PXKYMjB3r7uCUun135r2TmkVrMrHNRCoW1JHvPJUzHcqOOZqHjIj4icj/gL0ujitr2LjR9gw7eZKrV6F1a3j6aXcHpdStizNxTNg8gf8s+g8AlQtVZkW3FZoEPJwziaAvMAA78Pxx7NjCen8gjZ57DjZsgLZt3R2JUrfmjzN/0Hh6Y5798VmOXTxGZEzkzVdSHsGZpqEKxpguiSeISENgnWtCyiLOnoU9ewCYNw+mTIEhQ6BTJzfHpVQaRcdGM3r9aEasHkFg9kBmtJtBtxrdtEhcFnLT8QhE5DdjTK2bTcsomX48gogI8PWF5cuhTRsAyvgeo2LzEixeDD4+bo5PqTQ6deUUFSdUpFnZZox/YDx35LrD3SGpW3BL4xGISAPgXqCQiAxKNCsPdgxiBbZUxMaNdgQZsG0///0vvPQS/PILW/8qQNFPSzBnjiYB5TkiYyL57PfP6FunL4VzFmZHvx2UyFPC3WEpF0mtacgP23fAF0jcN/wioKNJxxs7Fl5/Pem0PHkw+QsgISHUCYF1PbSQnPIcvx79lV6LerH/7H7KFyhP87LNNQlkcTdMBMaY1cBqEZlhjDmSgTF5hqgoO5hwr17QoQOEhdnp2bJhatbimWfs6GLDhmkSUJ7h0tVLvLLiFSZumUjpvKVZ9tQyLRLnJZy5WRwuIh8AVYCEceSMMfe7LCpPMG8edOsGn39ufyYy+WP49FMYOtRNsSl1C9rPa88vf/3C8/WfZ+T9I8nll8vdIakM4kwimA3MAx7EPkraHTjtyqAyNWNg5UrbUQygTtJ7L7/+ah8VbdMG3nzTDfEplQbnIs7h7+tPYPZA3mr6FtJUaFCygbvDUhnMmX4EBYwx04BoY8xqY0xP4B4Xx5V57dwJzZvb50GzZ7ftPw7Hj9sB58uUgdmz9eawytzm75lPpYmVGL5qOAD3lrxXk4CXcuaKINrx86SItAVOAN5756hMGfjhB8iVC+6+G/LkSZi1caO9dbBwIeTN67YIlUrVyUsnGbBkAN/u+5baRWvTpVqXm6+ksjRn+hE8CKwFSgLjsY+PDjfGfO/68K6X2fsRhIVBUJC7o1AqZT/s/4Gnvn2KyJhIRoSMYFCDQfhm02FJvMEt9SOIZ4xZ7HgbBjR1bLBh+oXnYU6dslcELVokNAt9+ikUKAAdO2oSUJlb2XxlqVusLhPaTKB8gfLuDkdlEje8RyAiPiLyhIj8T0SqOqY9KCLrgQkZFmFmYgy8/DL07AknTgB2uIH+/e3DQze5uFIqw8XGxTJ241h6fdcLgEqFKrGs6zJNAiqJ1K4IpmGbgzYD40TkCNAAGGKMWZgBsWU+w4fbI/7gwVCvHseO2ZvDd90FM2dqfwGVuew5vYfei3qz4fgG2pRrQ2RMJP6+/jdfUXmd1BJBHaC6MSZORPyBM8Ddxph/Mia0TCY6Gt5+2x75332XiAjbjywy0t4c1iYhlVlExUYxat0o3lrzFrn9cjOrwyyerPakFolTN5Ta46NRxpg4AGNMJLA/rUlARFqLyB8ickBEhtxgmRARCRWR3SKyOi3bz1BxcRAbC7VqgQhffw3btsGsWVBRS7GrTORC5AU+2vgRHSp2YM+APXSp3kWTgEpValcEFUVkh+O9AHc5PgtgjDHVU9uwiPgAE4EW2HEMtojIImPMnkTL5AUmAa2NMUdFpPCt74qL+fjAe+9BYztKZ7duULWqzQtKuVtEdATTfp9G/7r9KZyzMDv77aRY7mLuDkt5iNQSQaXb3HY94IAx5hCAiMwF2gF7Ei3zJLDAGHMUwBhz6ja/M32cOQNz50JMTNLpvXrx676CBO2EatU0CajMYc2RNfRe1Js/z/1JpYKVaFa2mSYBlSapFZ273UJzxYFjiT4fB+onW6Y8kF1EVmErnI41xsxMviER6QP0AShVqtRthuWE2bPhhReum3yicnM6dClI6dKwebPeHFbudfHqRYYsH8LHWz+mTN4yLO+6nGZlm7k7LOWBXNmTJKXDZPIHLH2B2kAzIADYICIbjTH7k6xkzBRgCtgOZS6I1bp6FaZOhccfh9697Q1ih4gIaNc2N1FR9r6AJgHlbu3ntmfV4VW8eM+LvNX0LXL65XR3SMpDuTIRHMc+fhqvBLY8RfJlzhhjrgBXRGQNUAPYjzusWwcDB0KhQvDYYwmTjYE+z8K2UFi0CCpUcEt0SnEm/AyB2QMJzB7I2/e/jYhwTwnvLf2l0oczRecQkQARSevhbwtQTkTKiIgf0BlYlGyZ74BGIuIrIoHYpqO9afye9BN/T6BY0vbVefPsVcCbb8KDD7ohLuX1jDHM3TWXShMr8cYvbwDQoGQDTQIqXdw0EYjIQ0Ao8JPjc7CIJD+gX8cYEwMMBJZiD+5fGWN2i0hfEenrWGavY7s7sB3Xphpjdt3ivqSfbEl/LR076vgCyn3+vvg37ee154lvnqBM3jJ0q9Ht5isplQbONA0Nxz4BtArAGBMqIqWd2bgxZgmwJNm0yck+fwB84Mz2MtqxYxAQAAUL2lsGSmW0xfsX02VBF6JjoxndYjQv3PMCPtm0vrlKX84kghhjTJi3dUgJD4eHH7b3B3777bqLBKUyxN357+bekvcy/oHx3J3/bneHo7IoZxLBLhF5EvARkXLAc8B614blJo0awZEjmMJF6N0Ttm+HxYs1CaiMExsXy7hN49j+73ZmtJ9BxYIV+bHLj+4OS2VxzhzinsWOV3wVmIMtR/2CC2Nyn4AAKFWKDyfm4MsvYeRIO+SkUhlh96ndNPysIYOWDeJM+BkiYyLdHZLyEs5cEVQwxrwKvOrqYNzu0CEOvf0lY6d355FHSvDKK+4OSHmDqNgo3vv1PUauGUmQfxBzOs6hc9XOWh9IZRhnrgg+FJF9IvKWiFRxeUTudOAAZT97jQEPHWXGDO00pjLGhcgLjNs0jk5VOrGn/x6eqPaEJgGVoW6aCIwxTYEQ4DQwRUR2ishrrg4so0VEXOtGMHiwHZJYKVcJjw5n7MaxxMbFJhSJm91xNoVyFnJ3aMoLOXUb1BjzjzFmHNAX26dgmCuDymjGwNNPaz8BlTF++esXqn1cjReWvsCqw6sAKJq7qHuDUl7NmQ5llURkuIjswg5RuR5bLiLLGD3a9h7WaqLKlcIiw3jm+2e4f+b9CMIv3X/RInEqU3DmZvF04EugpTEmea0gj7dsGQwZAp062VpzTHd3RCqraj+vPWuOrOGle19ieMhwArMHujskpQAnEoExJssWMzl4EDp3hiqVDbNKv45U7QfnzkHu3O4OTWURp6+cJqdfTgKzB/Jus3fxER/qFq/r7rCUSuKGTUMi8pXj504R2ZHotTPRyGUeLSICypaFRdNO4/fB2/DDD5AvH/i6siir8gbGGObsnJOkSNw9Je7RJKAypdSOeM87fma5epvG2EdDq1aFLVtATjtmJB+RTKlbcPzicfr90I/F+xdTv3h9egT3cHdISqXqhlcExpiTjrf9jTFHEr+A/hkTnmuMGgX9+tnjvj6urdLToj8WUXliZVb+tZKPWn3Eup7rqFI4a3e/UZ7PmcdHW6Qw7YH0DiSjrF4Nr7wCFy7Y8eiVSk/lC5TnvlL3sbPfTq0UqjzGDZuGRKQf9sy/bLJ7ArmBda4OzFWWLbNXAVOn6tWAun0xcTGM2TiGHf/uYGaHmVQsWJElXZbcfEWlMpHU7hHMAX4E3gWGJJp+yRhzzqVRuVi2bJAz8fCu+fLZ0ejvvNNtMSnPs+PfHfRa1IutJ7bSrkI7ImMi8ff1d3dYSqVZaonAGGMOi8iA5DNEJL+nJ4MksmeHuvo0h3LO1ZirvLP2Hd759R3yB+Tnq0e/4tHKj2p9IOWxbnZF8CCwDTBA4n/lBijrwrhcplEjiItLNjEiwg5K3LAhVK7slriU57h49SKTtk7iiapP8FGrjygQWMDdISl1W8QY4+4Y0qROnTpm69at6bvRU6egSBGYOBH6e/QDUcpFrkRdYcq2KTxX/zl8svnw7+V/KZKriLvDUsppIrLNGFMnpXnO1BpqKCI5He+fEpEPRaRUegeZUSIi4Px5d0ehPMmKQyuo9nE1Bi0bxOojqwE0CagsxZnHRz8GwkWkBvAycAT4wqVRudDIkVC4sLujUJ7gQuQFei/qTfMvmuObzZfVPVZzf5n73R2WUunO2cHrjYi0A8YaY6aJSHdXB+YqHtYSptyow7wOrD2ylsENB/NGkzcIyB7g7pCUcglnEsElEXkF6Ao0EhEfILtrw3Kd2FjtSKZu7N/L/5LLLxc5/XLyXrP38M3mS+1itd0dllIu5UzT0OPYget7GmP+AYoDH7g0KheKjrZPiyaRPz/s3QtPPOGWmJT7GWP4YvsXVJ5UmTdW2SJx9UvU1ySgvIIzQ1X+A8wGgkTkQSDSGDPT5ZG5SIqJwNcXKla0HcuU1zkadpS2c9rSbWE3KhSoQK+avdwdklIZypmnhh4DNgOdgMeATSLyqKsDc5W2beHVVxNNeOEFqFTJjkyzfbu7wlJu8t2+76gyqQprjqxhXOtxrH16LZUKVXJ3WEplKGfuEbwK1DXGnAIQkULAcmC+KwNzldat7SvBV19du0QI1BGjvIUxBhGhYsGKhJQOYfwD4ymdt7S7w1LKLZxJBNnik4DDWZwc9D4zOn3alp8umnis8FatYMoUt8WkMk5MXAz/t/7/2HlqJ7M6zqJCwQp8/8T37g5LKbdy5oD+k4gsFZEeItID+AHw2PKKzz0HTZq4OwrlDtv/2U79qfUZsmII4dHhRMZEujskpTIFZ8YsfklEOgL3YesNTTHGfOvyyFzkupvFM2dqD7MsLjImkpFrRvL+uvcpEFCA+Z3m80jlR9wdllKZRmrjEZQDRgN3ATuB/xlj/s6owFwlOhr8/BJNaN7cbbGojHHp6iU+2fYJXap14cNWH5I/IL+7Q1IqU0mtaegzYDHwCLYC6fi0blxEWovIHyJyQESGpLJcXRGJzYinka67Ili6VJ8WyoIuR11m9PrRxMbFUihnIfb038OM9jM0CSiVgtQSQW5jzKfGmD+MMaOB0mnZsKMH8kTssJaVgSdE5Loaz47l3geWpmX7t+q6RPD007bqqMoylh1cRtVJVXn555dZc2QNAIVyFnJzVEplXqndI/AXkZpcG4cgIPFnY8xvN9l2PeCAMeYQgIjMBdoBe5It9yzwDZAhI8P072+fGlJZz7mIc/x32X+ZETqDCgUqsPbptTQs1dDdYSmV6aWWCE4CHyb6/E+izwa4WRnG4sCxRJ+PA/UTLyAixYEOjm3dMBGISB+gD0CpUrdXAbtDh9taXWViHeZ1YN3RdQy9byivN3ldh41Uykk3TATGmKa3ue2Uxu1LXvtzDDDYGBOb2jB/xpgpwBSwA9PcTlAHDkCOHFCy5O1sRWUW/1z+h9x+ucnpl5MPWnyAn48fwXcEuzsspTyKKzuGHQcSH25LACeSLVMHmCsih4FHgUki0t6FMfHoozBwoCu/QWUEYwwzQmdQeWJlhv0yDIB6xetpElDqFjjTs/hWbQHKiUgZ4G+gM/Bk4gWMMWXi34vIDGCxMWahC2O6/mbxN99AAR1z1pMcvnCYZxY/w7KDy7iv1H30qd3H3SEp5dFclgiMMTEiMhD7NJAP8JkxZreI9HXMn+yq705NVFSyRNCggTvCULfo273f0vXbrogIEx6YQL+6/cgmHlvxRKlMwZnqo+IYq3iY43MpEannzMaNMUuMMeWNMXcZY952TJucUhIwxvQwxri8kF3CFcErr9ibBQ8+CFu2uPpr1W0yjqHlqhSuQvOyzdnVbxcD6g3QJKBUOnDmf9EkoAEQP2rLJWz/AI+U0LN4xw47IE316lp1NBOLjo3mnbXv0GVBFwDKFyjPws4LuTPvnW6OTKmsw5mmofrGmFoi8juAMea8iPjdbKXM6sMPoXhx4F2gRAl45x13h6Ru4LeTv9FrUS9C/wnlsSqPcTXmKjl8c7g7LKWyHGcSQbSj96+BhPEI4lwalQs9/ri7I1A3ExEdwZur3+SD9R9QKGchvn38W9pXbO/usJTKspxJBOOAb4HCIvI29jHP11walQtt3AjFikGp55+HSC1DnBldib7CtN+n0b1Gd0a3HE2+AB1CVClXkvibcKkuJFIRaIbtJLbCGLPX1YHdSJ06dczWrVtvef0cOeDFF+G999IxKHXbLl29xMdbP+a/Df6LTzYfzoSfoWBgQXeHpVSWISLbjDF1Upp30ysCESkFhAPfJ55mjDmafiFmnISnhvbtsx+qVXN3SF7vpwM/8cziZzgWdox6xesRUjpEk4BSGciZpqEfsPcHBPAHygB/AFVcGJdLxMaCMY5E8N//wqlT+uioG50NP8ugZYOYuX0mlQpWYl3PdTQoqf06lMpozoxQluSUWURqAc+4LCIXio62P5N0KFNu0/Grjqw/tp7XG7/Oq41e1SeClHKTNPcsNsb8JiIZUjI6vUVF2Z9+Hvvwq+c7eekkuXPkJpdfLka3GI2fjx817qjh7rCU8mrO3CMYlOhjNqAWcNplEbmQvz/Mn++4LbDS3dF4F2MM00OnM2jpIHrW7MmHrT6kbnGPPJ9QKstx5oogd6L3Mdh7Bt+4JhzX8vODR3TM8gx36Pwhnln8DMsPLafxnY3pW6evu0NSSiWSaiJwdCTLZYx5KYPicamICFizxl4RFBsyBK5edXdIWd6CvQvo+m1XfMSHj9t+TJ/afbQ+kFKZzA0TgYj4OiqI1srIgFzp5Elo3RpmzIDu3Ru5O5wszRiDiFCtcDVa392aMa3GUDJIRwNSKjNK7YpgM/Z+QKiILAK+Bq7EzzTGLHBxbOkuyVNDv/9uJ9RzqpCqclJUbBSj1o1i9+ndzOk4h3IFyvHNYx7ZkqiU13DmHkF+4Cx2XOH4/gQG8OxE8Npr2o8gnW09sZVei3qx498ddK7amajYKH0kVCkPkFoiKOx4YmgX1xJAvNsaN9hdtB+Ba0RER/DGqjf4vw3/xx257uC7zt/xcIWH3R2WUspJqSUCHyAXzg1C7xE0EbjGlegrzAidQa+avRjVYhR5/fO6OySlVBqklghOGmPezLBIMkCFCvDTT1CrFna4HXXLLl69yKQtk3jp3pcoGFiQvQP2UiBQx35WyhOllghSuhLwaEFB0KqVu6PwfD/s/4G+P/TlxKUT3FPiHkJKh2gSUMqDpZYImmVYFBnkn3/g11+haVMo8Oab2o8gjU5fOc0LS19gzs45VClUhfmd5lO/RH13h6WUuk03TATGmHMZGUhG+O036NTJDk5ToH5td4fjcR756hE2Ht/I8CbDeaXRK/j5aNEmpbKCNBed82TxN4v9/ID16+0VQdOmbo0ps/v74t8E+QeRyy8XH7X6iBy+OahauKq7w1JKpSOv6usfFQU5iCRo+xoYNAheftndIWVaxhg+3fYplSdVZtgvwwCoXay2JgGlsiCvSgTR0fACYyj7dBPYtAly5775Sl7o4LmDNJvZjD6L+1C7aG0G1B3g7pCUUi7kdU1DubmEyZYNWbECKld2d0iZzvw98+n2bTey+2RnyoNT6F2rNyJZ7gEypVQiXpUI2rSBk5MfJCayONlDQtwdTqYSXySuRpEatC3flo9afUSJPCXcHZZSKgOIMZ7VSbhOnTpm69at7g4jy4iKjeLdte+y58we5j4yV8/+lcqiRGSbMaZOSvO86h7B7t0wb+w/RG3d4e5QMoXNf2+m9pTaDF89HN9svkTFRrk7JKWUG3hVIvjpJzjwwniyN/DuPgTh0eH8b9n/aDCtAecjzvP9E98zu+NsrRSqlJfyqnsEUXrCC9hqobN2zKJPrT683+J98uTI4+6QlFJu5NJEICKtgbHYSqZTjTHvJZvfBRjs+HgZ6GeM2e6qeKKjbSDeKCwyjAmbJzD4vsEUCCzA3gF7yReQz91hqUwsOjqa48ePExkZ6e5QVBr4+/tTokQJsqehzLLLEoFjvOOJQAvgOLBFRBYZY/YkWuwvoIkx5ryIPABMAVxWvCY62ssugRy+/+N7+v7Ql38u/0PDUg0JKR2iSUDd1PHjx8mdOzelS5fWhwg8hDGGs2fPcvz4ccqUKeP0eq68R1APOGCMOWSMiQLmAu0SL2CMWW+MOe/4uBFw6fOK0dGQLVsWLKt6A6evnOaJb57g4bkPUyCgAJt6byKkdIi7w1IeIjIykgIFCmgS8CAiQoECBdJ8FefKE+TiwLFEn4+T+tl+L+DHlGaISB+gD0CpUqVuOaAXX4RLtR6ByAq3vA1PEl8k7s2QNxl832AtEqfSTJOA57mVv5krE4HTI5uJSFNsIrgvpfnGmCnYZiPq1Klzyx0fihSBIo/VAmrd6iYyveMXj5PXPy+5/HIxpvUYcvjkoErhKu4OSymVibmyaeg4UDLR5xLAieQLiUh1YCrQzhhz1oXxsHIlzBt1xFYezWLiTByfbP2EyhMr8/rK1wGoVbSWJgHl0Xx8fAgODqZKlSrUqFGDDz/8kLi4uFva1rBhw1i+fPkN50+ePJmZM2feaqgA7Ny5k+DgYIKDg8mfPz9lypQhODiY5s2b39Z2Xc4Y45IX9mrjEFAG8AO2A1WSLVMKOADc6+x2a9eubW5Vr17GjM011Bhf31veRma0/8x+02R6E8NwTLPPm5mD5w66OySVBezZs8fdIZicOXMmvP/3339Ns2bNzLBhw9wYkfO6d+9uvv766+umR0dHu/y7U/rbAVvNDY6rLrsiMMbEAAOBpcBe4CtjzG4R6SsifR2LDQMKAJNEJFREXFo7IjoafLLY86Nf7/6a6pOrE/pPKNMensbPXX+mbL6y7g5LZUEhIde/JjnG/g4PT3n+jBl2/pkz189Lq8KFCzNlyhQmTJiAMYbY2Fheeukl6tatS/Xq1fnkk08Slh01ahTVqlWjRo0aDBkyBIAePXowf/58AIYMGULlypWpXr06//vf/wAYPnw4o0ePBiA0NJR77rmH6tWr06FDB86fP+/4HYQwePBg6tWrR/ny5Vm7dq2Tv7sQhg4dSpMmTRg7dizbtm2jSZMm1K5dm1atWnHy5EkADh48SOvWralduzaNGjVi3759af9F3QKXPk1pjFkCLEk2bXKi972B3q6MIbH4p4ayAuMoElezaE3aVWjHh60+pFjuYu4OSymXKlu2LHFxcZw6dYrvvvuOoKAgtmzZwtWrV2nYsCEtW7Zk3759LFy4kE2bNhEYGMi5c0kHWzx37hzffvst+/btQ0S4cOHCdd/TrVs3xo8fT5MmTRg2bBgjRoxgzJgxAMTExLB582aWLFnCiBEjUm1uSuzChQusXr2a6OhomjRpwnfffUehQoWYN28er776Kp999hl9+vRh8uTJlCtXjk2bNtG/f39Wrlx5u7+2m/Kqx+qjoyGbh18RXI25yttr32bvmb189ehX3J3/buY+OtfdYSkvsGrVjecFBqY+v2DB1OenhXEUyly2bBk7duxIOMsPCwvjzz//ZPny5Tz99NMEBgYCkD9//iTr58mTB39/f3r37k3btm158MEHk8wPCwvjwoULNGnSBIDu3bvTqVOnhPkdO3YEoHbt2hw+fNjpuB9//HEA/vjjD3bt2kWLFi0AiI2NpWjRoly+fJn169cn+a6rGTSuulclgqgo8PHgK4KNxzfSa1Ev9pzeQ9fqXYmKjdL6QMqrHDp0CB8fHwoXLowxhvHjx9OqVasky/z000+pPkLp6+vL5s2bWbFiBXPnzmXChAlpOuvOkcP+n/Px8SEmJsbp9XLmzAnYRFalShU2bNiQZP7FixfJmzcvoaGhTm8zvXjwYTHtpk6Fh758Er76yt2hpMmVqCu8+NOL3DvtXi5dvcSSJ5cws8NMTQLKq5w+fZq+ffsycOBARIRWrVrx8ccfE+0YjHz//v1cuXKFli1b8tlnnxEeHg5wXdPQ5cuXCQsLo02bNowZM+a6A29QUBD58uVLaP//4osvEq4O0kOFChU4ffp0QiKIjo5m9+7d5MmThzJlyvD1118DNmFs3+6yijtJeNUVQaFCwP1VAM96pDIyJpK5u+fSv25/3m32Lrlz6BCbyjtEREQQHBxMdHQ0vr6+dO3alUGDBgHQu3dvDh8+TK1atTDGUKhQIRYuXEjr1q0JDQ2lTp06+Pn50aZNG955552EbV66dIl27doRGRmJMYaPPvrouu/9/PPP6du3L+Hh4ZQtW5bp06en2z75+fkxf/58nnvuOcLCwoiJieGFF16gSpUqzJ49m379+jFy5Eiio6Pp3LkzNWrUSLfvvhGvGphm5kzIe/pPHq72F7Rsmc6Rpa8LkRcYv2k8rzR6Bd9svlyIvEBe/7zuDkt5kb1791KpUiV3h6FuQUp/Ox2YxuGTT+DShBnQtq27Q0nVwn0LqTyxMiNWj2D9Mdv5TZOAUspVvCoRREdn7pvF/17+l8e+fowO8zpQOGdhNvXeROM7G7s7LKVUFudV9wgyez+CR79+lM1/b2Zk05G83PBlsvs4X09cKaVuldclgszWs/ho2FHy+ecjd47cjGs9jhy+OahcqLK7w1JKeZFMfH6c/jLTFUGciWPi5olUmVSFYb8MA6Bm0ZqaBJRSGc6rrgi2bIFsh56GM03dGscfZ/6g9/e9+fXor7Qo24Ln73nerfEopbxbJjk/zhh58kCu4LvBjSVhv9r9FTUm12DXqV1MbzedpU8tpXTe0m6LR6nMLL4MddWqVXnooYdSrAt0K2bMmMHAgQPTZVuJhYSEUKFChYRS1PHlL9Lb4cOHmTNnTrptz6sSwVtvwcpxu+CbbzL8u+P7a9QuWpuOlTqyd8BeegT30BGglEpFQEAAoaGh7Nq1i/z58zNx4kR3h3RTs2fPJjQ0lNDQUB599FGn1klLqQpI/0TgVU1DH30E9cp+CdtH2RsGGSAyJpK3Vr/FvrP7mN9pPnflv4s5j6TfH1CpDJNS7ejHHoP+/W0d6jZtrp/fo4d9nTkDyQ+KaaxC16BBA3bs2AHA5s2beeGFF4iIiCAgIIDp06dToUIFZsyYwaJFiwgPD+fgwYN06NCBUaNGATB9+nTeffddihYtSvny5RNqBh05coSePXty+vRpChUqxPTp0ylVqhQ9evQgICCAffv2ceTIEaZPn87nn3/Ohg0bqF+/PjPia2zfxLlz5+jZsyeHDh0iMDCQKVOmUL16dYYPH86JEyc4fPgwBQsWZOzYsfTt25ejR48CMGbMGBo2bMjq1at5/nnbfCwirFmzhiFDhrB3716Cg4Pp3r07L774Ypp+l8l5VSLI6Oqj64+tp9eiXuw7s4/uNbprkTilblFsbCwrVqygV69eAFSsWJE1a9bg6+vL8uXLGTp0KN84rvRDQ0P5/fffyZEjBxUqVODZZ5/F19eXN954g23bthEUFETTpk2pWbMmAAMHDqRbt250796dzz77jOeee46FCxcCcP78eVauXMmiRYt46KGHWLduHVOnTqVu3bqEhoYSHBx8XaxdunQhICAAgBUrVjB8+HBq1qzJwoULWblyJd26dUuob7Rt2zZ+/fVXAgICePLJJ3nxxRe57777OHr0KK1atWLv3r2MHj2aiRMn0rBhQy5fvoy/vz/vvfceo0ePZvHixeny+/W6RJARHcouR11m6IqhTNg8gZJBJfmpy0+0urvVzVdUKjNzQx3q+FpDhw8fpnbt2gmlm8PCwujevTt//vknIpJQeA6gWbNmBAUFAVC5cmWOHDnCmTNnCAkJoVChQoAtCb1//34ANmzYwIIFCwDo2rUrL7/8csK2HnroIUSEatWqUaRIEapVqwZAlSpVOHz4cIqJYPbs2dSpc62Sw6+//pqQpO6//37Onj1LWFgYAA8//HBC0li+fDl79uxJWO/ixYtcunSJhg0bMmjQILp06ULHjh0pUaJEmn+PN+NV9wgy6vHRqNgo5u+Zz4C6A9jVb5cmAaVuUfw9giNHjhAVFZVwj+D111+nadOm7Nq1i++//57IyMiEdeKbfCBpqWhn78clXi5+W9myZUuy3WzZsjndrp9SPbf474gvTQ0QFxfHhg0bEu4v/P333+TOnZshQ4YwdepUIiIiuOeee1wyapnXJIK4OPtyVYeycxHnGL5qODFxMeQPyM/eAXsZ32a8VgpVKh0EBQUxbtw4Ro8eTXR0NGFhYRQvXhzAqbb6+vXrs2rVKs6ePUt0dHRCqWeAe++9l7lz7eBOs2fP5r777kvX2Bs3bszs2bMBWLVqFQULFiRPnjzXLdeyZUsmTJiQ8Dm++ejgwYNUq1aNwYMHU6dOHfbt20fu3Lm5dOlSusXoNYlAxF4RNJjxDDg5zqizvtnzDZUnVmbkmpEJReKC/IPS9TuU8nY1a9akRo0azJ07l5dffplXXnmFhg0bEhsbe9N1ixYtyvDhw2nQoAHNmzenVq1aCfPGjRvH9OnTqV69Ol988QVjx45N17iHDx/O1q1bqV69OkOGDOHzzz9Pcblx48YlLFe5cmUmT7aj+o4ZM4aqVatSo0YNAgICeOCBB6hevTq+vr7UqFEjxTLaaeVVZajT28lLJxn440AW7F1AzTtq8lm7zwi+I9jdYSmVLrQMtefSMtQ3cOUK9OsHWz/ZBk4+9nUzj81/jB/2/8B7zd5j8382axJQSnkkr3lq6PJlmDwZerVcACtH2Webb8GRC0fIH5Cf3DlyM/6B8QT4BlChYIX0DVYppTKQ11wRxD9ddqs3i+NMHOM3jafKpCq8/svrAATfEaxJQCnl8bzmiiA+EdzK46P7zuyj96LerDu2jtZ3t+bFe26vF59SSmUmXpcI0npFMHfXXLov7E4uv1zMbD+Tp6o/pfWBlFJZitckgthY8PcHn2zOPSUVZ+LIJtmoW6wunSp34v9a/h9FchVxcZRKKZXxvOYeQaVKEBEBlea8DmvW3HC5iOgIhiwfwiNfPYIxhrvy38WsjrM0CSjlBp5WhjomJoahQ4dSrly5hFLUb7/9drp/T3rzmkSQICAAGjRIcdbaI2sJ/iSY99e9T4GAAkTHZUyFUqVUyjytDPVrr73GiRMn2LlzJ6GhoaxduzZJHaTMymuahv74AyYMPcGwfOMoNLQPlC2bMO/S1UsMWT6ESVsnUSZvGX7u+jPNy7pv8BqlMqOQGSHXTXusymP0r9uf8Ohw2sy+vgx1j+Ae9AjuwZnwMzz6VdIy1Kt6rErT92f2MtTh4eF8+umnHD58GH9/fwBy587N8OHDATuGwIMPPsiuXbsAGD16NJcvX2b48OEcPHiQAQMGcPr0aQIDA/n000+pWLEiX3/9NSNGjMDHx4egoCDWrFnD7t27efrpp4mKiiIuLo5vvvmGcuXKpel3mZzXJIKTJ+H0gjUU4n3o1ynJvOi4aBb+sZAX6r/AyPtHktMv5w22opRyB08oQ33gwAFKlSpF7txpry/Wp08fJk+eTLly5di0aRP9+/dn5cqVvPnmmyxdupTixYsnNItNnjyZ559/ni5duhAVFeVUiY2b8ZpEEB0N97KeWP9AfKpX52z4WcZuGsuwJsPIH5CffQP2aYE4pVKR2hl8YPbAVOcXDCyY5isA8Mwy1PGmT5/O2LFjOXv2LOvXr7/hcpcvX2b9+vV06nTtBPXq1asANGzYkB49evDYY4/RsWNHwF4Zvf322xw/fpyOHTve9tUAuPgegYi0FpE/ROSAiAxJYb6IyDjH/B0iUiul7aSH6GhoyDouVanH1/sXUnlSZd799V02HNsAoElAqUzIk8pQ33333Rw9ejShKujTTz9NaGgoQUFBxMbG4uvrS1xcXMLy8THHxcWRN2/ehPLToaGh7N27F7Bn/yNHjuTYsWMEBwdz9uxZnnzySRYtWkRAQACtWrVi5cqVTu1XalyWCETEB5gIPABUBp4QkcrJFnsAKOd49QE+dlU8cRcvUyh3KE80O8lj8x+jZJ6SbP3PVhrd2chVX6mUSieeUIY6MDCQXr16MXDgwISDfGxsLFFRUQAUKVKEU6dOcfbsWa5evZowuliePHkoU6ZMQkzGGLZv3w7YEtT169fnzTffpGDBghw7doxDhw5RtmxZnnvuOR5++OGE+ya3w5VXBPWAA8aYQ8aYKGAu0C7ZMu2AmcbaCOQVkaKuCCb36YM81kn4JfAvRjUfxcbeG6lxRw1XfJVSygU8oQz122+/TdGiRalatSo1a9akUaNGdO/enWLFipE9e3aGDRtG/fr1efDBB6lYsWLCerNnz2batGnUqFGDKlWq8N133wHw0ksvUa1aNapWrUrjxo2pUaMG8+bNo2rVqgQHB7Nv3z66det2y/HGc1kZahF5FGhtjOnt+NwVqG+MGZhomcXAe8aYXx2fVwCDjTFbk22rD/aKgVKlStU+cuTILcW0/cTvBPj6U76wltZV6ma0DLXnSmsZalfeLE6pQS551nFmGYwxU4ApYMcjuNWAahSreaurKqVUluXKpqHjQMlEn0sAJ25hGaWUUi7kykSwBSgnImVExA/oDCxKtswioJvj6aF7gDBjzEkXxqSUSgNPG8FQ3drfzGVNQ8aYGBEZCCwFfIDPjDG7RaSvY/5kYAnQBjgAhANPuyoepVTa+Pv7c/bsWQoUKKAVdz2EMYazZ88m9Gx2lo5ZrJRKUXR0NMePH0/yjL7K/Pz9/SlRogTZs2dPMt1dN4uVUh4se/bslClTxt1hqAzgfdVHlVJKJaGJQCmlvJwmAqWU8nIed7NYRE4Dt9a1GAoCZ9IxHE+g++wddJ+9w+3s853GmEIpzfC4RHA7RGTrje6aZ1W6z95B99k7uGqftWlIKaW8nCYCpZTyct6WCKa4OwA30H32DrrP3sEl++xV9wiUUkpdz9uuCJRSSiWjiUAppbxclkwEItJaRP4QkQMiMiSF+SIi4xzzd4hIrZS240mc2Ocujn3dISLrRcTjx+m82T4nWq6uiMQ6Rs3zaM7ss4iEiEioiOwWkdUZHWN6c+LfdpCIfC8i2x377NFVjEXkMxE5JSK7bjA//Y9fxpgs9cKWvD4IlAX8gO1A5WTLtAF+xI6Qdg+wyd1xZ8A+3wvkc7x/wBv2OdFyK7Elzx91d9wZ8HfOC+wBSjk+F3Z33Bmwz0OB9x3vCwHnAD93x34b+9wYqAXsusH8dD9+ZcUrgnrAAWPMIWNMFDAXaJdsmXbATGNtBPKKSNGMDjQd3XSfjTHrjTHnHR83YkeD82TO/J0BngW+AU5lZHAu4sw+PwksMMYcBTDGePp+O7PPBsgtdtCEXNhEEJOxYaYfY8wa7D7cSLofv7JiIigOHEv0+bhjWlqX8SRp3Z9e2DMKT3bTfRaR4kAHYHIGxuVKzvydywP5RGSViGwTkW4ZFp1rOLPPE4BK2GFudwLPG2PiMiY8t0j341dWHI8gpaGUkj8j68wynsTp/RGRpthEcJ9LI3I9Z/Z5DDDYGBObRUbYcmaffYHaQDMgANggIhuNMftdHZyLOLPPrYBQ4H7gLuBnEVlrjLno4tjcJd2PX1kxERwHSib6XAJ7ppDWZTyJU/sjItWBqcADxpizGRSbqzizz3WAuY4kUBBoIyIxxpiFGRJh+nP23/YZY8wV4IqIrAFqAJ6aCJzZ56eB94xtQD8gIn8BFYHNGRNihkv341dWbBraApQTkTIi4gd0BhYlW2YR0M1x9/0eIMwYczKjA01HN91nESkFLAC6evDZYWI33WdjTBljTGljTGlgPtDfg5MAOPdv+zugkYj4ikggUB/Ym8Fxpidn9vko9goIESkCVAAOZWiUGSvdj19Z7orAGBMjIgOBpdgnDj4zxuwWkb6O+ZOxT5C0AQ4A4dgzCo/l5D4PAwoAkxxnyDHGgys3OrnPWYoz+2yM2SsiPwE7gDhgqjEmxccQPYGTf+e3gBkishPbbDLYGOOx5alF5EsgBCgoIseBN4Ds4Lrjl5aYUEopL5cVm4aUUkqlgSYCpZTycpoIlFLKy2kiUEopL6eJQCmlvJwmApUpOaqFhiZ6lU5l2cvp8H0zROQvx3f9JiINbmEbU0WksuP90GTz1t9ujI7txP9edjkqbua9yfLBItImPb5bZV36+KjKlETksjEmV3ovm8o2ZgCLjTHzRaQlMNoYU/02tnfbMd1suyLyObDfGPN2Ksv3AOoYYwamdywq69ArAuURRCSXiKxwnK3vFJHrKo2KSFERWZPojLmRY3pLEdngWPdrEbnZAXoNcLdj3UGObe0SkRcc03KKyA+O+ve7RORxx/RVIlJHRN4DAhxxzHbMu+z4OS/xGbrjSuQREfERkQ9EZIvYGvPPOPFr2YCj2JiI1BM7zsTvjp8VHD1x3wQed8TyuCP2zxzf83tKv0flhdxde1tf+krpBcRiC4mFAt9ie8HnccwriO1VGX9Fe9nx87/Aq473PkBux7JrgJyO6YOBYSl83wwc4xUAnYBN2OJtO4Gc2PLGu4GawCPAp4nWDXL8XIU9+06IKdEy8TF2AD53vPfDVpEMAPoArzmm5wC2AmVSiPNyov37Gmjt+JwH8HW8bw5843jfA5iQaP13gKcc7/NiaxDldPffW1/ufWW5EhMqy4gwxgTHfxCR7MA7ItIYWzqhOFAE+CfROluAzxzLLjTGhIpIE6AysM5RWsMPeyadkg9E5DXgNLZCazPgW2MLuCEiC4BGwE/AaBF5H9uctDYN+/UjME5EcgCtgTXGmAhHc1R1uTaKWhBQDvgr2foBIhIKlAa2AT8nWv5zESmHrUSZ/Qbf3xJ4WET+5/jsD5TCs+sRqdukiUB5ii7Y0adqG2OiReQw9iCWwBizxpEo2gJfiMgHwHngZ2PME058x0vGmPnxH0SkeUoLGWP2i0htbL2Xd0VkmTHmTWd2whgTKSKrsKWTHwe+jP864FljzNKbbCLCGBMsIkHAYmAAMA5bb+cXY0wHx431VTdYX4BHjDF/OBOv8g56j0B5iiDglCMJNAXuTL6AiNzpWOZTYBp2uL+NQEMRiW/zDxSR8k5+5xqgvWOdnNhmnbUiUgwIN8bMAkY7vie5aMeVSUrmYguFNcIWU8Pxs1/8OiJS3vGdKTLGhAHPAf9zrBME/O2Y3SPRopewTWTxlgLPiuPySERq3ug7lPfQRKA8xWygjohsxV4d7EthmRAgVER+x7bjjzXGnMYeGL8UkR3YxFDRmS80xvyGvXewGXvPYKox5negGrDZ0UTzKjAyhdWnADvibxYnsww7Lu1yY4dfBDtOxB7gN7GDln/CTa7YHbFsx5ZmHoW9OlmHvX8Q7xegcvzNYuyVQ3ZHbLscn5WX08dHlVLKy+kVgVJKeTlNBEop5eU0ESillJfTRKCUUl5OE4FSSnk5TQRKKeXlNBEopZSX+3/oLS1FxcbDaAAAAABJRU5ErkJggg=="/>

#### -> 의사결정나무보다 랜덤 포레스트가 좀 더 모델 성능이 좋다



```python
# 두 모델의 AUC 계산 (ROC 선으로 보기보다 정량적으로 보기)

from sklearn import metrics

roc_auc_DT=metrics.auc(fpr1, tpr1)
roc_auc_RF=metrics.auc(fpr2, tpr2)
```


```python
# 두 모델의 Roc curve 그래프 연혁에 AUC도 출력

plt.plot(fpr1, tpr1, 'b--', label="Decision Tree(area=%0.2f)" % roc_auc_DT)
plt.plot(fpr2, tpr2, 'r--', label="Random Forest(area=%0.2f)" % roc_auc_RF)

#최악의 모형을 비교 차원에서 그리도록 함
plt.plot([0,1], [0,1], 'g--', label="Random Guess")

# 타이틀
plt.title('Roc Curves')

#X축 이름
plt.xlabel('False Positive Rate')

#Y축 이름
plt.ylabel('True Positive Rate')

# 연혁 하단부에 위치
plt.legend(loc="lower right")
plt.show()
```

<img src="data:image/png;base64,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"/>

#### -> 랜덤 포레스트로 했을 시 의사결정나무보다 1% 정도의 성능 향상



```python
```
