---
layout: single
title:  "2nd Week Course"
categories: coding
tag: [python, blog, jupyter, Titanic Task]
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


# 2주차 강의



```python
import numpy as np
import pandas as pd
```


```python
df_train=pd.read_csv("./data/train.csv")
df_train
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
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
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
      <th>1</th>
      <td>2</td>
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
      <th>2</th>
      <td>3</td>
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
      <th>3</th>
      <td>4</td>
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
      <th>4</th>
      <td>5</td>
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



```python
df_train.head()
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
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
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
      <th>1</th>
      <td>2</td>
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
      <th>2</th>
      <td>3</td>
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
      <th>3</th>
      <td>4</td>
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
      <th>4</th>
      <td>5</td>
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



```python
df_train.tail()
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
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>



```python
# describe는 어떤 속성값을 가지고 있는지 알려준다
# 운임의 속성정보 보기

df_train['Fare'].describe()
```

<pre>
count    891.000000
mean      32.204208
std       49.693429
min        0.000000
25%        7.910400
50%       14.454200
75%       31.000000
max      512.329200
Name: Fare, dtype: float64
</pre>
### -> 중간값(50%)과 평균값(mean)의 차이가 많이 나는데, 평균이 어떤 이상치에 의해서 높여졌다는 걸 예측해볼 수 있다.



```python
df_train[df_train['Fare']>500]
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
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>


### -> 운임의 최댓값이 500 즈음이었으니 운임을 많이 내고 타는 사람들은 어떤 사람들이 있는지 확인



```python
df_train[df_train['Fare']==0]
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
      <th>PassengerId</th>
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
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0</td>
      <td>B94</td>
      <td>S</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0</td>
      <td>A36</td>
      <td>S</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


### -> 운임이 0원이었던 사람들은 어떤 사람들인지 확인

### -> 운임을 내지 않고 탑승한 사람들은 거의 생존하지 못했다.


 


## EDA(Exploratory Data Analytics) 실습

-  Titanic 데이터 활용

- https://www.kaggle.com/c/titanic

- 분석 목적: 승객의 정보를 보고 타이타닉에서 생존했을지를 분류



```python
from IPython.display import Image
Image("data/Titanic data info.PNG")
```

<img src="data:image/png;base64,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"/>

## 운임과 관련이 있을 만한 변수였던 탑승장소를 같이 시각화



```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
data = pd.concat([df_train['Fare'], df_train['Embarked']], axis=1) #운임과 탑승항구 데이터 연결(묶음)
f, ax = plt.subplots(figsize=(8,6)) #한번에 여러 그래프를 보여줌(subplots)
fig=sns.boxplot(x='Embarked', y='Fare', data=data)
```

<img src="data:image/png;base64,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"/>

### -> C 값은 경우 이상치 값이 폭넓게 분포한다.

### -> 탑승항구의 종류에 따라서 운임에 차이가 나는 것도 볼 수 있다



```python
f, ax=plt.subplots(figsize=(8,6))
sns.boxplot(x='Embarked', y='Fare', hue='Pclass', # hue에 따라 색깔이 서로 다르게 나타남
           data=df_train, palette='Set3')
```

<pre>
<AxesSubplot:xlabel='Embarked', ylabel='Fare'>
</pre>
<img src="data:image/png;base64,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"/>

### -> 각 항구별로 또 티켓등급(세 등급)에 따라서는 또 어떤 차이가 있는지

### -> 위에서는 S와 Q의 차이가 명확하게 보여지지 않았는데 Pclass로 나누니 차이가 보임

### -> 1등급이라고 하더라도 운임차이가 나는 걸 알 수 있다


 


 


# 2주차 과제(단톡)

### 60191315 박온지

- 임의의 데이터를 구해서 EDA수행하기

- 어떤 데이터를 어떠한 목적에 의해 어떠한 것들을 살펴봤는지를 위주로 설명을 넣을 것(정해진 형식은 없음)

- 자신이 알고있거나 검색 등을 통하여 데이터를 살펴보고 시각화할 것


## Top 100 Korean Drama 데이터 활용

#### https://www.kaggle.com/chanoncharuchinda/top-100-korean-drama-mydramalist/version/6


### 분석 목적

- 1. TV를 보는 인구가 줄어든 시대를 살고있는 본인은 주로 넷플릭스와 유튜브, 네이버TV로 드라마를 시청한다. TV 시청률로는 반응이 저조했어도, 넷플릭스에서는 흥행하는 드라마들을 여럿 목도하면서 실제로 TV인구보다 OTT서비스를 이용하는 고객이 많이 높아졌는지 데이터를 통해 확인해 보고 싶었다. 평가 변수를 통해 알아보도록 하자. 

- 2. 요즘 넷플릭스는 12회 시즌드라마가 많고, tv드라마는 주로 16회 회차를 유지하는 한편, 드라마 회차에 따라서도 평가도가 달라지는지 확인해보자.


### 변수설명


#### Name: 한국 드라마 이름

#### Year of release : 드라마 개봉연도

#### Aired Date : 방영일(시작) - (종료)

#### Aired On : 요일에 방영

#### Number of Episode: 에피소드 수

#### Network: 방영 중인 드라마 네트워크

#### Duration: 한 에피소드의 길이

#### Content Rating : 적절한 청중을 위한 콘텐츠 등급

#### Synopsis: 드라마의 단편

#### Cast : 드라마 속 배우

#### Genre: 드라마가 수록된 장르

#### Tags: 드라마가 등록된 태그

#### Rank : 웹사이트에서의 순위

#### Rating: 웹 사이트 사용자가 10점 만점에 평가



```python
df_train=pd.read_csv("data/top100_kdrama.csv")
print(df.shape)
df_train.head()
```

<pre>
(100, 14)
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
      <th>Name</th>
      <th>Year of release</th>
      <th>Aired Date</th>
      <th>Aired On</th>
      <th>Number of Episode</th>
      <th>Network</th>
      <th>Duration</th>
      <th>Content Rating</th>
      <th>Synopsis</th>
      <th>Cast</th>
      <th>Genre</th>
      <th>Tags</th>
      <th>Rank</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Move to Heaven</td>
      <td>2021</td>
      <td>May 14, 2021</td>
      <td>Friday</td>
      <td>10</td>
      <td>Netflix</td>
      <td>52 min.</td>
      <td>18+ Restricted (violence &amp; profanity)</td>
      <td>Geu Roo is a young autistic man. He works for ...</td>
      <td>Lee Je Hoon, Tang Jun Sang, Hong Seung Hee, Ju...</td>
      <td>Life,  Drama,  Family</td>
      <td>Autism, Uncle-Nephew Relationship, Death, Sava...</td>
      <td>#1</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hospital Playlist</td>
      <td>2020</td>
      <td>Mar 12, 2020 - May 28, 2020</td>
      <td>Thursday</td>
      <td>12</td>
      <td>Netflix,  tvN</td>
      <td>1 hr. 30 min.</td>
      <td>15+ - Teens 15 or older</td>
      <td>The stories of people going through their days...</td>
      <td>Jo Jung Suk, Yoo Yeon Seok, Jung Kyung Ho, Kim...</td>
      <td>Friendship,  Romance,  Life,  Medical</td>
      <td>Strong Friendship, Multiple Mains, Best Friend...</td>
      <td>#2</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flower of Evil</td>
      <td>2020</td>
      <td>Jul 29, 2020 - Sep 23, 2020</td>
      <td>Wednesday, Thursday</td>
      <td>16</td>
      <td>tvN</td>
      <td>1 hr. 10 min.</td>
      <td>15+ - Teens 15 or older</td>
      <td>Although Baek Hee Sung is hiding a dark secret...</td>
      <td>Lee Joon Gi, Moon Chae Won, Jang Hee Jin, Seo ...</td>
      <td>Thriller,  Romance,  Crime,  Melodrama</td>
      <td>Married Couple, Deception, Suspense, Family Se...</td>
      <td>#3</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hospital Playlist 2</td>
      <td>2021</td>
      <td>Jun 17, 2021 - Sep 16, 2021</td>
      <td>Thursday</td>
      <td>12</td>
      <td>Netflix,  tvN</td>
      <td>1 hr. 40 min.</td>
      <td>15+ - Teens 15 or older</td>
      <td>Everyday is extraordinary for five doctors and...</td>
      <td>Jo Jung Suk, Yoo Yeon Seok, Jung Kyung Ho, Kim...</td>
      <td>Friendship,  Romance,  Life,  Medical</td>
      <td>Workplace, Strong Friendship, Best Friends, Mu...</td>
      <td>#4</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>My Mister</td>
      <td>2018</td>
      <td>Mar 21, 2018 - May 17, 2018</td>
      <td>Wednesday, Thursday</td>
      <td>16</td>
      <td>tvN</td>
      <td>1 hr. 17 min.</td>
      <td>15+ - Teens 15 or older</td>
      <td>Park Dong Hoon is a middle-aged engineer who i...</td>
      <td>Lee Sun Kyun, IU, Park Ho San, Song Sae Byuk, ...</td>
      <td>Psychological,  Life,  Drama,  Family</td>
      <td>Age Gap, Nice Male Lead, Strong Female Lead, H...</td>
      <td>#5</td>
      <td>9.1</td>
    </tr>
  </tbody>
</table>
</div>



```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## 방영창구(Network)에 따른 평가도(Rating)를 시각화



```python
data = pd.concat([df_train['Network'], df_train['Rating']], axis=1)
f, ax = plt.subplots(figsize=(15,10)) 
fig=sns.boxplot(x='Network', y='Rating', data=data)
```

<img src="data:image/png;base64,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"/>

### -> MBC는 값의 이상치가 폭넓게 존재했다.

### -> 앱·리테일 분석서비스 와이즈앱·리테일·굿즈에 따르면 2021년 7월 기준 국내 스마트폰 사용자들이 가장 많이 사용한 OTT 앱은, 넷플릭스(910만명), 티빙(278만명) 인 걸로 나타났다. 그래프를 통해서도 요즘 잘 나가는 채널인 넷플릭스와 tvN의 Rating이 높은 것을 확인할 수 있다.

### -> 주요 TV 채널인 KBS2, SBS, MBC는 넷플릭스보다 낮을 걸로 보아 역시나 이 그래프를 통해서도 TV인구가 OTT 서비스로 넘어갔음을 확인할 수 있다


 


## 드라마 회차에 따라 평가가 달라지는지 시각화



```python
data = pd.concat([df_train['Number of Episode'], df_train['Rating']], axis=1)
f, ax = plt.subplots(figsize=(15,10)) 
fig=sns.boxplot(x='Number of Episode', y='Rating', data=data)
```

<img src="data:image/png;base64,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"/>


```python
df_train['Number of Episode'].describe()
```

<pre>
count    100.000000
mean      19.070000
std       12.378096
min        6.000000
25%       16.000000
50%       16.000000
75%       20.000000
max      100.000000
Name: Number of Episode, dtype: float64
</pre>
### -> 12회차를 방영하는 드라마가 가장 평가 측면에서 높은 점수를 보여준다.

### -> 32, 40 회차가 평가도 측면에서 낮았는데, 이는 흔한 회차수는 아니므로 하나 혹은 적은 드라마 수가 영향을 미쳤을 것으로 판단된다.

### -> 흔한 회차수인 16은 흔하기에 평가의 점수도 다양하여 다소 낮은 그래프 모습을 보여주는 것으로 판단된다.



```python
```
