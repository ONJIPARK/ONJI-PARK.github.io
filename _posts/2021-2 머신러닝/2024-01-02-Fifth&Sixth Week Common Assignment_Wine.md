---
layout: single
title:  "Fifth&Sixth Week Common Assignment"
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

## 와인 데이터로 의사결정나무 만들기

- 문제정의: 와인의 여러 속성들을 활용하여 와인의 품질을 분류하는 모델을 만들자


### 데이터셋 로딩

- 데이터는 캐글 사이트에 다운로드 가능(https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv)

- 현재 폴더에 data라는 이름의 하위 폴더를 만든 뒤 winequality-red.csv를 저장



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
df=pd.read_csv("data/winequality-red.csv")
print(df.shape)
df.head()
```

<pre>
(1599, 12)
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
#결측치 확인
df.isnull().sum()
```

<pre>
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64
</pre>

```python
df['quality'].value_counts()
```

<pre>
5    681
6    638
7    199
4     53
8     18
3     10
Name: quality, dtype: int64
</pre>

```python
#품질에서 3,4,5,6는 나쁨-> 0으로 지정/ 7,8은 좋음->1로 지정하여 매핑해줌
df['quality']=df.quality.map({3:0, 4:0, 5:0, 6:0, 7:1, 8:1})
```


```python
#Input 변수와 Output 변수 구분(생존 변수만 y 변수)
X=np.array(df.iloc[:, :-1]) # 모든 행에 대하여 맨 뒷열 데이터 전까지 가져오기
y=np.array(df['quality'])   #y변수에는 quality만
```


```python
X
```

<pre>
array([[ 7.4  ,  0.7  ,  0.   , ...,  3.51 ,  0.56 ,  9.4  ],
       [ 7.8  ,  0.88 ,  0.   , ...,  3.2  ,  0.68 ,  9.8  ],
       [ 7.8  ,  0.76 ,  0.04 , ...,  3.26 ,  0.65 ,  9.8  ],
       ...,
       [ 6.3  ,  0.51 ,  0.13 , ...,  3.42 ,  0.75 , 11.   ],
       [ 5.9  ,  0.645,  0.12 , ...,  3.57 ,  0.71 , 10.2  ],
       [ 6.   ,  0.31 ,  0.47 , ...,  3.39 ,  0.66 , 11.   ]])
</pre>

```python
y
```

<pre>
array([0, 0, 0, ..., 0, 0, 0], dtype=int64)
</pre>

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
X_train 크기: (1119, 11)
y_train 크기: (1119,)
X_test 크기: (480, 11)
y_test 크기: (480,)
</pre>

```python
#의사결정나무모델에 데이터 적합(fitting) 피팅은 학습시키는 것 학습시키는 모델이 의사결정나무모델
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(random_state=0, max_depth=6) #depth가 깊어질수록 모델이 복잡해지고 정확도가 올라감
tree.fit(X_train, y_train)
```

<pre>
DecisionTreeClassifier(max_depth=6, random_state=0)
</pre>

```python
temp_y_pred=tree.predict(X_test)
#Training값을 X_test에서 넣음
print('예측값\n', temp_y_pred)
print('실제값\n', y_test)
```

<pre>
예측값
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 1
 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0
 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0
 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
실제값
 [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0
 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0
 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0
 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0
 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
</pre>

```python
#정확도 계산
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

temp_acc=accuracy_score(y_test, temp_y_pred)
#y_test와 temp_y_pred로 예측한 값을 적용하여 temp_acc로 추출함.

print('정확도:', format(temp_acc))
```

<pre>
정확도: 0.9041666666666667
</pre>

```python
#오차 행렬
print(confusion_matrix(y_test, temp_y_pred))
```

<pre>
[[406  24]
 [ 22  28]]
</pre>

```python
from sklearn.metrics import precision_score, recall_score, f1_score
 # 정확도, 정밀도, 재현율
print('accuracy: ', accuracy_score(y_test, temp_y_pred))  
print('precision: ', precision_score(y_test, temp_y_pred))
print('recall: ', recall_score(y_test, temp_y_pred))
print('f1: ', f1_score(y_test, temp_y_pred))
```

<pre>
accuracy:  0.9041666666666667
precision:  0.5384615384615384
recall:  0.56
f1:  0.5490196078431373
</pre>

```python
#모델 성능을 보여주는 classification_report
print(classification_report(y_test, tree.predict(X_test)))

# 이렇게 해도 됨  > 0.5는 X_test를 정수형으로 보이게 하려는 것.?
# print(classification_report(y_test, (tree.predict(X_test) > 0.5).astype("int16")))
```

<pre>
              precision    recall  f1-score   support

           0       0.95      0.94      0.95       430
           1       0.54      0.56      0.55        50

    accuracy                           0.90       480
   macro avg       0.74      0.75      0.75       480
weighted avg       0.91      0.90      0.90       480

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
>1, train: 0.851, test: 0.896
>2, train: 0.876, test: 0.912
>3, train: 0.885, test: 0.865
>4, train: 0.907, test: 0.896
>5, train: 0.926, test: 0.875
>6, train: 0.950, test: 0.912
>7, train: 0.964, test: 0.896
>8, train: 0.978, test: 0.892
>9, train: 0.991, test: 0.894
>10, train: 0.996, test: 0.887
>11, train: 0.998, test: 0.879
>12, train: 1.000, test: 0.877
>13, train: 1.000, test: 0.885
>14, train: 1.000, test: 0.890
>15, train: 1.000, test: 0.887
>16, train: 1.000, test: 0.894
>17, train: 1.000, test: 0.896
>18, train: 1.000, test: 0.885
>19, train: 1.000, test: 0.883
</pre>

```python
from matplotlib import pyplot

pyplot.plot(range(1,20), train_scores, '-o', label='Train_acc')
pyplot.plot(range(1,20), test_scores, '-o', label='Test_acc')
pyplot.legend()
pyplot.show()

#최적의 의사결정나무 깊이는?
#훈련데이터의 경우 12정도
#테스트 데이터의 경우 2랑 6정도
```

<img src="data:image/png;base64,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"/>


```python
#트리 시각화, 와인 등급 분류 의사결정나무
import graphviz
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(random_state=0, max_depth=6)
tree.fit(X_train, y_train)

#속성들
feature_name=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
              'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#좋은 등급인지 나쁜 등급인지
treeviz = export_graphviz(tree, feature_names=feature_name,
                      class_names=['low_qual', 'high_qual'])
graphviz.Source(treeviz)
```

<pre>
<graphviz.files.Source at 0x2432b31ff40>
</pre>
- alcohol이 11.55 이하인 경우 low qual 

-  alcohol이 11.55 이하이고 volatile acidity 가 0.375 이하인 경우에는 low_qual

- alcohol이 11.55 이상이고 sulphates가 0.685 이하이며 pH가 3.275 이하이고 chlorides 가 0.109 이하인 경우 high_qual


# 8주차 와인 품질 분류모델의 성능평가와 분류 함수 만들기

- 60191315 박온지

- 와인 품질 데이터를 이용하여 만든 분류 모델의 성능을 평가하시오.  


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
accuracy:  0.9041666666666667
precision:  0.5384615384615384
recall:  0.56
f1:  0.5490196078431373
</pre>
### ROC curve와 AUC



```python
tree.predict(X_test)
```

<pre>
array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
       0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)
</pre>

```python
tree.predict_proba(X_test)
```

<pre>
array([[0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.94871795, 0.05128205],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [0.94871795, 0.05128205],
       [0.6       , 0.4       ],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.71428571, 0.28571429],
       [0.96103896, 0.03896104],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.        , 1.        ],
       [0.6       , 0.4       ],
       [0.98958333, 0.01041667],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.71428571, 0.28571429],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.98958333, 0.01041667],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.75      , 0.25      ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [0.82758621, 0.17241379],
       [0.18181818, 0.81818182],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [0.14285714, 0.85714286],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.6       , 0.4       ],
       [1.        , 0.        ],
       [0.14285714, 0.85714286],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.82758621, 0.17241379],
       [0.94871795, 0.05128205],
       [0.14285714, 0.85714286],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.6       , 0.4       ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.18181818, 0.81818182],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.98958333, 0.01041667],
       [0.96103896, 0.03896104],
       [0.94871795, 0.05128205],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.14285714, 0.85714286],
       [0.6       , 0.4       ],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.94871795, 0.05128205],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.66666667, 0.33333333],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.94871795, 0.05128205],
       [0.96888889, 0.03111111],
       [0.6       , 0.4       ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.68421053, 0.31578947],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [0.82758621, 0.17241379],
       [0.98958333, 0.01041667],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.6       , 0.4       ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.68421053, 0.31578947],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.68421053, 0.31578947],
       [0.18181818, 0.81818182],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [0.68421053, 0.31578947],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [0.        , 1.        ],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.98958333, 0.01041667],
       [0.96888889, 0.03111111],
       [0.71428571, 0.28571429],
       [0.66666667, 0.33333333],
       [0.82758621, 0.17241379],
       [0.14285714, 0.85714286],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [0.82758621, 0.17241379],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [0.6       , 0.4       ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.6       , 0.4       ],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.68421053, 0.31578947],
       [0.96888889, 0.03111111],
       [0.14285714, 0.85714286],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.6       , 0.4       ],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [0.94871795, 0.05128205],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.18181818, 0.81818182],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [0.98958333, 0.01041667],
       [0.        , 1.        ],
       [0.14285714, 0.85714286],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [0.14285714, 0.85714286],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [0.82758621, 0.17241379],
       [0.82758621, 0.17241379],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [0.33333333, 0.66666667],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [0.33333333, 0.66666667],
       [0.82758621, 0.17241379],
       [0.18181818, 0.81818182],
       [0.66666667, 0.33333333],
       [0.6       , 0.4       ],
       [0.66666667, 0.33333333],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.82758621, 0.17241379],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [0.14285714, 0.85714286],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [0.94871795, 0.05128205],
       [0.96103896, 0.03896104],
       [0.96103896, 0.03896104],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96103896, 0.03896104],
       [0.96103896, 0.03896104],
       [0.71428571, 0.28571429],
       [0.96888889, 0.03111111],
       [0.18181818, 0.81818182],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.        , 1.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.82758621, 0.17241379],
       [0.94871795, 0.05128205],
       [0.14285714, 0.85714286],
       [0.98958333, 0.01041667],
       [0.98958333, 0.01041667],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.        , 1.        ],
       [0.        , 1.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [0.68421053, 0.31578947],
       [0.94871795, 0.05128205],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96103896, 0.03896104],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.98958333, 0.01041667],
       [1.        , 0.        ],
       [0.96888889, 0.03111111],
       [0.96888889, 0.03111111]])
</pre>

```python
# 차례대로 비율을 보면 위 X_test가 왜 0 1이 나왔는지 알 수 있음. 1에 가까우면 좋음 0에 가까우면 나쁨
tree.predict_proba(X_test)[:,1]
```

<pre>
array([0.17241379, 0.03111111, 1.        , 0.05128205, 0.03111111,
       0.03111111, 0.03111111, 0.        , 0.03111111, 0.        ,
       0.03111111, 0.03111111, 0.03111111, 0.05128205, 0.05128205,
       0.4       , 1.        , 0.03111111, 0.        , 0.03111111,
       1.        , 0.03111111, 0.03111111, 0.        , 0.03896104,
       0.        , 0.03111111, 0.        , 0.        , 0.28571429,
       0.03896104, 0.17241379, 0.        , 0.01041667, 1.        ,
       0.4       , 0.01041667, 0.01041667, 0.        , 0.03111111,
       0.        , 0.03111111, 0.28571429, 0.        , 0.03111111,
       0.        , 0.03111111, 0.        , 0.03111111, 0.        ,
       0.03111111, 0.03896104, 0.        , 0.        , 0.03896104,
       0.        , 0.        , 0.03111111, 1.        , 0.01041667,
       0.01041667, 0.        , 0.        , 0.        , 0.03111111,
       0.17241379, 0.03111111, 0.17241379, 0.        , 0.        ,
       0.        , 0.25      , 0.03111111, 0.        , 0.        ,
       0.        , 0.03111111, 0.03896104, 0.17241379, 0.81818182,
       0.        , 0.        , 0.17241379, 0.85714286, 0.        ,
       0.03111111, 0.        , 0.03111111, 0.03111111, 0.        ,
       0.        , 0.        , 0.03111111, 0.03111111, 0.4       ,
       0.        , 0.85714286, 0.03111111, 0.03111111, 0.03896104,
       1.        , 0.        , 0.17241379, 0.        , 0.03111111,
       0.03111111, 0.03111111, 0.05128205, 0.        , 0.01041667,
       0.17241379, 0.05128205, 0.85714286, 0.        , 0.03111111,
       0.01041667, 0.03111111, 1.        , 0.03111111, 0.03111111,
       0.        , 0.        , 0.01041667, 0.        , 0.        ,
       0.4       , 0.        , 0.        , 0.03896104, 0.81818182,
       0.03896104, 0.03111111, 0.03111111, 0.03111111, 0.03896104,
       0.        , 0.01041667, 0.01041667, 0.03896104, 0.05128205,
       0.        , 0.01041667, 0.03111111, 0.01041667, 0.01041667,
       0.        , 0.85714286, 0.4       , 0.17241379, 0.        ,
       0.        , 0.        , 0.05128205, 0.03111111, 0.01041667,
       0.        , 0.33333333, 0.03111111, 0.        , 0.        ,
       0.05128205, 0.03111111, 0.4       , 0.        , 0.03111111,
       0.03111111, 0.17241379, 0.        , 0.        , 0.17241379,
       0.        , 0.01041667, 0.03111111, 1.        , 0.        ,
       0.31578947, 0.        , 1.        , 1.        , 0.03111111,
       0.        , 1.        , 0.        , 0.03896104, 1.        ,
       0.03111111, 0.        , 0.17241379, 0.        , 1.        ,
       0.17241379, 0.01041667, 0.03111111, 0.03111111, 0.        ,
       0.        , 0.03111111, 0.03111111, 0.03111111, 0.01041667,
       0.        , 0.17241379, 0.        , 0.03111111, 0.05128205,
       0.        , 0.03111111, 0.        , 0.4       , 0.        ,
       0.        , 0.31578947, 0.01041667, 0.        , 0.03111111,
       0.01041667, 0.03896104, 0.        , 0.        , 0.        ,
       0.03111111, 0.03111111, 0.31578947, 0.81818182, 0.03111111,
       0.03896104, 0.31578947, 0.17241379, 0.03111111, 0.03111111,
       0.        , 0.        , 0.03111111, 0.17241379, 1.        ,
       0.        , 0.        , 1.        , 1.        , 0.17241379,
       0.03111111, 0.03111111, 0.03111111, 0.03111111, 0.        ,
       0.        , 0.03896104, 0.        , 0.01041667, 1.        ,
       0.        , 0.        , 0.03111111, 0.03111111, 0.05128205,
       0.01041667, 0.        , 0.03111111, 0.        , 0.01041667,
       0.        , 0.        , 0.        , 0.03111111, 0.03111111,
       0.05128205, 0.03111111, 0.        , 0.01041667, 0.        ,
       0.03896104, 0.01041667, 0.03111111, 0.28571429, 0.33333333,
       0.17241379, 0.85714286, 0.        , 0.        , 0.17241379,
       0.        , 0.        , 0.01041667, 0.17241379, 0.03896104,
       0.        , 0.03896104, 0.03111111, 0.4       , 0.        ,
       0.        , 0.        , 0.03896104, 0.03111111, 0.        ,
       0.        , 0.03896104, 0.03111111, 0.03111111, 0.        ,
       0.        , 0.03111111, 0.05128205, 0.03896104, 0.        ,
       0.17241379, 0.03111111, 0.03111111, 0.03896104, 0.        ,
       0.4       , 0.17241379, 0.03111111, 0.03111111, 0.        ,
       0.31578947, 0.03111111, 0.85714286, 0.03111111, 1.        ,
       0.        , 0.03896104, 0.4       , 1.        , 0.03111111,
       0.05128205, 1.        , 0.03111111, 0.        , 0.17241379,
       1.        , 0.        , 0.03896104, 1.        , 0.03111111,
       0.        , 0.81818182, 0.01041667, 0.        , 0.01041667,
       0.        , 0.03111111, 0.03111111, 0.        , 0.        ,
       0.03111111, 0.01041667, 0.01041667, 1.        , 0.85714286,
       0.        , 0.        , 0.03111111, 0.03896104, 1.        ,
       0.03111111, 0.17241379, 1.        , 0.        , 1.        ,
       0.03111111, 1.        , 0.03111111, 0.        , 0.17241379,
       0.03111111, 0.01041667, 0.85714286, 0.03111111, 0.03111111,
       0.03111111, 0.        , 0.03111111, 0.        , 0.03111111,
       0.03111111, 0.        , 1.        , 0.        , 0.17241379,
       0.17241379, 0.17241379, 0.        , 0.        , 0.03896104,
       0.03111111, 0.66666667, 0.03896104, 0.        , 0.03111111,
       0.        , 0.03896104, 0.        , 0.66666667, 0.17241379,
       0.81818182, 0.33333333, 0.4       , 0.33333333, 0.        ,
       0.01041667, 0.        , 0.03111111, 0.        , 0.17241379,
       0.03111111, 0.17241379, 0.85714286, 0.03111111, 1.        ,
       0.01041667, 0.        , 0.        , 1.        , 0.05128205,
       0.03896104, 0.03896104, 0.        , 0.        , 0.        ,
       0.        , 0.01041667, 0.        , 0.03896104, 0.03896104,
       0.28571429, 0.03111111, 0.81818182, 0.        , 0.        ,
       0.01041667, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.03111111, 1.        , 0.03111111, 0.03111111,
       0.        , 0.03111111, 0.17241379, 0.05128205, 0.85714286,
       0.01041667, 0.01041667, 1.        , 0.        , 0.03111111,
       0.        , 0.        , 1.        , 1.        , 0.        ,
       0.        , 0.03111111, 0.03111111, 0.03896104, 0.03111111,
       0.03111111, 0.31578947, 0.05128205, 0.        , 0.        ,
       0.03111111, 0.03896104, 0.03111111, 0.03111111, 0.        ,
       0.        , 0.03111111, 0.        , 0.        , 0.        ,
       0.03111111, 0.01041667, 0.        , 0.03111111, 0.03111111])
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

<img src="data:image/png;base64,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g0e8rYceeqjG5RMmHPudee/evSvivfbaaxk9ejQXXXTRQeuUlJQQFRVYp97AiqYOlA8mq7N5CIoLndpAu9fDJW9A445w0at1s29jasmDH6xi9facWt1mj5aJ/G1MT6/Xb9asGVOnTuX4449n0qRJlJWVcc8997BgwQIOHDjArbfeyk033QTAE088wRtvvEFERAQjR47k8ccfP+jEfM899zBv3jyioqIYPnw4Tz31FJMmTSI+Pp4///nPpKWlMWHCBPLz8+nUqRPTpk2jUaNGDB06lEGDBvHVV1+xb98+Xn31VU455fDzgA8dOpSTTjqJb7/9lnPPPZehQ4dy5513kpeXR9OmTZkxYwYpKSls3LiRW2+9lV27dlG/fn1efvllunXz/XPDsEsEdToPwW/fwbyJThLoO84pEhcVAD2VjAlSHTt2pKysjJ07d/L++++TlJTEsmXLOHDgAEOGDGH48OGsXbuWuXPn8t1331G/fn327Nlz0Db27NnDe++9x9q1axER9u3bd8h+rr76ap5//nlOO+00HnjgAR588EGeeeYZwLmiX7p0KR9//DEPPvhgjc1Nnvbt28fXX39NcXExp512Gu+//z7Jycm8/fbb3HfffUybNo3x48czZcoUunTpwnfffcctt9zCl19+eax/bYcVdomgfGYynz4sPpAHXzzklIhIag1XzoHOgd1GaExNjuTK3ddUnWnLP/vsM1asWMHs2bMByM7O5pdffmH+/Plcd9111K9fH4DGjRsf9P3ExERiY2O58cYbOeeccxg9evRBy7Ozs9m3bx+nnXYaANdccw0XX3xxxfILLrgAgAEDBrB582av47700ksBWLduHStXruSss5yaYaWlpaSkpJCXl8eiRYsO2teBAwe83v6xCLtEsN1NBD4tOFdaBKvfhxP+AGc84IwRMMYcs02bNhEZGUmzZs1QVZ5//nnOPvvsg9b55JNPauxCGRUVxdKlS/niiy+YNWsWL7zwwhFddder59zVR0ZGUlJS4vX3GjRoADiJrGfPnixevPig5Tk5OTRs2LDiGUNdCrteQ5nZBUT6YjBZ/h746jEoLXGKxE1cCqOetCRgTC3ZtWsXEyZMYOLEiYgIZ599Ni+99BLFxcUArF+/nv379zN8+HCmTZtGfn4+wCFNQ3l5eWRnZzNq1CieeeaZQ068SUlJNGrUiG+++QaAN954o+LuoDZ07dqVXbt2VSSC4uJiVq1aRWJiIh06dOCdd94BnITx008/1dp+axJ2dwQZ2YU0T6hXu4PJVr8PH/0Z8rOgw6lOkbjYIJnrwJgAVlBQQGpqKsXFxURFRXHVVVdx5513AnDjjTeyefNm+vfvj6qSnJzM3LlzGTFiBGlpaQwcOJCYmBhGjRrFo48+WrHN3Nxcxo4dS2FhIarK008/fch+X3vttYqHxR07dmT69Om1dkwxMTHMnj2b22+/nezsbEpKSrjjjjvo2bMnM2fO5Oabb+bhhx+muLiYyy67jL59+9bavqsj5e1twWLgwIH6/fffH/X3L5+6hKLSMubcfNKxB5ObCR//GdZ84MwdPHYypPQ59u0aEwDWrFlD9+7d/R2GOQpV/e5EZLmqDqxq/bC7I8jMKaRHy8Ta2dg718K2H+DMSTD4NogMu79OY0wICKszl6qyfV8BZ3ZvdvQb2fcbxDVyi8Q9AdFx0LRL7QVpjDF1LKweFu8rn5nsaAaTlZXBd/8Lk0+ELx9xPkvpY0nAGBP0wuqOICP7KOch2LXeKRK3dYkzHmDwLT6Izhhj/CPMEsFRjCr+eTbMvRliGsD5/wt9LnVqBhljTIgIs0RQfkfgRdNQWRlERECr/tDjPDj7EYg/hmcLxhgToMLqGUFmdiGREUJyQg2DyYoL4PO/wb+uciqGNu4IF75sScAYPygvQ92rVy/GjBlTZV2gozFjxgwmTpxYK9vyVFJSwr333kuXLl0qylE/8sgjtb6f2hZWiWB7dkHNg8m2LIIpJ8O3zzg9g0qL6zQ+Y8zBystQr1y5ksaNGzN58mR/h1Sj+++/n+3bt/Pzzz+TlpbGN998UzHyOZCFVdNQZnUzkx3IhfmTYNkr0LAdXDUXOg2r6/CMCWzTzzn0s57nOTW1ivJh5sWHLk8dB/2ugP1Z8K+rD1523UdHtPvBgwezYsUKAJYuXcodd9xBQUEBcXFxTJ8+na5duzJjxgzmzZtHfn4+Gzdu5Pzzz+eJJ55wwp8+nccee4yUlBSOO+64ippBW7Zs4frrr2fXrl0kJyczffp02rZty7XXXktcXBxr165ly5YtTJ8+nddee43FixczaNAgZsyYcVB8+fn5vPzyy2zevJnYWOc8k5CQwKRJkwDYvHkzo0ePZuXKlQA89dRT5OXlMWnSpGrLT7/zzjs8+OCDREZGkpSUxMKFC1m1ahXXXXcdRUVFlJWVMWfOHLp0Obbei2GVCDKyC+mRUsVgstJiWPsRnHgLnH6/82DYGBMwSktL+eKLL7jhhhsA6NatGwsXLiQqKor58+dz7733MmfOHADS0tL48ccfqVevHl27duW2224jKiqKv/3tbyxfvpykpCSGDRtGv379AJg4cSJXX30111xzDdOmTeP2229n7ty5AOzdu5cvv/ySefPmMWbMGL799lteeeUVjj/+eNLS0khNTa2IccOGDbRt25aEhCOvL1Zd+emHHnqITz/9lFatWlU0i02ZMoU//vGPXHHFFRQVFVFaWnr0f7GusEkEzsxkBZzRzW3rz98DS16C0/7iFolbZgXijKlJTVfwMfVrXt6gyRHfAcDvtYY2b97MgAEDKko3Z2dnc8011/DLL78gIgc1v5xxxhkkJTm1vnr06MGWLVvYvXs3Q4cOJTk5GXBKQq9fvx6AxYsX8+677wJw1VVXcffdd1dsa8yYMYgIvXv3pnnz5vTu3RuAnj17snnz5oMSQWXTp0/n2WefJSsri0WLFlW7Xk3lp4cMGcK1117LJZdcUlH+evDgwTzyyCOkp6dzwQUXHPPdAPj4GYGIjBCRdSKyQUTuqWK5iMhz7vIVItLfV7FkFxRTWFxGi8R6sOo9mHwC/OefkL7UWcGSgDEBp/wZwZYtWygqKqp4RvDf//3fDBs2jJUrV/LBBx9QWFhY8Z3yJh84uFR0TaWpPXmuV76tiIiIg7YbERFxSAnqzp0789tvv5GbmwvAddddR1paGklJSZSWlhIVFXXQfMvlMZeVlVWUny7/WbNmDeBc/T/88MNs3bqV1NRUsrKyGDduHPPmzSMuLo6zzz67Viau8VkiEJFIYDIwEugBXC4iPSqtNhLo4v6MB17yVTzb9xXSjL2MXnu3UyMosRWMXwDtaqH4nDHGp5KSknjuued46qmnKC4uJjs7m1atWgEc0lZflUGDBrFgwQKysrIoLi6uKPUMcNJJJzFr1iwAZs6cycknn3xUMdavX58bbriBiRMnVpzkS0tLKSoqAqB58+bs3LmTrKwsDhw4wIcffghQY/npjRs3MmjQIB566CGaNm3K1q1b2bRpEx07duT222/n3HPPrXhucix8eUdwArBBVTepahEwCxhbaZ2xwOvqWAI0FJEUXwSTmVPA5JhnabbjGzjrIbjxC2jR2xe7Msb4QL9+/ejbty+zZs3i7rvv5q9//StDhgzxqo08JSWFSZMmMXjwYM4880z69/+98eG5555j+vTp9OnThzfeeINnn332qGN85JFHSElJoVevXvTr149TTjmFa665hpYtWxIdHc0DDzzAoEGDGD169EFzEc+cOZNXX32Vvn370rNnT95//30A7rrrLnr37k2vXr049dRT6du3L2+//Ta9evUiNTWVtWvXcvXVV1cXjtd8VoZaRC4CRqjqje77q4BBqjrRY50PgcdV9T/u+y+Av6jq95W2NR7njoG2bdsO2LJlyxHH8/3mPfx7/ufcclYvmrQLnGn3jAlUVoY6eAVSGeqqGuQqZx1v1kFVpwJTwZmP4GiCGdi+MQNvvPRovmqMMSHNl01D6UAbj/etge1HsY4xxhgf8mUiWAZ0EZEOIhIDXAbMq7TOPOBqt/fQiUC2qmb4MCZjzBEIthkMzdH9znzWNKSqJSIyEfgUiASmqeoqEZngLp8CfAyMAjYA+cB1vorHGHNkYmNjycrKokmTJl53vTT+papkZWVVjGz2VtjNWWyM8U5xcTHp6ekH9dE3gS82NpbWrVsTHR190Oc2Z7Ex5ohFR0fToUMHf4dh6kBYVR81xhhzKEsExhgT5iwRGGNMmAu6h8Uisgs48qHFjqbA7loMJxjYMYcHO+bwcCzH3E5Vk6taEHSJ4FiIyPfVPTUPVXbM4cGOOTz46pitacgYY8KcJQJjjAlz4ZYIpvo7AD+wYw4PdszhwSfHHFbPCIwxxhwq3O4IjDHGVGKJwBhjwlxIJgIRGSEi60Rkg4jcU8VyEZHn3OUrRKR/VdsJJl4c8xXusa4QkUUi0tcfcdamwx2zx3rHi0ipO2teUPPmmEVkqIikicgqEfm6rmOsbV78204SkQ9E5Cf3mIO6irGITBORnSKysprltX/+UtWQ+sEpeb0R6AjEAD8BPSqtMwr4N84MaScC3/k77jo45pOARu7rkeFwzB7rfYlT8vwif8ddB7/nhsBqoK37vpm/466DY74X+If7OhnYA8T4O/ZjOOZTgf7AymqW1/r5KxTvCE4ANqjqJlUtAmYBYyutMxZ4XR1LgIYiklLXgdaiwx6zqi5S1b3u2yU4s8EFM29+zwC3AXOAnXUZnI94c8zjgHdV9TcAVQ324/bmmBVIEGfShHicRFBSt2HWHlVdiHMM1an181coJoJWwFaP9+nuZ0e6TjA50uO5AeeKIpgd9phFpBVwPjClDuPyJW9+z8cBjURkgYgsF5Gr6yw63/DmmF8AuuNMc/sz8EdVLaub8Pyi1s9foTgfQVVTKVXuI+vNOsHE6+MRkWE4ieBkn0bke94c8zPAX1S1NERm2PLmmKOAAcAZQBywWESWqOp6XwfnI94c89lAGnA60An4XES+UdUcH8fmL7V+/grFRJAOtPF43xrnSuFI1wkmXh2PiPQBXgFGqmpWHcXmK94c80BglpsEmgKjRKREVefWSYS1z9t/27tVdT+wX0QWAn2BYE0E3hzzdcDj6jSgbxCRX4FuwNK6CbHO1fr5KxSbhpYBXUSkg4jEAJcB8yqtMw+42n36fiKQraoZdR1oLTrsMYtIW+Bd4Kogvjr0dNhjVtUOqtpeVdsDs4FbgjgJgHf/tt8HThGRKBGpDwwC1tRxnLXJm2P+DecOCBFpDnQFNtVplHWr1s9fIXdHoKolIjIR+BSnx8E0VV0lIhPc5VNwepCMAjYA+ThXFEHLy2N+AGgCvOheIZdoEFdu9PKYQ4o3x6yqa0TkE2AFUAa8oqpVdkMMBl7+nv8OzBCRn3GaTf6iqkFbnlpE3gKGAk1FJB34GxANvjt/WYkJY4wJc6HYNGSMMeYIWCIwxpgwZ4nAGGPCnCUCY4wJc5YIjDEmzFkiMAHJrRaa5vHTvoZ182phfzNE5Fd3Xz+IyOCj2MYrItLDfX1vpWWLjjVGdzvlfy8r3YqbDQ+zfqqIjKqNfZvQZd1HTUASkTxVja/tdWvYxgzgQ1WdLSLDgadUtc8xbO+YYzrcdkXkNWC9qj5Sw/rXAgNVdWJtx2JCh90RmKAgIvEi8oV7tf6ziBxSaVREUkRkoccV8ynu58NFZLH73XdE5HAn6IVAZ/e7d7rbWikid7ifNRCRj9z69ytF5FL38wUiMlBEHgfi3Dhmusvy3D/f9rxCd+9ELhSRSBF5UkSWiVNj/iYv/loW4xYbE5ETxJln4kf3z67uSNyHgEvdWC51Y5/m7ufHqv4eTRjyd+1t+7Gfqn6AUpxCYmnAezij4BPdZU1xRlWW39HmuX/+CbjPfR0JJLjrLgQauJ//BXigiv3NwJ2vALgY+A6neNvPQAOc8sargH7AhcDLHt9Ncv9cgHP1XRGTxzrlMZ4PvOa+jsGpIhkHjAfudz+vB3wPdKgizjyP43sHGOG+TwSi3NdnAnPc19cCL3h8/1HgSvd1Q5waRA38/fu2H//+hFyJCRMyClQ1tfyNiEQDj4rIqTilE1oBzYFMj+8sA6a5685V1TQROQ3oAXzrltaIwbmSrsqTInI/sAunQusZwHvqFHBDRN4FTgE+AZ4SkX/gNCd9cwTH9W/gORGpB4wAFqpqgdsc1Ud+n0UtCegC/Frp+3Eikga0B5YDn3us/5qIdMGpRBldzf6HA+eKyJ/d97FAW4K7HpE5RpYITLC4Amf2qQGqWiwim3FOYhVUdaGbKM4B3hCRJ4G9wOeqerkX+7hLVWeXvxGRM6taSVXXi8gAnHovj4nIZ6r6kDcHoaqFIrIAp3TypcBb5bsDblPVTw+ziQJVTRWRJOBD4FbgOZx6O1+p6vnug/UF1XxfgAtVdZ038ZrwYM8ITLBIAna6SWAY0K7yCiLSzl3nZeBVnOn+lgBDRKS8zb++iBzn5T4XAue532mA06zzjYi0BPJV9f+Ap9z9VFbs3plUZRZOobBTcIqp4f55c/l3ROQ4d59VUtVs4Hbgz+53koBt7uJrPVbNxWkiK/cpcJu4t0ci0q+6fZjwYYnABIuZwEAR+R7n7mBtFesMBdJE5EecdvxnVXUXzonxLRFZgZMYunmzQ1X9AefZwVKcZwavqOqPQG9gqdtEcx/wcBVfnwqsKH9YXMlnOPPSzldn+kVw5olYDfwgzqTl/8th7tjdWH7CKc38BM7dybc4zw/KfQX0KH9YjHPnEO3GttJ9b8KcdR81xpgwZ3cExhgT5iwRGGNMmLNEYIwxYc4SgTHGhDlLBMYYE+YsERhjTJizRGCMMWHu/wGWUvd0hmPm7gAAAABJRU5ErkJggg=="/>


```python
# AUC 계산 내가 만든 모형의 하단부위 면적. 최대값은 1이며 1에 가까울수록 우수한 모델을 나타냄. 0.5일 경우 분류능력 없음

from sklearn import metrics

roc_auc=metrics.auc(fpr, tpr)
print('Area Under Curve: %0.2f'% roc_auc)
```

<pre>
Area Under Curve: 0.76
</pre>
### 와인의 여러 변수값들을 입력하면 품질을 예측해주는 함수를 만드시오



```python
df.columns
```

<pre>
Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality'],
      dtype='object')
</pre>

```python
for i in df.columns[:-1]:
    print(type(df[i].values[1]))
```

<pre>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
<class 'numpy.float64'>
</pre>

```python
def predict_wine():
    Data=[]
    Data.append(float(input("fixed acidity: ")))
    Data.append(float(input("volatile acidity: ")))
    Data.append(float(input("citric acid: ")))
    Data.append(float(input("residual sugar: ")))
    Data.append(float(input("chlorides: ")))
    Data.append(float(input("free sulfur dioxide: ")))
    Data.append(float(input("total sulfur dioxide: ")))
    Data.append(float(input("density: ")))
    Data.append(float(input("pH: ")))
    Data.append(float(input("sulphates: ")))
    Data.append(float(input("alcohol: ")))
    result=tree.predict([Data])
    if(result==0):
        print("나쁜 품질의 와인")
    else:
        print("좋은 품질의 와인")
```


```python
predict_wine()
```

<pre>
fixed acidity: 8
volatile acidity: 8
citric acid: 8
residual sugar: 8
chlorides: 8
free sulfur dioxide: 8
total sulfur dioxide: 8
density: 8
pH: 8
sulphates: 8
alcohol: 8
좋은 품질의 와인
</pre>
## 9주차 단톡과제) 와인 품질 데이터를 랜덤포레스트로 분류

- 60191315 박온지

- 의사결정나무의 개수(n_estimators)를 변화시키며 정확도를 비교해볼 것

- 성능평가를 의사결정나무를 사용했을 때와 비교해볼 것


### Random Forest



```python
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100, random_state=0 )
# n_estimators : 사용할 tree의 개수 100개로 지정
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
array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)
</pre>

```python
# 랜덤 포레스트의 정확도 계산
temp_acc = accuracy_score(y_test, temp_y_pred_rf)

print(format(temp_acc))
```

<pre>
0.9270833333333334
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
>10, acc: 0.908
>15, acc: 0.919
>20, acc: 0.919
>25, acc: 0.912
>30, acc: 0.917
>35, acc: 0.917
>40, acc: 0.919
>45, acc: 0.923
>50, acc: 0.927
>55, acc: 0.925
>60, acc: 0.929
>65, acc: 0.927
>70, acc: 0.927
>75, acc: 0.925
>80, acc: 0.929
>85, acc: 0.927
>90, acc: 0.927
>95, acc: 0.925
>100, acc: 0.927
>105, acc: 0.927
>110, acc: 0.927
>115, acc: 0.923
>120, acc: 0.925
>125, acc: 0.925
>130, acc: 0.925
>135, acc: 0.927
>140, acc: 0.925
>145, acc: 0.925
>150, acc: 0.927
>155, acc: 0.923
>160, acc: 0.925
>165, acc: 0.925
>170, acc: 0.927
>175, acc: 0.927
>180, acc: 0.927
>185, acc: 0.929
>190, acc: 0.927
>195, acc: 0.929
>200, acc: 0.929
>205, acc: 0.929
>210, acc: 0.929
>215, acc: 0.929
>220, acc: 0.929
>225, acc: 0.929
>230, acc: 0.929
>235, acc: 0.931
>240, acc: 0.927
>245, acc: 0.927
>250, acc: 0.927
>255, acc: 0.927
>260, acc: 0.927
>265, acc: 0.927
>270, acc: 0.927
>275, acc: 0.927
>280, acc: 0.925
>285, acc: 0.925
>290, acc: 0.925
>295, acc: 0.923
>300, acc: 0.923
>305, acc: 0.923
>310, acc: 0.923
>315, acc: 0.923
>320, acc: 0.923
>325, acc: 0.923
>330, acc: 0.923
>335, acc: 0.925
>340, acc: 0.925
>345, acc: 0.925
>350, acc: 0.925
>355, acc: 0.925
>360, acc: 0.925
>365, acc: 0.925
>370, acc: 0.925
>375, acc: 0.925
>380, acc: 0.927
>385, acc: 0.925
>390, acc: 0.925
>395, acc: 0.923
>400, acc: 0.925
>405, acc: 0.923
>410, acc: 0.925
>415, acc: 0.925
>420, acc: 0.925
>425, acc: 0.925
>430, acc: 0.925
>435, acc: 0.925
>440, acc: 0.925
>445, acc: 0.925
>450, acc: 0.925
>455, acc: 0.925
>460, acc: 0.923
>465, acc: 0.923
>470, acc: 0.921
>475, acc: 0.921
>480, acc: 0.923
>485, acc: 0.921
>490, acc: 0.923
>495, acc: 0.921
</pre>

```python
# 위처럼 0부터 500까지 숫자를 늘려가며 정확도를 봄
# 나무개수가 200개 정도 지나면 정확도가 거의 최대치도달
pyplot.plot(range(10, 500, 5), scores, 'b--', label='RF_acc')
pyplot.legend()
```

<pre>
<matplotlib.legend.Legend at 0x2432abd3730>
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYAAAAD4CAYAAADlwTGnAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAwjUlEQVR4nO3deXhU5dn48e9NIIDseylhUylLE0BFRcWqdQV9i0tr1VYQRcWt0NZXEbW2tX21WNviBuKuFZdq+VVbWheqUtSySSDBgKxKAMMmiyxCwvP7457jnEwmySSZ/dyf65przpxl5jmTybnPs4tzDmOMMcHTKNUJMMYYkxoWAIwxJqAsABhjTEBZADDGmICyAGCMMQHVONUJqIuOHTu6Xr16pToZxhiTURYtWrTVOdcpcn1GBYBevXqxcOHCVCfDGGMyioh8Gm29FQEZY0xAWQAwxpiAsgBgjDEBlVF1AMYYA3Dw4EFKS0vZv39/qpOSVpo1a0ZeXh5NmjSJaX8LAMaYjFNaWkqrVq3o1asXIpLq5KQF5xzbtm2jtLSU3r17x3SMFQEZYzLO/v376dChg138fUSEDh061ClXZAHAGJOR7OJfVV2/EwsAxkSxaxc8+2yqU2FMYlkAMCaKMWNg9Gj4+ONUp8SYxLFKYGOi+O9/9fmLL1KbDpO+cnJyKCgooLy8nN69e/Pcc8/Rtm1b1q1bR//+/enbt+/X+86fP5/c3NwUpjY6ywEYE4V34bcJ80x1mjdvTmFhIcXFxbRv356HH374621HHHEEhYWFXz/S8eIPlgMwJqoVK+DLL6F//1SnxMTi1FOrrrv4Yrj+eti7F0aMqLr9iiv0sXUrfP/7lbe9+27dPv+EE05g6dKldTsIzRlMmDCBffv20bx5c5566in69u1LRUUFt956K2+88QYiwtVXX81NN93EggULGD9+PHv27KFp06bMnj2bVq1a1flzPRYAjImie/dUp8BkioqKCmbPns1VV1319brVq1czePBgAE466aRKuQO/fv36MWfOHBo3bszbb7/NpEmTePXVV5k+fTpr165l8eLFNG7cmO3bt3PgwAF++MMf8tJLL3Hssceya9cumjdv3qC0WwAwJsJHH+kd4CuvwNlnw113pTpFpjY13bEfdljN2zt2rPsdP8C+ffsYPHgw69at45hjjuHMM8/8eptXBFSbnTt3Mnr0aFauXImIcPDgQQDefvttxo0bR+PGeolu3749RUVFdO3alWOPPRaA1q1b1z3REawOwJgI//gH3HwzrF8Py5alOjUmXXl1AJ9++ikHDhyo9i6/JnfeeSennXYaxcXFvP7661934nLOVWnTH21dQ1kAMCZCUREcfrg+yspSnRqT7tq0acMDDzzA73//+6/v4GO1c+dOunXrBsDTTz/99fqzzjqLadOmUV5eDsD27dvp168fGzduZMGCBQDs3r376+31FVMAEJFzRGSFiKwSkYlRtrcTkZkislRE5otIfmh9s9DrJSKyTER+5TumvYi8JSIrQ8/tGnQmxsRJcTHk50PnzrB5c6pTYzLBUUcdxaBBg3jxxRfrdNwtt9zCbbfdxkknnURFRcXX68eOHUuPHj0YOHAggwYNYsaMGeTm5vLSSy9x0003MWjQIM4888wGD4YnrpZ2biKSA3wCnAmUAguAS51zH/v2uQ/40jn3KxHpBzzsnDtdNL/Swjn3pYg0AeYC451z/xWRycB259y9oaDSzjl3a01pGTJkiLMZwUwiffUVtGgBt90G27fDCy/os0kvJSUl9LcmWlFF+25EZJFzbkjkvrHkAI4DVjnn1jjnDgAvAiMj9hkAzAZwzi0HeolIF6e+DO3TJPTwIs5I4JnQ8jPA+TGkxZiEWrNG2/4XFMAxx8CwYXDoUKpTZUxixNIKqBuw3ve6FDg+Yp8lwIXAXBE5DugJ5AFloRzEIuBINGcwL3RMF+fcJgDn3CYR6Rztw0XkGuAagB49esR0UsbUV//+sGePLjdrBldemdr0mOzw1FNPMWXKlErramoemiyxBIBo1c6R5Ub3AlNEpBAoAhYD5QDOuQpgsIi0BWaKSL5zrjjWBDrnpgPTQYuAYj3OmPpq1izVKTCxSESrmEQZM2YMY8aMSfjn1FakHymWIqBSwN8tJg/YGPGhu5xzY5xzg4FRQCdgbcQ+O4B3gXNCq8pEpCtA6Nmq20zK3Xkn/OlPurxkCeTlwdtvpzRJJopmzZqxbdu2Ol/wspk3IUyzOtzBxJIDWAD0EZHewAbgEuAy/w6hu/u9oTqCscAc59wuEekEHHTO7RCR5sAZwO9Ch70GjEZzD6OBv8WcamMS5Omn4ZRTdLlFC9iwATZtSmmSTBR5eXmUlpayZcuWVCclrXhTQsaq1gDgnCsXkRuBN4Ac4Enn3DIRGRfaPg3oDzwrIhXAx4DXJ7or8EyoHqAR8LJz7u+hbfcCL4vIVcBnwA9iTrUxCbBjB5SWagUwaDNQsL4A6ahJkyYxT3toqhfTUBDOuVnArIh103zLHwJ9ohy3FDiqmvfcBpxel8Qak0jFoZopLwC0agVNm1pfAJO9bCygBKuogGnTtDVJrOM2FRbqReessxKatIy1bh1Mn67fLejkLf366bAN0WbxGjcOevfWMX5eeqnq9p/8BLp1g1df1df5+fosAl26WA7AZC8LAAn2j3/AjTfqneTYsbEdc1QozxT0+q1Jk+DYY+GCCyqvf+ghuP/+cGud007TALBmDTzwQNX3+d73NACsWBF9+6WXagDo1w+OPrrySKAXXwzW+thkq1p7AqeTTOwJ/OqrOtZ4YSEMGhTbMd/9LrzzjuYCOnVKaPLS1oEDWgl7881wzz2Vt519to7hvmhRatJmTKZpSE9g0wBe+XFdLuS33abPRUXxT0+m+OQTKC+He++F1asrb+vXD849N3lpsZ7AJltZAEgwLwCcdFLsx3gD/BXH3F0u+/iDX+Sw6lOmwK9/nZx03H03tGxpxXEmO1kASDAvAKxbBzt31r7/vn06fV2TJnDccQlNWlrzBwD/ckVFci/GLVvq3ySWv50xmcYCQILdeivccYcux3JH7wWMqVNh6NDEpSvd7dihdSZ9+lQOAFOnanHatm3JSYf1BTDZzAJAgvXoEW79E0uZvhcAWrWCBQuCW/78yCPabLOgoHLgLCrS76R9++SkwwsA1hfAZCMLAAn27LPau7R169hyAN6d5ltvaRHQp58mNn3prFEjzQV16hRu819crEEhWWOAWQAw2cwCQIKNHw8vvggTJsAJJ9S+v3ehOT3URzqILYE++gjOOEM7dv3v/8LcuZCTo2X/XgBIlu7dtR9Hz57J+0xjksU6giXQgQNalt25s44yGYtTTtEByc44Q18XF2tHpiBZtAhmz4bDDqu8fv162LUr3FM3Gdq3hwcfTN7nGZNMFgASyBuosEsXvXstK4M2bWoeEuKII/QB0KtXMHMARUXaCaxnT/3eTjlFH+PGaY5g2LDkpufAAW0J1KZNcj/XmESzIqAE8srzO3fWYoyuXWHOnJqPWbQIFi/W5fz84AaA/HytAxCBL7+E+fN1uIbJk5ObAwAdmuOqq2rfz5hMYwEggbzy/M6dYcAAXa7tgj5xItxwgy5PmqStYYLEOf2O/OX8Xkugdetg797kp6lzZ2sGarKTBYAEOvVUHYDsqKOgQwfNAdQWAMrKwi1PTjgBvvOdhCczrezbpwOynXhieF1+PmzcCCefDD9IwawRXbpYKyCTnSwAJFCzZvCtb4XL/PPza28KunlzOAAcOAD/7/8FqxjosMPgzTd1iGePlxvwT9aSTJYDMNnKAkACzZqlQxd7Cgrg44/DbdojHTqkFcdeAAAdjnjGjMSmM51EG+Zh0CDo31+XUxUAdu6Er75K/mcbk0gWABLoxRfh978Pv77kEnj00eoDwLZtGgS6dNHXubnQt2+wBoW79trwnLyerl3hrrt0OdkVwKB9Mu65p/q/mzGZypqBJpC/OAd0cpNjj4WHH9bhjkW0qMObJ6BVK+0B3Mc3uWZBAbzxhnYo++lPtWloJOd02OTPP9cmpD/5ia7/7W81DRdfXLfRSD2HDmkb+FGjoF276vebMwcOHgx3XqtNeTn85jd6URfRWbo++EC3zZoVvtv384rB+vWr2znEwwkn6OO++7QYyu/II+Gmm3T57rt1ngK//Hy4+mpdvvNO7cfg16qVrm/aNDFpzwRbt8ILL2iHu2T18DYhzrmMeRxzzDEuXe3b51xFReV1Rx3l3LnnVt135Ejn2rZ1rlEj537845rfd8YM59q31/0XLIi+T1GRc+BcixbODR8eXn/ssc7l5ERPQyzeeUffd9SomvfTEBT7+773nu6/cqW+njBBz69tW+fatXPuwQerHjN1qn5vqfSd74TT6T3OOy+8/eijq26/5JLw9r59K29r1cq5++5zbufO5J9LOrnoIv09zJuX6pRkL2Chi3JNTflFvS6PdA4Ao0frt/nZZ+F13bo5d+WV1R9zzjnODRoUfv3JJ8795S/O7dkTff+dO5274Qbn3nyz8voZM/SzlyypeszZZ2sgqI+dO/V9L7ig5v3OO69uAeChh3T/9evrly6TXYYP199Ddb9703DVBQCrA4iTFSv0+a9/1WfnqhYBRSoogJKS8AQw//ynNnOsrq178+bw2GNaTORXVASNG0cvHmlIC5bWrbUp66ZNNe+Xn6+fH+s4/cXF0LatduwKuiVL4L33Up2K1Pr8c53mM3LoD5N4FgDizCurFoEvvtDOXNX56U91IvOcHH29ebMuVzfUcZMmWj4eWSl89NFaR5CbW/WYLl20OWl9vPQS7N6tn1fdsNTz58P06RrEduyI7X29jl5W3qsd/yZMSHUqUqe8XFvGlZXB88+nOjXBYwEgTryOQv42+y1aaCVfdbp21btg70JYVqZDHzeq4a8SbXiI73+/cmsjv8mTa7+Dr8748bBhg47Bs39/9H3+9S/Yvl0rSavbxy8VI3qms8hcYNDs3QtXXgmffVbzzZJJDAsAceIFgGXL9G55+XL4+c91+IKa3H8/vPpq+D1qKjICvWCUlmruAvTuvqYinvreZW/Zou/7v/+rrV+qy54XFWnLow8+0IAWi3//O9xSKejy87V/wcqVqU5JarRurcOd3HyzBoHIVlImsSwAxMGhQ/oD/sEPYM8evegXF8Mf/lD7D/rxx+HPf9Zl/zAQ1SkogLw8HRoBdOz8b3wD/v736PuXlMAPf1j3vgTe/gUFOjxDdUMhRI7bUxsRLbLq27du6clW3ncXpL4eftu3a+7H698R1O8hVSwAxEGjRtqm/cEH4cMPtVjHuyv3OnVVp6AgXKTz/PO1jz0/fLiOi//tb+tr71hvsLlI+/fDyy/DqlWxnYvHe9+CAv3njFZOvW+f3rnm5Wl7+CeeqP19330XnnsuuRO7p7P+/fX3E6ThPvzGjtUbAi8QBvV7SBXrCBYHe/fqnX6nTuEL/ubNerfboUPNx+bnwyuvaM7BmwegJpFFOsXFWtcQrYMY1H9Kw+JiTXuXLhpcov1jfvqpVkyfeKJ2bvvss9rf9/HHtePY5ZfXLT3ZqlkzeP99HTMqiIqLYeBAnfuhVSutEDbJYzmAOPj3v7X8+6OP4B//0CEgNm+Gjh21eWRNCgr0bvijj+BPf4rtH+COO+DHP9Zl/9j50XTqpM91bQr64IPawkdE33/58qqtifr107H6L7pIzzWWIOOl14QNHZq8Se7Tyd69mjP1WoSVlMAf/5jqVAWLBYA48E/88uijOiTAl1/WXp4P+uNv3BjmzdNmofPm1X7Mtm0aaKKNnR8pN1fb3Nc1B9C0KRx+eDiN5eXhvg5+jRvrZ8TS3+DgQQ0k1gKosuXL9TezZ0+qU5JcH3+sv2HvhqBbt5pbwJn4s687DvwTvxQU6IXy8cehsLD2Yw8/XP/xTz45/B61yc/XNvfr12vzz1Gjat7/29+O3kegOhs2aJn/8uX6urqKyltu0TGIvHTXFmRWrtRchAWAykpK4Be/CF7xh7+eCbRT3NVX29DbyWQBIA7KyrT8snlz/TFXVOjFs7biH9A7ntzc2CuNIfwPs2wZjB4dDh7VmTtXm5vGatEimDIl3NS0b19t0TRkSOX9ZszQNACcdVbtA86VlOizFQFVFtQWMMceqwMWenVf27frjdOSJalNV5BYJXAc+NvvexfnIUO0LuCii2o//oUX4LLLdDnWHABo654uXXTGsXj2qvUuRF5Lo9xcLZ7y++ILzSl45ztxYu3ve+GF2imtY8f4pTUbHH643jwErQVMfn7lmwF/S6CzzkpNmoLGcgBxcPnlWjEL4dYc5eXhIpTafP55eDmWANC+PZx3Hjz9tDYLre3iP3267heroiJtldG6deU0vvlm+LUXJPz/wN64oNUR0T4LseSMgiQnR4Nt0ALAe+/pXb+nY0f9fQTte0ilmAKAiJwjIitEZJWIVLnXE5F2IjJTRJaKyHwRyQ+t7y4i74hIiYgsE5HxvmN+KSIbRKQw9BgRv9NKruHD4YordLlJE209A7FdzCF8EX3sMW0WGIvXX9dcRizl6Zs26ZANBw/G9t7RKpafeEIH7PI6tkWW386Yob2FI8fL9xs/Xqe4NFXl59e9r0Ym27ZNBxp86qnK6wsKglcUlkq1BgARyQEeBoYDA4BLRSSy29EkoNA5NxAYBUwJrS8Hfu6c6w8MBW6IOPaPzrnBocesBp5LyhQW6g/a492Rx1KeD+GLaF1agRw6pP8oXjFNTbx0RE5WEk1FhQ4AFxkAvNf+ikqvVzJAy5ba6ay6Crw9e+CBB+zurjpTpgQrAHi/g8j6oIEDtaGAdRRMjlgy48cBq5xzawBE5EVgJOBvszAAuAfAObdcRHqJSBfn3CZgU2j9bhEpAbpFHJvRKirgmGPg9tvh17/Wdd6wDLHO8uRdoG+/Xe+SY/Hgg3rBjeUzvJxIWVl4vJ7CQm2yGvmPNm6cdvCKnP7QCwB33KEDv919N1x/fdXPiGwJtGCB5h68CmVrARRd69aaU/vVr6pu+/GPYdiw5KXlgw/g2Wcrr+vdG269VZfvvlvrf/z69Qv3Fr/jjqo3G4MH628LdNiURYt0OfL3cN992rLtww/hmWc0Rz1xYuxDhy9erEWezsF11+lse0VF2lEx0oQJmm7vNxpp4kTtYDl3bni4Fr9f/lKLrDJZLAGgG7De97oUOD5inyXAhcBcETkO6AnkAV/fD4pIL+AowN/S/UYRGQUsRHMKX0R+uIhcA1wD0KNHjxiSm1zePL7+4p7zztPevUOHxvYeInDNNXUbH/+CCzT7fN11te8b7eK8YYMWC+3bV3nfkSP1n9UbotrTsyd85zua69izRy8Cfv4e0H4bNoSLffr21V7DJrrdu6MXkZ18cnIDwB/+oAMU+nOwRx8dDgDvvFO1yerJJ4cDwJtvVu0V/tVX4QDwz3/q/83pp1cdQNDLPX/2mX4XZWXQvbs2OY417S+8oPUJ55+vAaCsLPr36nWm9P9G/a69VgPAunXRt//851okesMNevN3wgmxpTGtRJslxv8AfgA87nt9OfBgxD6tgaeAQuA5YAEwyLe9JbAIuNC3rguQgxZD/RZ4sra0pOOMYN50jC+9lOqUVG/1ap2e8u23E/cZX36p38O99ybuM4Js//7kfdahQ85t3Zq8z6tJt27OXX555XVlZfpb69696v6DBulMe8myYYOm5YEHkveZ9UEDZgQrBbr7XucBGyOCyC7n3Bjn3GC0DqATsBZARJoArwLPO+f+6jumzDlX4Zw7BDyGFjVlnLq030+Vww/XoSa8SdsrKrRVTzzLWVu00Ds8f5b++ON18nfTMNOm6fe7c2dyPi+WMaySJdr8F97r9esr5zjLy7WvSTL7mXTtqq3yMrVuK5YAsADoIyK9RSQXuAR4zb+DiLQNbQMYC8xxzu0SEQGeAEqcc3+IOMaf+bsAyMi6f38v4EyxerX+cOM9A9PUqTAi1JZr925tDWVd+xsuL0+DttfpLpGWLtUJWtasSfxnxWLkSDjllMrr/K2E/Mv79+v8FSOS2J5QJLNbLtX67+mcKwduBN4ASoCXnXPLRGSciIRK9egPLBOR5WhrIa8q8yS0yOi7UZp7ThaRIhFZCpwGRHQ1ygxDh2p7/DSsnqhk5MhwOap3t9K/f3w/49AhHQMJovcTMPXjfYfJuMv873+1bildAvd11+kgiX7+78G/3LKl5jhPOy0pSftafr7+3jOx5VJMXXKcNtGcFbFumm/5Q6BPlOPmAlG7KTnnsmJA4N699ZHuNmwI9wMoKtI7l3gHgIsugrVrtYWRf0IZ0zDeUMnJCABFRXoh7dkz8Z8Vq4oKvbtv0UJfFxXpRb6oqPJ3smGDtqaqaRrWRDjxRB3/a9cuaNMmuZ/dUGkS5zNXYWG4SVs68w/WVlysE7hUN81jffmHhE7HC0mm8obkTlYAyM+P79AiDXHwoI5m+7vfhdf16qUB4IQTKt9133gjHJeCmsTLLoO33sq8iz/YWEAN9otfaGXU4sWpTknNunQJX0DqOo1jrLwgc+iQTiJz5ZXpU5SQ6a69VptSJpI3vHgs41clS5Mm2jzaX8b+0kvR9y0u1ibMqeJc+gTOWFkAaKBYJnJPB97F2TkNWt5EMfH+jIoK7fTltfk28TF6dOI/w5vVLpUX0WgKCsJDq1d3kd2zRxs3pGqmubPO0k5hkR3o0p3dnzVQLBO5p4PBg3Usn6++gh/9KDGjLXpNYUtLtczWxI9z2kN7y5bEfUabNjqAob+HdzooKNCL+549cM890KeP/o7XrtWmxv/6V3hymVTVOeXmpn8pQDQWABooU3IAl14Kr72mFWWLFmkxTbwNHgx33aV3ay1aaHd+Ex/btmnZd6bdYcZDfr5e3EtKtJlqebkOgdKxozY1XrSo+rGFkqWgIPq0qenOAkADfPmlzmuazp3AIk2dqhO3JKLJWr9+Oj6KNzzGkUfG/zOCKhlDJd98c3h4hHRy/PE61EKnTpXrr1q10qBYVKTDlDzySHga02Tzpk395JPUfH59WR1AA+Tmau1/qn50dfHJJ9pyYuNGnUAmcqyfeHBOc0Rz5mhQTEQ9Q5AVFCQ2ALz7LrRrl7j3r69u3eDOO7XY55NPdIwfj/edHHlkam84/LO6ZVLfF8sBNEBuLpxxRmYEgFat9OIPiSsnPXQIvvlN+NvfrP1/IhQUaFl35Eit8eD1NE7Xv9u2bToDXnl55TTm52tQ+Mtfqo5Qmkz9+umcIN7w6JnCAkADrF2ro356vV/TmX8axkT9k+fk6LgokFl3QZkiP18r11evjv97r16t752uAeDmm2HUKBg7tvLc1MOGaX+Aiy+uvnloMuTmag/qZI7aGg8WABrg7bfhBz8Ij3Wfzpo0CS8n8uLsBYALL0zcZwTVGWfoRS4RdU7p3nPbS9f//V/lop4RI8JzKKQ67c6lNhdSHxYAGiDTBoLLy9NelYkct7xHD33/k09O3GcEVffueqebiB6nhx2mdUQDIuf6SxPexX3p0qrbUt0CyDN5sv6P7d6d2nTUhVUCN0BZmf4zxjrzV6pdcYUGgER2Wd+1KzPbQ2eKxYv1QhP5m5s8WW9EZs3SsvJIDzyg4+T89a/aHDjSo4/COeckJs3x4F3czzijags2bxa9VM/O5Y2tdfnl+n+Wk6MDMPbtW/0x77+vI69GdmDbskVbNd1+OzRO4FXaAkADbN6cWS1dImfxSoTx48NTYpr4e/316P0rvJndPvtMW/NE8tqnr10bfXsi+oXE0ze+obmfyy6ruu3++/WCmephGE48UWcgKyzUIPW974UHsKvOj36kRciRAWDCBJgxQ9/zzDMTlWIQl0FjmA4ZMsQtXLgw1cn42mmn6WBVc+emOiXGmEzUvTuceio891zl9WefrVNrTp8OV1/d8M8RkUXOuSGR6y0H0ABPPGFDHhhjolu1Siecv/TS6Nu/+EKHTfEqj/1zgs+apUU/iR4B1iqBG+Dww9O30swYk1qvvKJFVtVN5em1vHr++ao5gJwcnQynT5VZVuLLAkA9lZfDlCmZOxeoMSax/L2Do/HWi1TeZ/167e9w/fVw002JTaMFgHraulUrav7zn1SnxBiTjrymq9UFgPPP11ZZw4dXvpH86CMtXt6zR280y8sTl0YLAPWUaX0AjDHJ1aNHzVN5du0KF1yggaKkpPKUraAjDLRoAe+8k7g0WgCoJy8AZNJIoMaY5KlpKk/nYNo0nUu4oEAv/t5IokVFOs/4wIHafLe6HEQ8WCugeior02fLARhjqvPUU+HhUfw2bNBK3oce0ik4Z80Kz5/tDXndqVPlqVwTwQJAPVkRkDGmNtX1AvYu6gUF2slt+HB97Y30OmiQPleXg4gXKwKqp2uv1Xa+bdumOiXGmHS1bRv85jfhOY09kYPvzZ2rFcI5OTrkt3+Au2XLEtdT23IA9XTYYXDEEalOhTEm3d15JzRrplOmeoqKtOOXNwHPgw9qpzFvFF1vWIvzz9dioK++gubN4582ywHU05NPVu28YYwxfh06aGufyIrcyJnDCgp0nKa77oKRI8MD3p1yCkycmJiLP1gOoN6mTdM/buQgTsYY4xdtKs85c2DHjvBrLxg8/LDOqucf2O7zz3Wwv9694582ywHUU1mZVQAbY2qXn191Ks+WLStPH+nVBWzbVnVim2HD4NZbE5M2CwD14E1+bgHAGFMb74JeWqrP778PkyZVzgH47+4jJ7aJloOIFwsA9bB7t44Cap3AjDG1ufRS7dXrtfN/80343e8qT+rTqBE8/bQuR+YA8vNh5crEjDxsdQD1sGWLPlsOwBhTG+9CP26c3vXPm6ejfEZW7B5xhFb6DhxYeX1BgRYfrVgR7h8QLxYA6uGII2Dv3tTPQGSMyRzLlunNY9OmOj1rpGHDos/WNmIEbNyYmCkvLQDUU6KaZRljslN9Rw5u2VIfiWB1APUwe7YOBb1nT6pTYowx9WcBoB4++EAng2nSJNUpMcaY+ospAIjIOSKyQkRWicjEKNvbichMEVkqIvNFJD+0vruIvCMiJSKyTETG+45pLyJvicjK0HO7+J1WYpWVaRfu3NxUp8QYY+qv1gAgIjnAw8BwYABwqYhEzoQ7CSh0zg0ERgFTQuvLgZ875/oDQ4EbfMdOBGY75/oAs0OvM4L1ATDGZINYcgDHAaucc2uccweAF4GREfsMQC/iOOeWA71EpItzbpNz7qPQ+t1ACdAtdMxI4JnQ8jPA+Q05kWQqK7M+AMaYzBdLAOgGrPe9LiV8EfcsAS4EEJHjgJ5Ann8HEekFHAXMC63q4pzbBBB6jnpPLSLXiMhCEVm4xWuAn2IHDlgAMMZkvliagUZr7e4iXt8LTBGRQqAIWIwW/+gbiLQEXgUmOOd21SWBzrnpwHSAIUOGRH5uSnz4YXi0PmOMyVSxBIBSoLvvdR6w0b9D6KI+BkBEBFgbeiAiTdCL//POub/6DisTka7OuU0i0hXYXO+zSAHrBGaMyXSxFAEtAPqISG8RyQUuAV7z7yAibUPbAMYCc5xzu0LB4AmgxDn3h4j3fQ0YHVoeDfytvieRTFu2wPe/X/9OHcYYky5qDQDOuXLgRuANtBL3ZefcMhEZJyLjQrv1B5aJyHK0tZDX3PMk4HLguyJSGHqMCG27FzhTRFYCZ4Zep70NG+DVV2Hr1lSnxBhjGiamoSCcc7OAWRHrpvmWPwT6RDluLtHrEHDObQNOr0ti04FNBm+MyRbWE7iOysr02QKAMSbTWQCoI8sBGGOyhQWAOhKBHj2gdetUp8QYYxrGAkAd/exn8Omn1gzUGJP5AjcfwMcfw113QXmom9p118FZZ6U2TcYYkwqBywG8/jq88gqsWgVr1sCuOvVLhvPPhz/9KREpM8aY5ApcDmDfPn1eskQnYq7rsa+/Hv95OY0xJhUClwMYNgwmTdKL/+bNcMMNOklzLEpK4NAhyM9PbBqNMSYZAhcAzjgDfvtbXT5wAB55BJYuje3YoiJ9LihITNqMMSaZAhcAduyAL77Q5TZt9HnnztiOLSqCpk3hyCMTkjRjjEmqwAWAm26CY47R5ZYttSgo1gDQuTNccAE0DlzNiTEmGwXuUrZvHzRvrssi2qEr1gBwyy2JS5cxxiRb4HIAe/eGAwBAx47hPgE1cc4mgTHGZJfABQB/DgBg5UqtCK7Nf/4DnTrF3mLIGGPSXeADQKyKimDbNugWORuyMcZkqMDVAdxwg7bk8UyZAuvXw+9/X/NxxcXQtq0FAGNM9ghcDuDyy+Hii8Ov58+HmTNrP66oSNv/2yBwxphsEbgAsHq1FuV4WreufTwg5zQHYB3AjDHZJHABYOhQuOOO8Os2bbQZaE0tfL76Cq65Bs49N/HpM8aYZAlcHcC+fXDYYeHXbdrAwYOwf3/1lcPNmsHkyclJnzHGJEugcgDOVW0F1KUL5OXBnj3VH7d5c3gUUWOMyRaBCgAHD+ponv4AcOWV2gqoY8fqj/vZz6Bfv8SnzxhjkilQRUDeXXxt/QCc0zGDVq/W1/Pmad2BMcZkk0DlAJo2hWnT4PTTw+uWL4cRI2DBgvC6Q4fg1FO16Gf7dvjWt2DMmKQn1xhjEkpcBg1wM2TIELdw4cK4vueSJTB4sE4TedFFcX1rY4xJCyKyyDk3JHJ9oHIAe/fCokWVR/+MNifAxo3w2muxjxJqjDGZKFABYPlyGDIE3n03vC5aAPjPf2DkSCgtTWryjDEmqQIVAKJVArdurc/+ALB5sz537pycdBljTCoEPgDk5GgdgBcIQANAo0bQoUNSk2eMMUkVqGage/fqc2Qz0MWLK78uK9Ox/xsFKjwaY4ImUJe4WPsBbN6sPYSNMSabBSoHMHQoPPccdO9eef2NN0JFBUydqq/vu6/2EUKNMSbTBSoA9Oypj0irV8PWreHXffokL03GGJMqgSoC+vRTmDNH7/b9vCGhPY89BkuXJjdtxhiTbDEFABE5R0RWiMgqEZkYZXs7EZkpIktFZL6I5Pu2PSkim0WkOOKYX4rIBhEpDD1GNPx0avb883DKKToonJ8/AOzdq2P/z5qV6NQYY0xq1RoARCQHeBgYDgwALhWRARG7TQIKnXMDgVHAFN+2p4Fzqnn7PzrnBoceCb/k7tunUzr65wSGygHA+gAYY4IilhzAccAq59wa59wB4EVgZMQ+A4DZAM655UAvEekSej0H2B6/JNefNxdA5Ly+3/oWHH88lJdbADDGBEcsAaAbsN73ujS0zm8JcCGAiBwH9ATyYnjvG0PFRk+KSLtoO4jINSKyUEQWbtmyJYa3rF7kZDCesWPhvfegceNwALBmoMaYbBdLAJAo6yKHEL0XaCcihcBNwGKgvJb3nQocAQwGNgH3R9vJOTfdOTfEOTekU6dOMSS3etUFAL+yMn22HIAxJtvFEgBKAX/L+Txgo38H59wu59wY59xgtA6gE7C2pjd1zpU55yqcc4eAx9CipoQaPx6efLLq+jlzdMavpUvhhz+Ejz+GbpF5HGOMyTKx9ANYAPQRkd7ABuAS4DL/DiLSFtgbqiMYC8xxztXYlUpEujrnNoVeXgAU17R/PAwaFH39oUOwYoX2BRg4EPr3T3RKjDEm9WoNAM65chG5EXgDyAGedM4tE5Fxoe3TgP7AsyJSAXwMXOUdLyIvAKcCHUWkFLjLOfcEMFlEBqPFSeuAa+N4XlHNnavj+5x4YuX1/iGhX34Z9u+HUaMSnRpjjEmtQM0Idsop2gLIPx8AwJo1cMQR8NRT8Oc/a1+ADz5oWFqNMSZd2IxgVF8J7M8BlJVZCyBjTDBYAEADwNlnQ16eNgO1FkDGmCAI1GBw1QWAxo3hX//SMYIuvthyAMaYYAhUDmDv3pr7AWzfri2CLAdgjAmCQOUAZs4Ml/dH+u53dajonTt1mkhjjMl2gQoAxx9f/bY9e2DjxspzAxtjTDYLTBGQc/Dss7BsWfTtbdrAm2/CrbdqUZAxxmS7wASA/fth9Gh47bXo272iocmTNVgYY0y2C0wA8CaEP+yw6Nu9ANC4MbSLOi6pMcZkl8AFgOpaAQ0bps+dOulwEcYYk+0Cc6mrLQBccQX8z/9YE1BjTHBYAPDZvdsCgDEmOALTDPTII2HxYm3rH82MGfD++1Cc8EGpjTEmPQQmADRvDoMHV7+9WTM4eFD7AxhjTBAEpgho3Tp45JHwnL+RmjTR55kzk5YkY4xJqcAEgMJCuOEG2LAh+vaDB/X5k0+SliRjjEmpwASA2iqBzz0XrrkG/vjH5KXJGGNSKTB1ALUFgKZN4dFHk5ceY4xJNcsBGGNMQFkAMMaYgApMALj6ali5Elq0SHVKjDEmPQSmDqBNm+ongzHGmCAKTA7gjTdgypRUp8IYY9JHYALAzJnw29+mOhXGGJM+AhMA9u2rfi4AY4wJokAFAGsBZIwxYRYAjDEmoCwAGGNMQAWmGejMmeEB34wxxgQoALRqleoUGGNMeglMEdD998PLL6c6FcYYkz4CEwAeegj+/vdUp8IYY9JHYAKAVQIbY0xlFgCMMSagYgoAInKOiKwQkVUiMjHK9nYiMlNElorIfBHJ9217UkQ2i0hxxDHtReQtEVkZem7X8NOpngUAY4yprNYAICI5wMPAcGAAcKmIDIjYbRJQ6JwbCIwC/MOuPQ2cE+WtJwKznXN9gNmh1wlRUaFNQC0AGGNMWCzNQI8DVjnn1gCIyIvASOBj3z4DgHsAnHPLRaSXiHRxzpU55+aISK8o7zsSODW0/AzwLnBrfU6iNjk5GgCcS8S7G2NMZoqlCKgbsN73ujS0zm8JcCGAiBwH9ATyannfLs65TQCh586xJLi+GjeGJk0S+QnGGJNZYgkAEmVd5L30vUA7ESkEbgIWA+UNS1row0WuEZGFIrJwy5Yt9XqPbdvg+uth3rx4pMgYY7JDLAGgFOjue50HbPTv4Jzb5Zwb45wbjNYBdALW1vK+ZSLSFSD0vDnaTs656c65Ic65IZ06dYohuVVt3QpTp8KqVfU63BhjslIsAWAB0EdEeotILnAJ8Jp/BxFpG9oGMBaY45zbVcv7vgaMDi2PBv4We7LrxiaEN8aYqmoNAM65cuBG4A2gBHjZObdMRMaJyLjQbv2BZSKyHG0tNN47XkReAD4E+opIqYhcFdp0L3CmiKwEzgy9TggvANiEMMYYExbTYHDOuVnArIh103zLHwJ9qjn20mrWbwNOjzmlDWA5AGOMqSoQPYEPHoTcXAsAxhjjF4jhoM8+G776KtWpMMaY9BKIHIAxxpiqAhEAZs+G0aNhx45Up8QYY9JHIALAsmXw7LM6JpAxxhgViABgrYCMMaaqQAWAZs1Smw5jjEkngQkATZtCo0CcrTHGxCYQl8TcXPjmN1OdCmOMSS+BCAB33w1r1qQ6FcYYk14CEQCMMcZUZQHAGGMCygKAMcYElAUAY4wJKAsAxhgTUBYAjDEmoCwAGGNMQFkAMMaYgLIAYIwxASXOuVSnIWYisgX4tJbdOgJbk5CcdBTUcw/qeYOdu517bHo65zpFrsyoABALEVnonBuS6nSkQlDPPajnDXbudu4NY0VAxhgTUBYAjDEmoLIxAExPdQJSKKjnHtTzBjv3oIrLuWddHYAxxpjYZGMOwBhjTAwsABhjTEBlTQAQkXNEZIWIrBKRialOT7yJyJMisllEin3r2ovIWyKyMvTczrftttB3sUJEzk5NquNDRLqLyDsiUiIiy0RkfGh9Vp+/iDQTkfkisiR03r8Krc/q8/YTkRwRWSwifw+9DsS5i8g6ESkSkUIRWRhaF/9zd85l/APIAVYDhwO5wBJgQKrTFedz/A5wNFDsWzcZmBhangj8LrQ8IPQdNAV6h76bnFSfQwPOvStwdGi5FfBJ6Byz+vwBAVqGlpsA84Ch2X7eEd/Bz4AZwN9DrwNx7sA6oGPEurife7bkAI4DVjnn1jjnDgAvAiNTnKa4cs7NAbZHrB4JPBNafgY437f+RefcV865tcAq9DvKSM65Tc65j0LLu4ESoBtZfv5OfRl62ST0cGT5eXtEJA84F3jctzoQ516NuJ97tgSAbsB63+vS0Lps18U5twn0Igl0Dq3P2u9DRHoBR6F3w1l//qEikEJgM/CWcy4Q5x3yJ+AW4JBvXVDO3QFvisgiEbkmtC7u5944TolNNYmyLsjtW7Py+xCRlsCrwATn3C6RaKepu0ZZl5Hn75yrAAaLSFtgpojk17B71py3iJwHbHbOLRKRU2M5JMq6jDz3kJOccxtFpDPwlogsr2Hfep97tuQASoHuvtd5wMYUpSWZykSkK0DoeXNofdZ9HyLSBL34P++c+2todWDO3zm3A3gXOIdgnPdJwPdEZB1apPtdEfkzwTh3nHMbQ8+bgZlokU7czz1bAsACoI+I9BaRXOAS4LUUpykZXgNGh5ZHA3/zrb9ERJqKSG+gDzA/BemLC9Fb/SeAEufcH3ybsvr8RaRT6M4fEWkOnAEsJ8vPG8A5d5tzLs851wv9f/63c+7HBODcRaSFiLTyloGzgGISce6pru2OY635CLR1yGrg9lSnJwHn9wKwCTiIRvyrgA7AbGBl6Lm9b//bQ9/FCmB4qtPfwHMfhmZplwKFoceIbD9/YCCwOHTexcAvQuuz+ryjfA+nEm4FlPXnjrZmXBJ6LPOuZ4k4dxsKwhhjAipbioCMMcbUkQUAY4wJKAsAxhgTUBYAjDEmoCwAGGNMQFkAMMaYgLIAYIwxAfX/AXlcZEygEOx4AAAAAElFTkSuQmCC"/>


```python
# 의사결정나무 모델 성능
# 정확도, 정밀도, 재현율, f1-score
print('accuracy: ', accuracy_score(y_test, temp_y_pred))  
print('precision: ', precision_score(y_test, temp_y_pred))
print('recall: ', recall_score(y_test, temp_y_pred))
print('f1: ', f1_score(y_test, temp_y_pred))
```

<pre>
accuracy:  0.9041666666666667
precision:  0.5384615384615384
recall:  0.56
f1:  0.5490196078431373
</pre>

```python
# # n_estimators = 200으로 새로 적합
RF = RandomForestClassifier(n_estimators=200, random_state=0 )

# 모델 적합
RF.fit(X_train, y_train)
temp_y_pred_rf = RF.predict(X_test)
```


```python
# 렌덤 포레스트 모델 성능이 의사결정나무모델의 성능보다 전체적으로 높다
# 정확도, 정밀도, 재현율, f1-score
print('accuracy: ', accuracy_score(y_test, temp_y_pred_rf))  
print('precision: ', precision_score(y_test, temp_y_pred_rf))
print('recall: ', recall_score(y_test, temp_y_pred_rf))
print('f1: ', f1_score(y_test, temp_y_pred_rf))
```

<pre>
accuracy:  0.9291666666666667
precision:  0.6666666666666666
recall:  0.64
f1:  0.6530612244897959
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

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABNW0lEQVR4nO3dd3yN5/vA8c8lQWLF3rtmkERRu6i9StFFjZafWt9+u6nWanWpb4dVVS0dWkpbRRW11aaN2IoaMWrHiJBx//64T4aIOMjJSXKu9+t1Xmc8zznnOsZzPc89rluMMSillPJcmdwdgFJKKffSRKCUUh5OE4FSSnk4TQRKKeXhNBEopZSH00SglFIeThOBUkp5OE0EKsMQkUMiclVELovISRGZLiI5XPA9FURktoicEZEwEQkRkRdFxCulv0up1KCJQGU07Y0xOYAgoDrwWkp+uIjcB2wEjgLVjDF+wKNATSDnXXyed0rGp9Td0ESgMiRjzElgMTYhACAiD4vIThG5ICIrRaRygm0lROQnETktImdFZMItPnoUsM4Y86Ix5oTju/YaY7oaYy6ISGMRCU34BseVSjPH45EiMkdEvhWRi8BQx1VM3gT7V3dcbWR2PH9GRHaLyHkRWSwipRyvi4h8JCKnElyZVE2JPz/lWTQRqAxJRIoDrYH9jucVgO+B54ECwEJgvohkcTTpLAAOA6WBYsDMW3x0M2DOPYbXwfEZuYEPgPVA5wTbuwJzjDGRItIRGAp0csS9xvE7AFoADwIVHJ/1OHD2HmNTHkgTgcpo5orIJWzTzSlghOP1x4FfjTG/G2MigbGAL1APeAAoCrxijLlijIkwxvxxi8/PB5y4xxjXG2PmGmNijDFXge+AJ8Ge5QNPOF4DeBZ41xiz2xgTBbwDBDmuCiKxzVGVAHHsc6+xKQ+kiUBlNB2NMTmBxtgDZH7H60WxZ/wAGGNisMmiGFACOOw40N7OWaDIPcZ4NNHzOUBdESmKPcM32DN/gFLAJ47mrAvAOUCAYsaY5cAEYCLwr4hMEZFc9xib8kCaCFSGZIxZBUzHnvkDHMceVIG4M+8SwDHsgbmkkx23S7mxGSexK0C2BN/jhW3SuSG8RLFeAJYAj2Gbhb438WWBjwLPGmNyJ7j5GmPWOd47zhhTA6iCbSJ6xYnfoNQNNBGojOxjoLmIBAE/AG1FpKmjE/Yl4BqwDtiEbe55T0Syi4iPiNS/xWeOAOqJyAciUhhARMo5On9zA/sAHxFp6/ieN4CsTsT6HdADm2S+S/D6ZOA1Eani+C4/EXnU8biWiNR2fM8VIAKIdupPRqkENBGoDMsYcxr4GhhmjNkLPAWMB84A7bFDTa8bY6Idz8sBR4BQbJ9CUp95AKiL7VTeKSJhwI/AFuCSMSYMGABMxV5tXHF83u3MA8oD/xpjtiX4vp+B94GZjlFGO7Cd4AC5gM+B89hmr7PEXwEp5TTRhWmUUsqz6RWBUkp5OE0ESinl4TQRKKWUh9NEoJRSHi7dFbzKnz+/KV26tLvDUEqpdGXr1q1njDGJ57QA6TARlC5dmi1btrg7DKWUSldE5PCttmnTkFJKeThNBEop5eE0ESillIfTRKCUUh5OE4FSSnk4lyUCEfnSsYTejltsFxEZJyL7HUvs3e+qWJRSSt2aK68IpgOtktneGlttsTzQF/jUhbEopZS6BZfNIzDGrBaR0sns0gH42rEAxwYRyS0iRXSpPaVS2PXrsGQJbN4MsdWGe/eGUqUgOBh++unm9/TvD0WKwKZNsGDBzduffx7y5oU1a+D332/e/uqrkCMHLF0Kq1ffvH3YMMicGRYuhA0bbtyWKROMHGkfz50Lf/5543YfHxg61D6eNQt27rxxe65c8PLL9vE338Dff9+4vUAB+M9/7OOpU+HIkRu3FysGzz5rH0+aBCdP3ri9bFno1cs+/ugjOH/+xu2VKkHXrvbx++/DlSs3bg8MhM6OtY3efBOiEi2MV6sWtG8P0dEwahQAkUTzT2EfKgwYhksYY1x2w9Zs33GLbQuABgmeLwNq3mLfvth671tKlixplFJOWrvWmLx5jbEpwBgRe1uzxm7/+uv41xLe/vrLbp84Ment+/fb7e+9l/T2f/+12994I+nt4eF2+/PP37zN2zs+/t69b96eO3f89sceu3l7iRLx21u3vnm7v3/89gYNbt5eu3b89qCgm7c3axa//b77bt7esWP89oIFb97evXv8dl/fm7cPGGCMMSbicqSJETFbCmOCnsUUHZLZXL522fm/+0SALeYWx2qXrkfguCJYYIypmsS2X7GLcv/heL4MeNUYszW5z6xZs6bRmcVKJcEY2LoVvv/ennX26AEXLtiz3yeegObNIUsWd0epkmEMbNkC06fDdz9EcCFwFNT/gPzZ8vPZw5PoVLnTXX+2iGw1xtRMaps7S0yEYteMjVUcu66sUupO7NljD/7ff2+bQTJnhpdestty57bNIypNMwZEYMUKaNrUtn7l/k9HyL6YXoFP82HL/5HHN4/Lvt+diWAeMEhEZgK1gTCj/QMqo7t8GebNs6d9I0aAn59tv1+06OZ933nHHhHmzYOVK2/ePnasbU9/9lnbVt+kCQweDJ06QR7XHTRUyrh2zf7VTp8O998Pb70F1WtfYvLUzDze2YfgC0OIjH6J5vc1d3ksLksEIvI90BjILyKh2EW/MwMYYyYDC4E2wH4gHHjaVbEo5VbXr9sD/fff2//54eHg62s7NP38bGfo1Kk3v2/kSJsINm1KevtYx/LE48dD/vxQtKhLf4ZKGVu3wrRp8N13tp+5WDF7FbB4/2L6LujLU9WeInfut2mcu3GqxZTu1izWPgKVLkRHw9Wr8SNnmjeHfPngscfgySehfn17Nq88wpkzNleD/eufO9deuPXsCdXrneOVpS/y1bavqJS/El88/AX1StRL8RiS6yPQf4lKpRRj7Nn7Cy9AiRJxQ/9o3NgOkzxxwg5HbNhQk4AHiIiA2bOhbVsoXBh27bKvjxljR6TOmAFe5ZZRbbI/M7bP4PWGr/PXs3+5JAncTrpbj0CpJM2caXvaEpswwXaeJrXdy8semME21K5ff+P27Nnhww/t488+u3k8e758th0f7H6TJsGBA3ZkTps28NBDdpu3N7RufU8/T6Uf//5rpwd8/71t+ile3E6ryJ3bbi+RYIhMwewFKZOnDIueWkRQ4SB3hAtoIlAZxfnz8MMPtk09oXHj7P327bZ9PqHMmeMTwV9/3bw9T574RLB5M/z6643bixePTwSbN0OZMnaiU6dO8f/rlUc4ccKe5Vevbv8Jzpxpc3+vXvZ8wMvL7meM4attX/HniT8Z13oc1QpVY90z6xARt8avfQQqfbpyxXam+vvD009DTIwdf+fm/1DKc0RE2HOHr76yYwGqV7eDwcCOCMqa9cb9/zn/D88ueJbfD/5Ow5INWfzUYnwz+6ZavNpHoDKWBQtsAhg7Nr68QKZMmgRUqhk3zlbgePxxCAmBIUPsKKBYCZNAdEw04zaOo+qnVVkfup5JbSaxstfKVE0Ct6NNQ57u3XehfHno0sU2rwwadPM+XbvaHq+TJ+MnKiX0zDN2/Ns//8Abb9y8fcAAO0pm924YPfrm7S++CDVq2OaZ2CGRCb3+uj3wr1tna9QsX26fr15tO16VcrETJ+Dbb+3FZ/78tsxSmzY3N/0k5Uz4GYavGE6jUo2Y3G4yJf1KplrcztJE4MnCwmyb9qOP2kQQGWlHvSTWpIm9j4hIenu7dvY+PDzp7Y8/bu8vXUp6+4UL9v78+aS3X7xo78+ehePHbbv8Sy9puQTlUrFNP9Onw+LFtvWxRAlbreOpp+ztViKjI5mxfQY9AntQKEch/nz2T8rkLuP2voBb0T4CT7Z2LTRoAPPnxx/MlVKEhdkio+fO2TEBPXrYMf8VKtz+vVuPb+WZec8Q8m8Ii7otomW5lq4P2AlptdaQcrft2+19QIB741DKzY4ft00/J07YytJ+fnY6SJ069oI4uaafWFcjrzJq1SjGrhtLwewF+fnxn9NMErgdTQQZ2TvvwMaNN75WuLAdEw/wwQe2dnvCgc1KeYikmn4aN7aTwr28ku7uSk7HWR1ZcmAJfar34YMWH5DbJ7cLonYNTQQZzciR8MADtifr9OmbF92IiYl/3KyZ7flKo+2WSqW02IUZMmWyU0Ref902/bz2mm3+cabpJ6GL1y6SxSsLPt4+DG0wlFfrvUrTsk1dE7wLaR9BRhISAkFB8MordmUkpRQQ3/Qzfbqt8tm5s31t587bj/q5lYV/L6Tfgn48FfAU7zR9J8VjTmnaR+ApXnvNNm4OGeLuSJRyu+houwrn9Ol2wldMjB3FnDOn3V606N0VbD0TfoYXFr/AtyHf4l/An4crPpyicbuDJoL04uJFO2g5LCz+teHDoVEjO53xpZfsuPr339da9MpjGQOhobbbK1MmOzo6IsKeI/XsaafM3IvfD/xOt5+6cT7iPMMfHM7QhkPJ6p319m9M4zQRpBerVsHPP0PBgvENmbHNejEx9ta9e/yi3Ep5kIRNPydP2uc+PrYCePHid9f0k5QiOYtQIV8FPm37KdUKVUuZD00DNBGkJ9Wr24QQe20b64EH7ApVSnmYzZvtQm+xo37q17cT1WOVKnVvn2+M4Yu/vuCvE38xse1Eqhasypqn16TZiWF3SxNBetG+vb0p5cGMsQf/PHlsM090NOzYkXJNPwkdPH+Q/5v/fyz/ZzmNSzfmauRVfDP7ZrgkAJoI0q61a+3on+ho+7xRI7uihVIeKGHTz+7dtnzVxIlQu7YtcZVSTT8QXyTu9eWv453Jm8/afUaf+/uQSTJujU5NBGnVBx/YJY3q1rXPc+RwbzxKuUnXrjBrlm36adDALt/86KN2m0jKJgGwo4JGrRpF07JN+bTtpxTPVTxlvyAN0kSQFkVEwB9/QJ8+SVfjVCqDil3tc/58O95fBCpWdE3TT0LXo6/zbci39ArqRaEchQjuF0wpv1IZshkoKZoI0iIfHzh61C5+rpQHOH4cvvnGNv3s2QO+vvEH/hEjXPvdm49t5pl5z7Dj1A6K5ypOi/taUDp3add+aRqTcRu90jtfX1v0XKkMbvVqO+5/yBBb8WTqVDsE1FVn/7HCI8N5ecnL1PmiDuevnmfeE/NocV8L135pGqVXBGlNVJSd8/7cc3aNAKUykNimn+nTbZPP88/bDt/hw6FbNyhXLvVi6TCzA0sPLqXv/X0Z03wMfj5+qfflaYxeEaQ1f/9t5wSEh7s7EqVSzLFjdtK7v78t7fzVV/Dvv3Zb1qy2+Sc1kkBYRBgRUREADHtwGMt7LOez9p95dBIATQRpT0iIvdc1AlQ6FxkZ/3jgwJubft59N3XjWbBvAVUmVWHUylEAPFjqQZqUaZK6QaRRmgjSmu3b7Xi4ypXdHYlSd8wY2LAB+veHQoXg0CH7+ujRsG+fvdjt3dsug5FaTl85Tdcfu9L++/bk9c1Lp8qdUu/L0wntI0hrQkKgUiV7vaxUOnH+vF3vaPp02LvXjnXo3Nl2eQFUreqeuJYcWEK3n7oRFhHGqMajGNJgCFm8dK3rxDQRpDXFi7t+uIRSKSAiwjbxlC5tD/jDh9uO31desRO+UvOs/1aK5SxG5fyV+bTtp1QpWMXd4aRZujCNUsppxtjVT6dPh5kzITDQ1kEE2/lbqJBbwyPGxDD1z6n8deIvPm33qXuDSWN0YRql1D375ht4++0bm3569Yrf7u4ksP/cfv5v/v+x8tBKmpRuElckTt2edhanJStX2qahrVvdHYlSXL1qz/pj10K6cMEuh/HFF7ZJ6JtvoGkaWJ43Oiaa/637HwGfBvDniT/5vP3nLOuxTJPAHXBpIhCRViKyV0T2i8hN6yeKiJ+IzBeRbSKyU0SedmU8ad61a3bA9fXr7o5EeajYUT/9+kGRIvDkk/DLL3bboEF2FvAzz6SN9v9YZ8LPMHrNaJrf15xdA3bR5/4+HlMjKKW4rGlIRLyAiUBzIBTYLCLzjDG7Euw2ENhljGkvIgWAvSIywxijR0KlUllYmO3sjW366dLFNv00bmy3p6Vj67Woa3y97Wt639/bFol7NpiSfiU1AdwlV/YRPADsN8YcBBCRmUAHIGEiMEBOsX97OYBzQJQLY0q73njDNsCCXWxVKRc4c8YWdUt4y5/fdv76+dmD/quv2iSQls76E9oYupHe83qz8/ROSuUuRYv7WlAq9z0uRebhXJkIigFHEzwPBWon2mcCMA84DuQEHjfGxCT+IBHpC/QFKFmypEuCdbsnnoArV2wjbPXq7o5GpWORkXDwoD2z37MHzp2D996z27p1gyVL7GMfHztSOeF/qcmTUz9eZ125foVhK4bx8YaPKZarGL92/dVji8SlNJcNHxWRR4GWxpg+jufdgQeMMf9JsE8XoD7wInAf8DsQaIy5eKvP1eGjSlnnz9uD/d690L27vZAcOtSuaRSV4Lq6WDE4fNhOWF+50o7/r1jRJoCUXtTFlZp/05ylB5fSv2Z/3mv2HrmyptFLljTKXcNHQ4ESCZ4Xx575J/Q08J6x2Wi/iPwDVAI2uTCutGnlSltork0bd0ei0pDoaHsQL1rUnsHPn2/XKtqzB06dit/voYdsKefYCV2VKtmDfcWKkDt3/H6x7f3pxYWIC2T1yopvZl+GPzicYQ8O48FSD7o7rAzHlYlgM1BeRMoAx4AngK6J9jkCNAXWiEghoCJw0IUxpV0ffgihoZoIPNyxYzB3LmzbBsHBdmH2q1ftEtb16tnEEB0N7drZg33srWhR+/4OHewtI5i3dx79f+1P94DuvNfsPRqWaujukDIslyUCY0yUiAwCFgNewJfGmJ0i0s+xfTLwFjBdRLYDAgw2xpxxVUxKpQXG2GJs27bFH/D794cWLWzb/qBBdk2iwEB49llbp6dsWfvejh3tLSM7deUUz/32HLN2ziKgUABd/HVdDldz6cxiY8xCYGGi1yYneHwc0N4elWFdvQo7d0K2bLYW//HjtrDsRUcvmIjtsD1/3j5/4AG7SmmxYmlruGZqWbR/Ed1+6sbl65d5q8lbDK4/mMxemd0dVoanJSZc6do12xvn7W0fJ7XYTK5cdp+ICHuqqNI1Y2wb/l9/2bP9PXsgJsaWXp46FQoXtmvxVqkCQUH2bD979vj3Z81qJ5d7qhK5SlCtYDUmtZ2EfwF/d4fjMTQRuMrly/Z6/rffoEYNmDbNXv8ntm+fPSXMlg2yaHnc9CAy0o7UiW3a2bbN1tn5+mt7Fj91qs3rgYHQqZO9r1XLvjdTJhg3zr3xpyUxJobPtnxG8MlgPmv/GVUKVmFlr5XuDsvjaCJwlblz4fRpO4MHbE/fxx/fvF/+/PZ+2DB76qjSlPPn7YE+NBSeesq+1ro1LFtmH2fNas/uAwPj3xMcbGfmquTtO7uPPvP6sObIGpqXbU5EVAQ+3j7uDssjaRlqV2nVyrYLHDyoM4XTgZgYezYvAnPm2LP7bdvgyBG7PXNme5GXJYutvXP5sj34V6xotynnRcVE8b91/2PEyhH4Zvblo5Yf0TOwp5aHcDEtQ53a9uyBxYvt7B5NAmlOeLhdETQ4OL5pJyTEttIVKWJH9OzfD/Xrw4AB9oAfFBTfcpdRhme6y9nws7y/9n3alG/DxDYTKZKziLtD8niaCFxh9Gh7362be+PwcMbYcfmxB/wnnoD77rNn/D172n38/OyB/umn41vmXn7Z3lTKuRZ1jenB0/m/Gv9HoRyF2NZvGyX8Stz+jSpVaCJISTEx9gpg4ECbBPx11ENquX7dDszKmRP+/tuOv9+2zdbZiVWunE0EzZrZLpzAQChVyjOHaaam9UfX03teb3af2c19ee+jWdlmmgTSGE0EKeXyZTs05M037YKtymWiouzyiLGTsbZtg927YcQIeP11W1Lh8mU7YicoyB7wAwLiq2kWLarNO6nh8vXLvLH8DcZtHEcJvxIs6raIZmWbuTsslQRNBCnlo49s30AJPdNJKdHR9uw+9oBfsmT8CNy2be0VQNGi9kDfti00aWK3FSgAmzyvWlWa03FmR5b9s4xBtQbxTtN3yJk1p7tDUrego4ZSSv78tncxdjkndUcuXbLt+ZUq2eedOsGiRXZmLtg5eV27wldf2ecbN9pmntjRtyptOH/1PD7ePvhm9uWPI38A0KBkAzdHpSCFRg2JSHZjzJWUCyuDuXRJ+wTuwJo1sGJFfNPOwYN2/t2BA3Z7lSq2/T62aadyZTtmP1btxCtbKLf7afdPDFw4kB4BPXi/+fuaANKR2yYCEakHTMWuIFZSRAKBZ40xA1wdnErfYuvsxA7R3L3bnuV7ecG338Lnn9tJ1TVq2HVwg4Li3/vWW24LW92hk5dPMmjhIH7c/SNBhYN4ouoT7g5J3SFnrgg+AlpiVxLDGLNNRDyzILgx9sgWEXHj66VK2fX9GnjuGdClS7BunW0dy5EDJkyA55+37fxg6+kEBNhRPAUK2BG2H354Y50dlf789vdvdPupG+GR4bzz0Du8XO9lLRKXDjnVNGSMOZpo1l+0a8JJ4xYtSnq9gM8+88hT2Kgo+P13e3b/88/2CmDFCrv4Sc2a8Npr8ZOxypa9cW5dgQLuilqlpFK5S1G9SHUmtplIpfyV3B2OukvOJIKjjuYhIyJZgOeA3a4NK41q1AhWr7b1gxLWFahWzX0xucmBA7Z80qlTkCePnaDVubNNAAB16tibylhiTAyTNk9i28ltfP7w5/gX8GdZj2XuDkvdI2cSQT/gE+xi9KHAEsAz+weyZYOGnrlK0j//wIwZdrnEl1+GMmXsWPy2bW0RNi2cmvHtPbOX3vN6s/boWlre11KLxGUgziSCisaYG2oliEh9YK1rQkrDfvzRXhEMG+YR4xbPnYPZs+Gbb+xSiRA/Vy5TJpgyxX2xqdQTGR3J2HVjGbVqFNkyZ2N6h+n0COyhReIykNvOIxCRP40x99/utdTitnkEERHxtYWPHs2wq4dcu2bP7kXsYipffmmHbnbvbsfxlyrl7ghVajt15RSVJlSiadmmjG89nsI5Crs7JHUX7moegYjUBeoBBUTkxQSbcmHXIPYskZH2vmfPDJcEYmLgjz9sp+/s2bB8OVSvbgdCDRpkO3v15M+zRERF8OVfX9KvZj8KZi9ISP8QiufKWP/uVbzkmoayYOcOeAMJ54ZfBDx3NekM1DF8/rxdVnHGDDh82A7l7NQpfuJWxYrujU+5xx9H/qD3vN7sO7uPCvkq0KxsM00CGdwtE4ExZhWwSkSmG2MOp2JM6g6Ehdna+hcuxL+WM6cd4AT2TD/hNrDLKtaqZQ/4kybZ0T3vvGM7f3Vcv+e6dO0Sry17jYmbJ1I6d2mWPLVEi8R5CGc6i8NF5AOgChA3RMAY85DLokqLfHzgu+9sm4kbREfbxVJCQuDECXjuOft6x46wcuWN+1atapMDwCuvwIYNSW/Pls0uwagHfwXQcVZHVvyzgv/W/i+jHxpNjiw53B2SSiXOdBYvAWYBL2OHkvYEThtjBrs+vJu5pbP41Ck7byBPnlT5urNnIW9e2y7/+ed2dM6OHfETmn194eJFW4jt999tLf5CheLf7+tra/WALetwJVGFqLx57QQvpc5dPYePtw/ZMmdj3dF1CELdEnXdHZZygXstOpfPGPOFiPw3QXPRqpQNMY374AOYPBlOnkzx0+djx+wZfUhI/O34cXsrUsTO3s2Tx651ExBgb5Ur2yQA0Lx58p9fuXKKhqsykDm75jBw4UB6BvZkTPMx1CtRz90hKTdxJhE4hstwQkTaAseBjN9ztGGDXbwWbJNQkya3TQKXL8PChXYUTu3adtLVmTOwdKndHrt0YkgIDB5sz9pXrLBDM7NkscVLmzWzB/vYCVr9+8fX4FcqJZy4dIKBCwfy856fqVGkBt2q6ZKqns6ZRDBaRPyAl4Dx2OGjz7syKLczxra5fPhhfE/rhAm3fdtjj8Fvv9nH06bZRPD33/DkkzfuV6yYXcmyShVbumj7djtCJ7PW6lIu9uu+X3nq56eIiIrg/Wbv82LdF/HOpOtTebrb/gswxixwPAwDmkDczOKMS8T2ssZOo82aFUqXTvYtK1bYJDBsmJ14VaSIfT0oyLbTxypQAPLli3+eN6+9KZUayuYpS62itZjQZgIV8lVwdzgqjbhlZ7GIeAGPYWsMLTLG7BCRdsBQwNcY45bhM6nSWWyM7SDOnt3WVHZi9zp1bLv+vn3xE5CVcrfomGgmbJpAyL8hfNHhC3eHo9wouc7iTEm96PAF0AfIB4wTkWnAWGCMu5JAqtm3DwoXtk1DThoyBMaP1ySg0o5dp3fRcFpDnl/8PCevnCQiKuL2b1IeKbmmoZpAgDEmRkR8gDNAOWPMydQJzY1eecXeF3aupooIPPKIC+NR6g5cj77OmLVjeGv1W+TMkpNvH/mWrtW6apE4dUvJXRFcN8bEABhjIoB9d5oERKSViOwVkf0iMuQW+zQWkWAR2ZlmhqVGRto+gb59b7vrb7/ByJHxi6wr5W4XIi7w0YaPeKTSI+wauItuAd00CahkJXdFUElEQhyPBbjP8VwAY4wJSO6DHX0ME4Hm2HUMNovIPGPMrgT75AYmAa2MMUdEpODd/5QUVvD2oRgDI0bYCWBvvJEKMSl1C1cjr/LFX18woNYACmYvyPb+2ymas6i7w1LpRHKJ4F6nIj0A7DfGHAQQkZlAB2BXgn26Aj8ZY44AGGNO3eN3poxq1ezU3dtYtgw2b7YrVXrrCDzlJqsPr6bPvD78fe5vKuevTNOyTTUJqDuSXNG5ey00Vww4muB5KFA70T4VgMwishJb4fQTY8zXiT9IRPoCfQFKlix5j2E5YcwYp3Z75x07TLRnTxfHo1QSLl67yJClQ/h0y6eUyV2Gpd2X0rRsU3eHpdIhV57HJtUomXisqjdQA2gK+ALrRWSDMWbfDW8yZgowBezwURfEesfWr7dzB/73v/iyzUqlpo4zO7Ly0EpeqPMCbzV5i+xZtHqgujuuTAShQIkEz4tjy1Mk3ueMMeYKcEVEVgOBwD7cqU8fW9959uxb7pIjBzzxhFP9yUqlmDPhZ8iWORvZMmfj7YfeRkSoU7yOu8NS6Vxyo4biiIiviNzpMiWbgfIiUkZEsgBPAPMS7fML0FBEvEUkG7bpaDfuduwYHDmS7C7VqsH33zs130ype2aMYeaOmVSeWJkRK0YAULdEXU0CKkXcNhGISHsgGFjkeB4kIokP6DcxxkQBg4DF2IP7D8aYnSLST0T6OfbZ7fjcEGATMNUYs+Muf0uqmToVDhxwdxTKUxy7eIyOszry5I9PUiZ3GXoE9nB3SCqDcaZpaCR2BNBKAGNMsIiUdubDjTELgYWJXpuc6PkHwAfOfF5a8Pff8Oyz8PLL8P777o5GZXQL9i2g20/diIyOZGzzsTxf53m8MnnekuHKtZxJBFHGmDCdkGKNGWNLRL/4orsjUZ6gXN5y1CtRj/Gtx1Mubzl3h6MyKGcSwQ4R6Qp4iUh54DlgnWvDcrMHHoBLl256OTQUvvrKdhAnXBFMqZQSHRPNuI3j2PbvNqZ3nE6l/JX4rdtv7g5LZXDOJIL/AK8D14DvsG3+o10ZlNuNGpXky/Pn2+oTsesFK5WSdp7aSe95vdl4bCNty7clIioCH2+f279RqXvkTCKoaIx5HZsMPFpUlF36sXx5d0eiMpLr0dd574/3GL16NH4+fnzX6TueqPqE1gdSqcaZxetXAEWA2cBMY8zO1AjsVlJlPYKnnrIrky1YcNtdlbpXp66cwn+iPy3LteTjlh9TIHsBd4ekMqC7XY8AAGNME6AxcBqYIiLbRSRjllibMwcGDYLVq+H06Rs2RUXZtWqUSgnhkeF8suETomOi44rEzeg0Q5OAcgunJpQZY04aY8YB/bBzCoa7Mii3GT4cpkyB8HCoWzfuZWPsAvK1asUvYazU3VrxzwqqfVqN5xc/z8pDKwEokrOIe4NSHu22fQQiUhl4HOgCnAVmYheyz3h697ZV5Lp2veHlUaPsJLKhQyF3bveEptK/sIgwXv39Vab8OYX78tzHip4raFy6sbvDUsqpzuJpwPdAC2NM4lpBGctLN+e3KVNsIujVC0Zn7LFSysU6zurI6sOreaXeK4xsPJJsmbO5OySlACcSgTHGc4qZXLhgFxZwFBBassQ2CbVubROCDuJQd+r0ldNkz5KdbJmz8W7Td/ESL2oVq+XusJS6wS37CETkB8f9dhEJSXDbnmDlsoylbl3bPORQsCB88IEtQpo5sxvjUumOMYbvtn93Q5G4OsXraBJQaVJyVwT/ddy3S41A0qKgIHtT6k6EXgyl/6/9WbBvAbWL1aZXUC93h6RUsm55RWCMOeF4OMAYczjhDRiQOuG5z9WrsHAhnDvn7khUejJv7zz8J/qz/J/lfNTyI9Y+s5YqBau4OyylkuXM8NHmSbzWOqUDcbs5c2wxIYcdO6BtW1i1yo0xqXSnQr4KNCjZgO39t2ulUJVu3LJpSET6Y8/8yybqE8gJrHV1YKnuzBkoVQpatAAgxPGLAwLcGJNK86Jiovh4w8eE/BvC1498TaX8lVjYbeHt36hUGpJcH8F3wG/Au8CQBK9fMsZknAaT0FDYvduuQN+vX9zLISGQPTuUKePG2FSaFvJvCL3n9WbL8S10qNhBi8SpdCu5piFjjDkEDAQuJbghInldH1oqmT/fXgUkKikREmKXo8zk1Nxr5UmuRV1jxIoR1JhSgyNhR/ihyw/8/PjPmgRUunW7K4J2wFbAAAlH0RugrAvjSh379sGAAZArF5QoEfeyMTYRdOnixthUmnXx2kUmbZnEk1Wf5KOWH5EvWz53h6TUPbllIjDGtHPcZ9zGkTlz7H3//jfNFlu1yq5EphTAletXmLJ1Cs/Vfo4C2Quwo/8OCuXQ1YlUxuDM4vX1RSS74/FTIvKhiJR0fWipYPt2KF0a3nvvhpdFoGpVqFDBPWGptGXZwWVU+7QaLy55kVWH7TAyTQIqI3GmBfxTIFxEAoFXgcPANy6NKrV07GhXoU9k6VJbZO42SzWoDO5CxAX6zOtDs2+a4Z3Jm1W9VvFQmYfcHZZSKc7ZxeuNiHQAPjHGfCEiPV0dWKp4/PEkX54+3S5J0KdP6oaj0pZHZj3CmsNrGFx/MCMajcA3s6+7Q1LKJZxJBJdE5DWgO9BQRLyA9F95Jzoajh6FvHltZ3ECISE6f8BT/Xv5X3JkyUH2LNl5r+l7eGfypkbRGu4OSymXcqZp6HHswvXPGGNOAsWAD1waVWo4e9ZOEvj22xtevn7dTivQROBZjDF8s+0b/Cf5M2KlLRJXu3htTQLKIzizVOVJYAbgJyLtgAhjzNcuj8xN9u61y1JqIvAcR8KO0Pa7tvSY24OK+SrSu3rv279JqQzEmVFDjwGbgEeBx4CNIpJhR9jv3WvvNRF4hl/2/EKVSVVYfXg141qNY83Ta6hcoLK7w1IqVTnTR/A6UMsYcwpARAoAS4E5rgzMXbp0sa1Gfn7ujkS5kjEGEaFS/ko0Lt2Y8a3HUzp3aXeHpZRbONNHkCk2CTicdfJ96VbevOClRSMzpKiYKN7/4326/9wdgIr5KzL/yfmaBJRHc+aAvkhEFotILxHpBfwKpP/yijlywPjx0LDhDS/36QO//OKmmJRLbTu5jdpTazNk2RDCI8OJiIpwd0hKpQnOrFn8ioh0Ahpg6w1NMcb87PLIXC1bNhg06IaXzp6FL76ASpXcFJNyiYioCEavHs37a98nn28+5jw6h87+nd0dllJpRnLrEZQHxgL3AduBl40xx1IrMJeLirLjRIsWhXy2aNg//9hN5cu7MS6V4i5du8RnWz+jW7VufNjyQ/L6ZpziuUqlhOSahr4EFgCdsRVIx9/ph4tIKxHZKyL7RWRIMvvVEpHoVB2NdO6cHRo0a1bcS0eP2vsEhUhVOnX5+mXGrhtLdEw0BbIXYNeAXUzvOF2TgFJJSK5pKKcx5nPH470i8uedfLBjBvJE7FKXocBmEZlnjNmVxH7vA4vv5PNdIXalyuLF3RuHujdLDiyh7/y+HAk7Qo0iNWhSpgkFshdwd1hKpVnJXRH4iEh1EblfRO4HfBM9v50HgP3GmIPGmOvATKBDEvv9B/gROJXEtlR1/ToULAgF9JiRLp27eo6nf3malt+2xMfbhzVPr6FJmSbuDkupNC+5K4ITwIcJnp9M8NwAtyvDWAw4muB5KFA74Q4iUgx4xPFZtW71QSLSF+gLULKk6ypgv/SSvan06ZFZj7D2yFqGNhjKsEbDdMUwpZyU3MI093oqJUm8lriw88fAYGNMtEhSu8fFMgWYAlCzZk0tDq3inLx8kpxZcpI9S3Y+aP4BWbyyEFQ4yN1hKZWuuHJiWCiQsNu1OHA80T41gZkicgjoAkwSkY4ujClezpwwbRo8FH9h8+ijMHFiqny7ukfGGKYHT8d/oj/DVwwH4IFiD2gSUOouOFNi4m5tBsqLSBngGPAE0DXhDgmXwRSR6cACY8xcF8YUz9cXevWKexoTYyeS3Xdfqny7ugeHLhzi2QXPsuTAEhqUbEDfGn3dHZJS6ZrLEoExJkpEBmFHA3kBXxpjdopIP8f2ya76bqc89JDtFXYMHz11CiIjdehoWvfz7p/p/nN3RIQJrSfQv1Z/MkmGrniilMvdNhGIbbzvBpQ1xrzpWK+4sDFm0+3ea4xZSKJyFLdKAMaYXk5FnFIuXbJXBQ46dDRtiy0SV6VgFZqVbcYnrT6hVO5S7g5LqQzBmVOpSUBd4EnH80vY+QEZiiaCtCkyOpJ31rxDt5+6AVAhXwXmPjFXk4BSKciZRFDbGDMQiAAwxpwHsrg0KjfIlAmqVdOmobTkzxN/8sDUB3h9+etEm2iuRV1zd0hKZUjOJIJIx+xfA3HrEcS4NCo3ePhhu1ZxwYLujkRdjbzKa0tf44HPH+Dk5ZP8/PjPzOoyi6zeWd0dmlIZkjOdxeOAn4GCIvI2dpjnGy6NKjW0b3/TovUqbbgSeYUv/vqCnoE9GdtiLHl887g7JKUyNDHm9vOzRKQS0BQ7SWyZMWa3qwO7lZo1a5otW7ak+Oc+8QTkzw8TJqT4RysnXLp2iU+3fMpLdV/CK5MXZ8LPkD9bfneHpVSGISJbjTE1k9rmzJrFJYFwYD4wD7jieC1D2bQJzp93dxSeadH+RVT9tCpDlg5hzZE1AJoElEpFzjQN/YrtHxDABygD7AWquDAu16tTx84jmD+fmBg4dkw7ilPb2fCzvLjkRb7e9jWV81dm7TNrqVuirrvDUsrjOLNCWbWEzx2VR591WUSpJTraTicGTp+2lUd16Gjq6vRDJ9YdXcewB4fxesPXtTNYKTe545nFxpg/ReSWlULThdhpxA6xcwj0isD1Tlw6Qc6sOcmRJQdjm48li1cWAgsHujsspTyaMzOLX0zwNBNwP3DaZRGlhnLl7MziMrbUkZcXtG6tS1S6kjGGacHTeHHxizxT/Rk+bPkhtYql7/MJpTIKZ64IciZ4HIXtM/jRNeGkkg8/tFcEjsqjQUGwcGHyb1F37+D5gzy74FmWHlzKg6UepF/Nfu4OSSmVQLKJwDGRLIcx5pVUisf1/v4b2rWDwoXdHYlH+Gn3T3T/uTte4sWnbT+lb42+WiROqTTmlv8jRcTbGBONbQrKOOrVg7feuuGlp5+GBx90UzwZVOz8lGoFq9GqXCt2DthJv5r9NAkolQYld0WwCZsEgkVkHjAbuBK70Rjzk4tjc43ISFtYKIGDB90USwZ0Pfo6Y9aOYefpnXzX6TvK5yvPj4+l75ZEpTI6Z07P8gJnsesKtwPaO+7Tn3//hbAw21mcQGiojhhKCVuOb6HW57UYtmIYYJOCUirtS+6KoKBjxNAO4ieUxUqf6waHhNj7gIC4l4yxiUDnENy9q5FXGbFyBP9b/z8K5yjML0/8wsMVH3Z3WEopJyWXCLyAHDi3CH36EJsIqsXPkdPJZPfuSuQVpgdPp3f13oxpPobcPrndHZJS6g4klwhOGGPeTLVIUkOHDrasRP74OjbGQN++UDPJUkzqVi5eu8ikzZN4pd4r5M+Wn90Dd5MvWz53h6WUugvJJYKkrgTSt3LlbuofKFQIPvvMTfGkU7/u+5V+v/bj+KXj1Cleh8alG2sSUCodS66zuGmqRZEaoqLgu+9sdbkErl61ZYfU7Z2+cppuP3Wj3fft8Mvqx7pn1tG4dGN3h6WUuke3TATGmHOpGYjL7dsH3brB8uU3vDxqlF2fJibDrbmW8jr/0JnZO2czstFI/nz2T2oXr+3ukJRSKeCOi86lW3v32nt//xteDg21y1Nm0nlOSTp28Rh+Pn7kyJKDj1p+RFbvrFQtWNXdYSmlUpDnHP6iouy9j88NL+vQ0aQZY/h86+f4T/Jn+IrhANQoWkOTgFIZkOckgls4elQnkyV24NwBmn7dlL4L+lKjSA0G1hro7pCUUi7k0YlAJ5PdbM6uOVT7tBpbT2xlSrspLOuxjPvy3ufusJRSLuQ5fQQPPQQbNsStQQC27NAbb0D9+m6MK40wxiAiBBYKpG2FtnzU8iOK59IMqZQnkNgqkelFzZo1zZYtW9wdRoZxPfo67655l11ndjGz80xEMt70EaUUiMhWY0ySU2c9p2non3/gyy/h/Pm4ly5cgOPHPXfo6KZjm6gxpQYjV43EO5O3FolTykN5TiLYsgV697ZHfofvvoNixewSxp4kPDKcl5e8TN0v6nL+6nnmPzmfGZ1m6OLxSnkoz+kjSMLRo5A5s51H4EmuRl7l25Bv6Xt/X95v/j65suZyd0hKKTdyaSIQkVbAJ9hKplONMe8l2t4NGOx4ehnob4zZ5sqYEgoNtVcEnjCZLCwijAmbJjC4wWDyZcvH7oG7yeObx91hqTQsMjKS0NBQIiIi3B2KugM+Pj4UL16czJkzO/0elyUCx3rHE4HmQCiwWUTmGWN2JdjtH6CRMea8iLQGpgCpVrfg6FHPGDo6f+98+v3aj5OXT1K/ZH0al26sSUDdVmhoKDlz5qR06dI6iCCdMMZw9uxZQkNDKZNghOTtuPJc+AFgvzHmoDHmOjAT6JBwB2PMOmNMbO/tBiBVD8sZfQ7B6SunefLHJ3l45sPk883Hxj4btUicclpERAT58uXTJJCOiAj58uW746s4VzYNFQOOJngeSvJn+72B35LaICJ9gb4AJUuWvLtoWrSAHTvgvvjJUaNG2TLUGVXnHzqzIXQDbzZ+k8ENBpPFK4u7Q1LpjCaB9Odu/s5cmQicXtlMRJpgE0GDpLYbY6Zgm42oWbPm3U188POztwS6dburT0rTQi+GktsnNzmy5ODjVh+T1SsrVQpWcXdYSqk0zJVNQ6FAwio+xYHjiXcSkQBgKtDBGHPWZdH8/Td88gmctV9x/jxs3Ajh4S77xlQVY2L4bMtn+E/0Z9hyu3j8/UXu1ySg0jUvLy+CgoKoUqUKgYGBfPjhh8Tc5cSf4cOHs3Tp0ltunzx5Ml9//fXdhgrA9u3bCQoKIigoiLx581KmTBmCgoJo1qzZPX2uyxljXHLDXm0cBMoAWYBtQJVE+5QE9gP1nP3cGjVqmLvyww/GgDE7dhhjjJk3zz7dtOnuPi4t2Xdmn2k0rZFhJKbpV03NgXMH3B2SygB27drl7hBM9uzZ4x7/+++/pmnTpmb48OFujMh5PXv2NLNnz77p9cjISJd/d1J/d8AWc4vjqsuuCIwxUcAgYDGwG/jBGLNTRPqJSD/HbsOBfMAkEQkWkVSrHXHU0XuR3juLZ++cTcDkAIJPBvPFw1/we/ffKZunrLvDUhlQ48Y33yZNstvCw5PePn263X7mzM3b7lTBggWZMmUKEyZMwBhDdHQ0r7zyCrVq1SIgIIDPEqw5O2bMGKpVq0ZgYCBDhgwBoFevXsyZMweAIUOG4O/vT0BAAC+//DIAI0eOZOzYsQAEBwdTp04dAgICeOSRRzjvqEjQuHFjBg8ezAMPPECFChVYs2aNk392jRk6dCiNGjXik08+YevWrTRq1IgaNWrQsmVLTpw4AcCBAwdo1aoVNWrUoGHDhuzZs+fO/6DugkvnERhjFgILE702OcHjPkAfV8ZwK6Gh4O2dfjuLjaNIXPUi1elQsQMftvyQojmLujsspVyqbNmyxMTEcOrUKX755Rf8/PzYvHkz165do379+rRo0YI9e/Ywd+5cNm7cSLZs2Th37sbFFs+dO8fPP//Mnj17EBEuXLhw0/f06NGD8ePH06hRI4YPH86oUaP4+OOPAYiKimLTpk0sXLiQUaNGJdvclNCFCxdYtWoVkZGRNGrUiF9++YUCBQowa9YsXn/9db788kv69u3L5MmTKV++PBs3bmTAgAEsT7Sqoit47Mzio0fT52Sya1HXeHvN2+w+s5sfuvxAubzlmNllprvDUh5g5cpbb8uWLfnt+fMnv/1OGEehzCVLlhASEhJ3lh8WFsbff//N0qVLefrpp8mWLRsAefPmveH9uXLlwsfHhz59+tC2bVvatWt3w/awsDAuXLhAo0aNAOjZsyePPvpo3PZOnToBUKNGDQ4dOuR03I8//jgAe/fuZceOHTRv3hyA6OhoihQpwuXLl1m3bt0N33Xt2jWnP/9eeGwiSI9zCDaEbqD3vN7sOr2L7gHduR59XesDKY9y8OBBvLy8KFiwIMYYxo8fT8uWLW/YZ9GiRckOofT29mbTpk0sW7aMmTNnMmHChDs6686a1f6f8/LyIip25UMnZM+eHbCJrEqVKqxfv/6G7RcvXiR37twEBwc7/ZkpJZ2dD9+Dtm3h8GEoXx6Ad96Bt992c0xOunL9Ci8seoF6X9Tj0rVLLOy6kK8f+VqTgPIop0+fpl+/fgwaNAgRoWXLlnz66adERkYCsG/fPq5cuUKLFi348ssvCXcMCUzcNHT58mXCwsJo06YNH3/88U0HXj8/P/LkyRPX/v/NN9/EXR2khIoVK3L69Om4RBAZGcnOnTvJlSsXZcqUYfbs2YBNGNu2pU7FHc+5IsiWDRJMRqtb142x3KGIqAhm7pzJgFoDeLfpu+TMmtPdISmVKq5evUpQUBCRkZF4e3vTvXt3XnzxRQD69OnDoUOHuP/++zHGUKBAAebOnUurVq0IDg6mZs2aZMmShTZt2vDOO+/EfealS5fo0KEDERERGGP46KOPbvrer776in79+hEeHk7ZsmWZNm1aiv2mLFmyMGfOHJ577jnCwsKIiori+eefp0qVKsyYMYP+/fszevRoIiMjeeKJJwgMDEyx774Vz1mYZtcu+OknePZZrmQrwKJFUK8eFCmS8jGmhAsRFxi/cTyvNXwN70zeXIi4QG6f3O4OS3mQ3bt3U7lyZXeHoe5CUn93ujANwM6dMGwYnDrFvn3QpQskaqJLM+bumYv/RH9GrRrFuqPrADQJKKVcxnMSQQKhofY+rXUW/3v5Xx6b/RiPzHqEgtkLsrHPRh4s9aC7w1JKZXCe00eQQGwiKFEi+f1SW5fZXdh0bBOjm4zm1fqvktnL+XriSil1tzwyERw9aieTpYWVyY6EHSGPTx5yZs3JuFbjyOqdFf8C/u4OSynlQTy2aahYMfDycl8MMSaGiZsmUmVSFYavGA5A9SLVNQkopVKd51wRdOgA585BrlyMHu3eBev3ntlLn/l9+OPIHzQv25z/1vmv+4JRSnk8z7kiyJIF8uQBLy9KloSaSQ6icr0fdv5A4ORAdpzawbQO01j81GJK5y7tnmCUSuNiy1BXrVqV9u3bJ1kX6G5Mnz6dQYMGpchnJdS4cWMqVqwYV4o6tvxFSjt06BDfffddin2e5ySCkBAYPBhz4iTjxtmnqSl2vkaNIjXoVLkTuwfupldQL10BSqlk+Pr6EhwczI4dO8ibNy8TJ050d0i3NWPGDIKDgwkODqZLly5OvedOSlWAJoK7t3cvjBlD2MGz/Pe/sGJF6nxtRFQEry97nS6zu2CM4b689/Fd5+8onKNw6gSgVEpxcx3qunXrcuzYMQA2bdpEvXr1qF69OvXq1WPv3r2APdPv1KkTrVq1onz58rz66qtx7582bRoVKlSgUaNGrF27Nu71w4cP07RpUwICAmjatClHjhwBbNnq/v3706RJE8qWLcuqVat45plnqFy5Mr169XI67nPnztGxY0cCAgKoU6cOIY6z0JEjR9K3b19atGhBjx49OH36NJ07d6ZWrVrUqlUrLsZVq1bFXWFUr16dS5cuMWTIENasWUNQUFCSM6PvlOf0ETicPGnvU2MOwbqj6+g9rzd7zuyhZ2BPLRKn1F2Kjo5m2bJl9O7dG4BKlSqxevVqvL29Wbp0KUOHDuXHH38E7FoCf/31F1mzZqVixYr85z//wdvbmxEjRrB161b8/Pxo0qQJ1atXB2DQoEH06NGDnj178uWXX/Lcc88xd+5cAM6fP8/y5cuZN28e7du3Z+3atUydOpVatWoRHBxMUFDQTbF269YNX19fAJYtW8bIkSOpXr06c+fOZfny5fTo0SOuvtHWrVv5448/8PX1pWvXrrzwwgs0aNCAI0eO0LJlS3bv3s3YsWOZOHEi9evX5/Lly/j4+PDee+8xduxYFixYkCJ/vh6bCFw5h+Dy9csMXTaUCZsmUMKvBIu6LaJluZa3f6NSaZkb6lDH1ho6dOgQNWrUiCvdHBYWRs+ePfn7778RkbjCcwBNmzbFz7E+ub+/P4cPH+bMmTM0btyYAgUKALYk9L59+wBYv349P/30EwDdu3e/4Sqiffv2iAjVqlWjUKFCVKtWDYAqVapw6NChJBPBjBkzqJmgE/KPP/6IS1IPPfQQZ8+eJSwsDICHH344LmksXbqUXbt2xb3v4sWLXLp0ifr16/Piiy/SrVs3OnXqRHEXnMV6TtOQw7//2ntXXhFcj77OnF1zGFhrIDv679AkoNRdiu0jOHz4MNevX4/rIxg2bBhNmjRhx44dzJ8/n4iIiLj3xJaJhhtLRTvbH5dwv9jPypQp0w2fmylTJqfb9ZOq5xb7HbGlqQFiYmJYv359XP/CsWPHyJkzJ0OGDGHq1KlcvXqVOnXquGTVMo9LBCdPumZlsnNXzzFy5UiiYqLI65uX3QN3M77NeK0UqlQK8PPzY9y4cYwdO5bIyEjCwsIoVqwYYPsFbqd27dqsXLmSs2fPEhkZGVfqGaBevXrMnGkXd5oxYwYNGjRI0dgffPBBZsyYAcDKlSvJnz8/uXLlumm/Fi1aMGHChLjnsc1HBw4coFq1agwePJiaNWuyZ88ecubMyaVLl1IsRs9JBJ07Q2Qk//eRP3v2pOxksh93/Yj/RH9Grx4dVyTOz8cv5b5AKUX16tUJDAxk5syZvPrqq7z22mvUr1+f6Ojo2763SJEijBw5krp169KsWTPuv//+uG3jxo1j2rRpBAQE8M033/DJJ5+kaNwjR45ky5YtBAQEMGTIEL766qsk9xs3blzcfv7+/kyebFf1/fjjj6latSqBgYH4+vrSunVrAgIC8Pb2JjAwMEU6iz2nDLULnLh0gkG/DeKn3T9RvXB1vuzwJUGFg9wdllIpQstQp19ahvpW/vwTBg5k/GvHWbgwZT7ysTmP8eu+X3mv6Xts+r9NmgSUUumS54waOnAAJk1iWpYBNLlelDZt7u5jDl84TF7fvOTMmpPxrcfj6+1LxfwVUzZWpZRKRZ5zReBw7frdDR2NMTGM3zieKpOqMGzFMACCCgdpElBKpXuec0WQwJ0OHd1zZg995vVh7dG1tCrXihfqvOCawJRSyg08MhHcyRXBzB0z6Tm3Jzmy5ODrjl/zVMBTWh9IKZWheE4iyJSJqMw+mEhx6oogxsSQSTJRq2gtHvV/lP+1+B+FcqTw5AOllEoDPKePoHNnvK9fZcsVf4oUufVuVyOvMmTpEDr/0DmuSNy3nb7VJKCUG6S3MtRRUVEMHTqU8uXLxxWKe/vtt1P8e1Ka5yQCh2zZINMtfvWaw2sI+iyI99e+Tz7ffETGRCa9o1IqVaS3MtRvvPEGx48fZ/v27QQHB7NmzZob6iClVZ7TNLRpEyF9J7C18zs8PezGtqFL1y4xZOkQJm2ZRJncZfi9++80K9vMTYEqlTY1nt74ptceq/IYA2oNIDwynDYzbh6T3SuoF72CenEm/AxdfrixNv/KXivv6Pvr1q0bV8J506ZNPP/881y9ehVfX1+mTZtGxYoVmT59OvPmzSM8PJwDBw7wyCOPMGbMGMCWoX733XcpUqQIFSpUiKsddPjwYZ555hlOnz5NgQIFmDZtGiVLlqRXr174+vqyZ88eDh8+zLRp0/jqq69Yv349tWvXvqm0RXh4OJ9//jmHDh3Cx8cHgJw5czJy5EjAriHQrl07duzYAcDYsWO5fPkyI0eO5MCBAwwcOJDTp0+TLVs2Pv/8cypVqsTs2bMZNWoUXl5e+Pn5sXr1anbu3MnTTz/N9evXiYmJ4ccff6R8+fJ39GeZmOdcERw+TMC2b9ixNuymTZExkczdO5fnaz/P9v7bNQkolcbElqF++OGHgfgy1H/99RdvvvkmQ4cOjds3ODiYWbNmsX37dmbNmsXRo0c5ceIEI0aMYO3atfz+++83VPmMLUMdEhJCt27deO655+K2xZah/uijj2jfvj0vvPACO3fujDvjT2j//v2ULFmSnDnvvL5Y3759GT9+PFu3bmXs2LEMGDAAgDfffJPFixezbds25s2bB8DkyZP573//S3BwMFu2bEmRaqQec0VgDAjxxebOhp/lk42fMLzRcPL65mXPwD1aIE6pZCR3Bp8tc7Zkt+fPlv+OrwAgfZahjjVt2jQ++eQTzp49y7p162653+XLl1m3bh2PPvpo3GvXrl0DoH79+vTq1YvHHnuMTp06AfbK6O233yY0NJROnTrd89UAuPiKQERaicheEdkvIkOS2C4iMs6xPURE7k/qc1LC5Sv2vlBhw+yds/Gf5M+7f7zL+qPrATQJKJUGpacy1OXKlePIkSNxVUGffvppgoOD8fPzIzo6Gm9vb2JiYuL2j405JiaG3Llzx5WfDg4OZvfu3YA9+x89ejRHjx4lKCiIs2fP0rVrV+bNm4evry8tW7Zk+fLlTv2u5LgsEYiIFzARaA34A0+KiH+i3VoD5R23vsCnrorn7Bk4nhOm5Povj815jBK5SrDl/7bQsFRDV32lUiqFpIcy1NmyZaN3794MGjQo7iAfHR3N9evXAShUqBCnTp3i7NmzXLt2LW51sVy5clGmTJm4mIwxbNu2DbAlqGvXrs2bb75J/vz5OXr0KAcPHqRs2bI899xzPPzww3H9JvfClVcEDwD7jTEHjTHXgZlAh0T7dAC+NtYGILeIJDO48+5diczCI496szV6LWOajWFDnw0EFg50xVcppVwgPZShfvvttylSpAhVq1alevXqNGzYkJ49e1K0aFEyZ87M8OHDqV27Nu3ataNSpUpx75sxYwZffPEFgYGBVKlShV9++QWAV155hWrVqlG1alUefPBBAgMDmTVrFlWrViUoKIg9e/bQo0ePu443lsvKUItIF6CVMaaP43l3oLYxZlCCfRYA7xlj/nA8XwYMNsZsSfRZfbFXDJQsWbLG4cOH7yqmbSe34ZvZlwr5KtzV+5XyJFqGOv260zLUruwsTqpBLnHWcWYfjDFTgClg1yO424D0CkAppW7myqahUCBhVZ/iwPG72EcppZQLuTIRbAbKi0gZEckCPAHMS7TPPKCHY/RQHSDMGHPChTEppe5AelvBUN3d35nLmoaMMVEiMghYDHgBXxpjdopIP8f2ycBCoA2wHwgHnnZVPEqpO+Pj48PZs2fJly+fVtxNJ4wxnD17Nm5ms7N0zWKlVJIiIyMJDQ29YYy+Svt8fHwoXrw4mTNnvuF1d3UWK6XSscyZM1OmTBl3h6FSgefUGlJKKZUkTQRKKeXhNBEopZSHS3edxSJyGri7qcWQHziTguGkB/qbPYP+Zs9wL7+5lDGmQFIb0l0iuBcisuVWveYZlf5mz6C/2TO46jdr05BSSnk4TQRKKeXhPC0RTHF3AG6gv9kz6G/2DC75zR7VR6CUUupmnnZFoJRSKhFNBEop5eEyZCIQkVYisldE9ovIkCS2i4iMc2wPEZH7k/qc9MSJ39zN8VtDRGSdiKT7VXpu95sT7FdLRKIdq+ala878ZhFpLCLBIrJTRFaldowpzYl/234iMl9Etjl+c7quYiwiX4rIKRHZcYvtKX/8MsZkqBu25PUBoCyQBdgG+Cfapw3wG3aFtDrARnfHnQq/uR6Qx/G4tSf85gT7LceWPO/i7rhT4e85N7ALKOl4XtDdcafCbx4KvO94XAA4B2Rxd+z38JsfBO4Hdtxie4ofvzLiFcEDwH5jzEFjzHVgJtAh0T4dgK+NtQHILSJFUjvQFHTb32yMWWeMOe94ugG7Glx65szfM8B/gB+BU6kZnIs485u7Aj8ZY44AGGPS++925jcbIKfYRRNyYBNBVOqGmXKMMauxv+FWUvz4lRETQTHgaILnoY7X7nSf9OROf09v7BlFenbb3ywixYBHgMmpGJcrOfP3XAHIIyIrRWSriPRItehcw5nfPAGojF3mdjvwX2NMTOqE5xYpfvzKiOsRJLWUUuIxss7sk544/XtEpAk2ETRwaUSu58xv/hgYbIyJziArbDnzm72BGkBTwBdYLyIbjDH7XB2cizjzm1sCwcBDwH3A7yKyxhhz0cWxuUuKH78yYiIIBUokeF4ce6Zwp/ukJ079HhEJAKYCrY0xZ1MpNldx5jfXBGY6kkB+oI2IRBlj5qZKhCnP2X/bZ4wxV4ArIrIaCATSayJw5jc/DbxnbAP6fhH5B6gEbEqdEFNdih+/MmLT0GagvIiUEZEswBPAvET7zAN6OHrf6wBhxpgTqR1oCrrtbxaRksBPQPd0fHaY0G1/szGmjDGmtDGmNDAHGJCOkwA492/7F6ChiHiLSDagNrA7leNMSc785iPYKyBEpBBQETiYqlGmrhQ/fmW4KwJjTJSIDAIWY0ccfGmM2Ski/RzbJ2NHkLQB9gPh2DOKdMvJ3zwcyAdMcpwhR5l0XLnRyd+coTjzm40xu0VkERACxABTjTFJDkNMD5z8e34LmC4i27HNJoONMem2PLWIfA80BvKLSCgwAsgMrjt+aYkJpZTycBmxaUgppdQd0ESglFIeThOBUkp5OE0ESinl4TQRKKWUh9NEoNIkR7XQ4AS30snsezkFvm+6iPzj+K4/RaTuXXzGVBHxdzwemmjbunuN0fE5sX8uOxwVN3PfZv8gEWmTEt+tMi4dPqrSJBG5bIzJkdL7JvMZ04EFxpg5ItICGGuMCbiHz7vnmG73uSLyFbDPGPN2Mvv3AmoaYwaldCwq49ArApUuiEgOEVnmOFvfLiI3VRoVkSIisjrBGXNDx+stRGS9472zReR2B+jVQDnHe190fNYOEXne8Vp2EfnVUf9+h4g87nh9pYjUFJH3AF9HHDMc2y477mclPEN3XIl0FhEvEflARDaLrTH/rBN/LOtxFBsTkQfErjPxl+O+omMm7pvA445YHnfE/qXje/5K6s9ReSB3197Wm96SugHR2EJiwcDP2FnwuRzb8mNnVcZe0V523L8EvO547AXkdOy7GsjueH0wMDyJ75uOY70C4FFgI7Z423YgO7a88U6gOtAZ+DzBe/0c9yuxZ99xMSXYJzbGR4CvHI+zYKtI+gJ9gTccr2cFtgBlkojzcoLfNxto5XieC/B2PG4G/Oh43AuYkOD97wBPOR7nxtYgyu7uv2+9ufeW4UpMqAzjqjEmKPaJiGQG3hGRB7GlE4oBhYCTCd6zGfjSse9cY0ywiDQC/IG1jtIaWbBn0kn5QETeAE5jK7Q2BX42toAbIvIT0BBYBIwVkfexzUlr7uB3/QaME5GsQCtgtTHmqqM5KkDiV1HzA8oD/yR6v6+IBAOlga3A7wn2/0pEymMrUWa+xfe3AB4WkZcdz32AkqTvekTqHmkiUOlFN+zqUzWMMZEicgh7EItjjFntSBRtgW9E5APgPPC7MeZJJ77jFWPMnNgnItIsqZ2MMftEpAa23su7IrLEGPOmMz/CGBMhIiuxpZMfB76P/TrgP8aYxbf5iKvGmCAR8QMWAAOBcdh6OyuMMY84OtZX3uL9AnQ2xux1Jl7lGbSPQKUXfsApRxJoApRKvIOIlHLs8znwBXa5vw1AfRGJbfPPJiIVnPzO1UBHx3uyY5t11ohIUSDcGPMtMNbxPYlFOq5MkjITWyisIbaYGo77/rHvEZEKju9MkjEmDHgOeNnxHj/gmGNzrwS7XsI2kcVaDPxHHJdHIlL9Vt+hPIcmApVezABqisgW7NXBniT2aQwEi8hf2Hb8T4wxp7EHxu9FJASbGCo584XGmD+xfQebsH0GU40xfwHVgE2OJprXgdFJvH0KEBLbWZzIEuy6tEuNXX4R7DoRu4A/xS5a/hm3uWJ3xLINW5p5DPbqZC22/yDWCsA/trMYe+WQ2RHbDsdz5eF0+KhSSnk4vSJQSikPp4lAKaU8nCYCpZTycJoIlFLKw2kiUEopD6eJQCmlPJwmAqWU8nD/D9fLGPtMkBGxAAAAAElFTkSuQmCC"/>

#### -> 의사결정나부모다 랜덤 포레스트가 보다 모델 성능이 좋다



```python
# 두 모델의 AUC 계산

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

<img src="data:image/png;base64,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U0XEX0RSLHfEeUvvRVrvLF5wcAHe07zxv+zPzDYzWd19NaVzlXZ0WE6hYcP4j6lTddm9ewmX+/np8mvX4pc9ivz58zN9+nQmT56MUoqIiAjeffddatasibe3N9/HWH913LhxeHl54ePjw7BhwwDo1asXCxcuBGDYsGF4enri7e3NO++8A8CoUaMYP348AP7+/tSuXRtvb29eeOEFblhm5zds2JChQ4fy1FNPUb58eTZv3mzj360hH3zwAQ0aNODbb79l9+7dNGjQgOrVq9O8eXMuXboEwMmTJ2nRogXVq1enfv36HDlyBIB169ZRrVo1XF11y/YPP/xAzZo18fHxoUOHDtyz3Ob36tWLIUOG0KhRI4YOHfrQ4y1dupRatWpRtWpVmjRpwn///WfT79ixYwdly5aldOnSZMqUiS5duvDnn38m+plff/2Vl156CYBbt26xadMm+vTpA0CmTJnImTMnACVKlCAoKIjLly/bFMuTsus8AqXUcmB5nG3TYrzuC/S1ZwwPExgIrq5pt7NYWZLEVS1UlbYV2vJ1868p7FHY0WEZKah06dJERkZy5coV/vzzT3LkyMHOnTt58OAB9erVo1mzZhw5coTFixezfft2smTJYm2SiHL9+nUWLVrEkSNHEBFu3rwZ73t69OjBpEmTaNCgASNGjGD06NF88803AISHh7Njxw6WL1/O6NGjE21uiunmzZts3LiRsLAwGjRowJ9//km+fPmYP38+H374IT/++CP9+vVj2rRplCtXju3btzNw4EDWrVvHli1bqF69uvVY7du353//+x8AH330ETNnzuS1114D4NixY6xZswYXFxcaN26c4PGefvpptm3bhogwY8YMxo0bx1dffcX69et566234sWeJUsWtm7dyoULFygWI4d90aJF2b59+0N/871791ixYgWTJ08G4NSpU+TLl4/evXuzb98+qlevzrfffkvWrFkBqFatGlu2bKFDhw42/U2fhNPOLD5/Pm1OJnsQ/oBPN3/K4WuH+a3jb5TNXZZ5Hec5OiyntGHDw8uyZEm8PG/exMttpSxJI1etWkVAQID1Kj84OJjjx4+zZs0aevfuTZYsWQDInTt3rM9nz54dNzc3+vbtS6tWrWjdunWs8uDgYG7evEmDBg0A6NmzJ506dbKWt2/fHoDq1atz5swZm+Pu3LkzAEePHuXAgQM0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yCuvvEKlSpXo1auXzXFfv36ddu3a4e3tTe3atQmwXJGNGjWKfv360axZM3r06MHVq1fp0KEDNWvWpGbNmtYYN27caL3DqFq1Krdv32bYsGFs3rwZX19fJkyYwO3btwkICMDHxyfJv0+nTp14/vnnadasGXfv3uWVV16hZs2aVK1a1ZpS+syZM9SvX59q1apRrVo1a4qJpCilWLdunbUC7NmzJ4sXL46336FDh2jQoAGurq5kzZoVHx8fVqxYAejUGFHHun//vvXCUURo2LAhy5Yts/lvn2SwaenxpAvTHF54QIFSCxc+3mEexZZzW1TFyRUVo1A9F/VUIWEh9v9Sw27iLfbRoEH8x5Qpuuzu3YTLZ83S5Vevxi9LQtTCNOHh4apjx47q77//VkopFRwcbF14ZfXq1ap9+/ZKKaVmzZqlSpUqpW7evKnu37+vihcvrs6dO6cuXryoihUrpq5cuaIePHig6tatqwYNGqSUUqp169bKz89PKaXUzJkzVdu2bZVSemGZzp07q8jISLV48WLl4eGhAgICVEREhKpWrZrau3dvvHgbNGigypcvr3x8fJSPj4+6du2aGjx4sBo1apRSSqm1a9cqHx8fpZRSI0eOVNWqVVP37t1TSin10ksvqc2bNyullDp79qyqWLGiNb5//vlHKaXU7du3VVhYmFq/fr1q1aqV9XvXrVtn/Rsk9fcpUqSICgoKUkop9f7776s5c+YopZS6ceOGKleunLpz5466e/euun//vlJKqWPHjqmoc9CtW7esvy3u4+DBg+rq1auqTJky1jjOnTunKleuHO/vtHLlSlW3bl119+5ddfXqVVWqVCk1fvx4a3mvXr1U/vz5VcOGDdXdu3et23/++Wc1ePDgeMdT6tEXpnGezmKLqHUe7DmH4E7oHT5Y+wGTd0ymWI5irOi2guZlmyf9QSNtSeE81FG5hs6cOUP16tWtqZuDg4Pp2bMnx48fR0SsiecAGjduTA7LWt2enp6cPXuWa9eu0bBhQ/LlywfolNDHjh0D4N9//+WPP/4AoHv37rHuIp5//nlEBC8vLwoUKICXlxcAlStX5syZM/j6+saLee7cubHy5fzzzz/8/vvvADz77LMEBQURHBwMQJs2bazZNtesWcOhQ4esn7t16xa3b9+mXr16DBkyhG7dutG+fXuKJjAr9NKlS9bfltTfp2nT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"/>

#### -> 랜덤 포레스트로 했을 시 의사결정나무보다 17% 정도의 성능 향상



```python
TP=110/200
TP
```

<pre>
0.55
</pre>

```python
a=0.57*0.4
```


```python
b=0.57+0.4
```


```python
2*(a/b)
```

<pre>
0.47010309278350515
</pre>