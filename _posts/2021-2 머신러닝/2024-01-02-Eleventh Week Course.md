---
layout: single
title:  "Eleventh Week Course"
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


# 11주차 강의내용



```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
```

<pre>
지정된 경로를 찾을 수 없습니다.
</pre>

```python
df=pd.DataFrame(columns=('x','y'))
```


```python
# 데이터 준비: 임의의 2차원 좌표
df.loc[0]=[1,1]
df.loc[1]=[1,0]
df.loc[2]=[2,1]
df.loc[3]=[5,3]
df.loc[4]=[6,7]
df.loc[5]=[6,6]
df.loc[6]=[8,0]
df.loc[7]=[11,2]
df.loc[8]=[12,1]
df.loc[9]=[2,2]
df.loc[10]=[12,3]
df.loc[11]=[7,7]
df.loc[12]=[7,6]
df.loc[13]=[13,4]
df.loc[14]=[13,1]
df.loc[15]=[6,5]
df.loc[16]=[4,5]
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 산점도 그리기
sns.lmplot('x','y', data=df, fit_reg=False, scatter_kws={"s":200})
# fit_reg=False 회귀직선 없이 산점도만 그림
# scatter_kws 는 size를 조절(s는 점의 사이즈 조절)

plt.title('k-means plot')
plt.xlabel('x')
plt.ylabel('y')
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
</pre>
<pre>
Text(16.424999999999997, 0.5, 'y')
</pre>
<img src="data:image/png;base64,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"/>

### 적절한 클러스터 갯수 구하기 : 엘보우 메소드

- 오차제곱합(SSE)값이 inertia_에 저장되며 이 값을 이용하여 그래프 작성



```python
X=df.values # 인덱스를 제외한 값을 ndarray로 추출
inertia_arr=[] # SSE값을 저장하기 위한 list
K=range(1,10) # 전체 데이터 개수가 17개였으니 그렇게 군집을 많이 만들 필요는 없고 10개까지만
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X) # 모델 접합
    inertia = kmeanModel.inertia_
    inertia_arr.append(inertia)
    
# Plot the elbow
plt.plot(K, inertia_arr, 'bx-')
plt.xlabel(('K'))        # 클러스터 개수
plt.ylabel('Distortion') # 클러스터 내 오차제곱합(SSE)
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py:881: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
  warnings.warn(
</pre>
<img src="data:image/png;base64,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"/>

### 대략 K 3개 정도가 적절한 개수라고 할 수 있다. 3개일 때 확 꺾이는 지점-> elbow point이므로



```python
# 최적 K 개수로 클러스터링 수행
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
```

<pre>
KMeans(n_clusters=3)
</pre>

```python
kmeans.labels_ # 클러스터 결과 확인
```

<pre>
array([0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2])
</pre>

```python
df['cluster_id']=kmeans.labels_
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
      <th>x</th>
      <th>y</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>5</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 클러스터링 시각화
sns.lmplot('x','y', data=df, fit_reg=False, scatter_kws={"s":150}, hue='cluster_id')
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
</pre>
<pre>
<seaborn.axisgrid.FacetGrid at 0x2153c6fad30>
</pre>
<img src="data:image/png;base64,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"/>

## Scaling



```python
import matplotlib.pyplot as plt

df = pd.read_csv("./data/player_salary.csv")
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
      <th>선수</th>
      <th>연봉</th>
      <th>활동기간</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>김연경</td>
      <td>14</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>박태환</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박지성</td>
      <td>48</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>손흥민</td>
      <td>65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차두리</td>
      <td>8</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>황희찬</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>류현진</td>
      <td>14</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>김민재</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>이윤열</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>김병현</td>
      <td>6</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>기성용</td>
      <td>20</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



```python
x = df['활동기간']
y = df['연봉']

plt.scatter(x,y)
plt.axis('scaled') # x, y축 눈금 간격을 동일하게 이걸 생략하면 기본값으로 보여질거다
plt.title('k-mean plot')
plt.xlabel('x')
plt.ylabel('y')
```

<pre>
Text(0, 0.5, 'y')
</pre>
<img src="data:image/png;base64,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"/>

### -> 활동기간은 1-20까지의 범위고 연봉은 1-65(억)까지 존재. 그래서 범위가 차이가 많이 나지

### -> 그래서 y값에 비하면 x값의 범위는 미미하게 보일 것이다 

### -> 이럴 때 변수 간의 스케일을 조정해 주는 것이 Scale



```python
# 스케일링 작업
from sklearn.preprocessing import StandardScaler

df_scaled = StandardScaler().fit_transform(df.drop(['선수'], axis=1)) # 선수 칼럼을 제거하는 건 수치형 변수만 남겨놓기 위함
df_scaled
```

<pre>
array([[-0.16222491,  1.80363227],
       [-0.67207461, -0.23123491],
       [ 1.57126409,  1.80363227],
       [ 2.43800859, -0.7399517 ],
       [-0.46813473,  0.27748189],
       [-0.62108964, -1.07909623],
       [-0.16222491, -0.7399517 ],
       [-0.67207461, -1.24866849],
       [-0.82502952, -0.40080717],
       [-0.57010467, -0.06166264],
       [ 0.14368492,  0.61662642]])
</pre>

```python
df2 = pd.DataFrame(df_scaled, columns = ('연봉', '활동기간'))

x = df2['활동기간']
y = df2['연봉']

plt.scatter(x,y)
plt.axis('scaled')
plt.title('k-mean plot')
plt.xlabel('x')
plt.ylabel('y')
```

<pre>
(-1.4012835303104763,
 1.9562473046908628,
 -0.9881814281889821,
 2.6011604948389153)
</pre>
<img src="data:image/png;base64,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"/>

### -> 위에서 봤던 산점도에 비해서는 활동기간에 대한 값이 반영되는 것을 볼 수 있음.


### 변수가 여러 개인 데이터 클러스터링



```python
df = pd.read_csv("./data/player.csv")
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
      <th>선수</th>
      <th>외식</th>
      <th>영화관람</th>
      <th>공연관람</th>
      <th>쇼핑횟수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>김연경</td>
      <td>97</td>
      <td>13</td>
      <td>14</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>박태환</td>
      <td>88</td>
      <td>6</td>
      <td>9</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박지성</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>손흥민</td>
      <td>135</td>
      <td>9</td>
      <td>15</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차두리</td>
      <td>18</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>황희찬</td>
      <td>81</td>
      <td>11</td>
      <td>8</td>
      <td>110</td>
    </tr>
    <tr>
      <th>6</th>
      <td>류현진</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>김민재</td>
      <td>11</td>
      <td>8</td>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>이윤열</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>김병현</td>
      <td>19</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>기성용</td>
      <td>38</td>
      <td>9</td>
      <td>2</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>


### -> 외식이 단위 큰 경우가 많으므로 스케일링을 하지 않으면 외식의 의사성에 따라서 클러스터링이 크게 좌우가 되고 다른 칼럼은 상대적으로 반영되지 않을 수 있음



```python
# StandardScaler로 스케일링 수행 스케일링을 함으로 칼럼들 간의 값의 격차가 줄어듦
from sklearn.preprocessing import StandardScaler

# 선수 칼럼은 수치형 변수가 아니므로 드롭
df_scaled = StandardScaler().fit_transform(df.drop(['선수'], axis=1))
df_scaled
```

<pre>
array([[ 1.16449073,  1.4888528 ,  1.66804677,  0.        ],
       [ 0.96080773, -0.19705405,  0.67084489, -0.10085944],
       [-0.91760224, -1.16042938, -1.12411847, -0.67239626],
       [ 2.02448566,  0.52547746,  1.86748714, -0.30257832],
       [-0.62339345, -1.64211705, -0.52579735, -0.33619813],
       [ 0.80238761,  1.00716513,  0.47140452,  3.02578316],
       [-0.82707646, -1.64211705,  0.07252377, -0.43705757],
       [-0.78181357,  0.28463362, -0.9246781 ,  0.33619813],
       [-1.03075947,  0.28463362, -0.9246781 , -0.67239626],
       [-0.600762  ,  0.52547746, -0.52579735, -0.60515663],
       [-0.17076454,  0.52547746, -0.72523772, -0.23533869]])
</pre>

```python
# 오차제곱합으로 최적 클러스터 갯수 찾기
X = df_scaled

inertia_arr = [] # SSE 값을 저장하기 위한 list
K = range(1,10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X) # 모델 적합
    inertia = kmeanModel.inertia_
    inertia_arr.append(inertia)
    
# Plot the elbow
plt.plot(K, inertia_arr, 'bx-')
plt.xlabel(('K'))        # 클러스터 개수
plt.ylabel('Distortion') # 클러스터 내 오차제곱합(SSE)
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py:881: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
  warnings.warn(
</pre>
<img src="data:image/png;base64,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"/>

### -> 3 정도가 최적의 개수



```python
# 최적 K 개수 3으로 클러스터링 수행
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
```

<pre>
KMeans(n_clusters=3)
</pre>

```python
kmeans.labels_
```

<pre>
array([2, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1])
</pre>

```python
# 원데이터에 클러스터링 결과 추가
df['cluster_id'] = kmeans.labels_
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
      <th>선수</th>
      <th>외식</th>
      <th>영화관람</th>
      <th>공연관람</th>
      <th>쇼핑횟수</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>김연경</td>
      <td>97</td>
      <td>13</td>
      <td>14</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>박태환</td>
      <td>88</td>
      <td>6</td>
      <td>9</td>
      <td>17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박지성</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>손흥민</td>
      <td>135</td>
      <td>9</td>
      <td>15</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차두리</td>
      <td>18</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>황희찬</td>
      <td>81</td>
      <td>11</td>
      <td>8</td>
      <td>110</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>류현진</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>김민재</td>
      <td>11</td>
      <td>8</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>이윤열</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>김병현</td>
      <td>19</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>기성용</td>
      <td>38</td>
      <td>9</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
# cluster_id 를 기준으로 오름차순으로 정렬
df = df.sort_values(by='cluster_id')
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
      <th>선수</th>
      <th>외식</th>
      <th>영화관람</th>
      <th>공연관람</th>
      <th>쇼핑횟수</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>황희찬</td>
      <td>81</td>
      <td>11</td>
      <td>8</td>
      <td>110</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박지성</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차두리</td>
      <td>18</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>류현진</td>
      <td>9</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>김민재</td>
      <td>11</td>
      <td>8</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>이윤열</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>김병현</td>
      <td>19</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>기성용</td>
      <td>38</td>
      <td>9</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>0</th>
      <td>김연경</td>
      <td>97</td>
      <td>13</td>
      <td>14</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>박태환</td>
      <td>88</td>
      <td>6</td>
      <td>9</td>
      <td>17</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>손흥민</td>
      <td>135</td>
      <td>9</td>
      <td>15</td>
      <td>11</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


### ->  변수가 여러 개 이므로 2차원 그래프에 시각화를 시킬 순 없다 이때 사용되는 게 차원축소 개념


# 차원 축소 - PCA(주성분 분석)



```python
# 스케일링
df = pd.read_csv("./data/player.csv")
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
df['cluster_id'] = kmeans.labels_

df_scaled = StandardScaler().fit_transform(df.drop(['선수'], axis=1))
```


```python
# 4차원 데이터임. 변수 4개
df_scaled
```

<pre>
array([[ 1.16449073,  1.4888528 ,  1.66804677,  0.        ,  0.83205029],
       [ 0.96080773, -0.19705405,  0.67084489, -0.10085944,  0.83205029],
       [-0.91760224, -1.16042938, -1.12411847, -0.67239626, -0.69337525],
       [ 2.02448566,  0.52547746,  1.86748714, -0.30257832,  0.83205029],
       [-0.62339345, -1.64211705, -0.52579735, -0.33619813, -0.69337525],
       [ 0.80238761,  1.00716513,  0.47140452,  3.02578316,  2.35747583],
       [-0.82707646, -1.64211705,  0.07252377, -0.43705757, -0.69337525],
       [-0.78181357,  0.28463362, -0.9246781 ,  0.33619813, -0.69337525],
       [-1.03075947,  0.28463362, -0.9246781 , -0.67239626, -0.69337525],
       [-0.600762  ,  0.52547746, -0.52579735, -0.60515663, -0.69337525],
       [-0.17076454,  0.52547746, -0.72523772, -0.23533869, -0.69337525]])
</pre>

```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) # 차원 갯수 설정(그래프 상에 나타내기 위해서 2개로 설정)
pca.fit(df_scaled)

df_pca = pca.transform(df_scaled)

print("축소전:", df_scaled.shape)
print("축소후:", df_pca.shape)
```

<pre>
축소전: (11, 5)
축소후: (11, 2)
</pre>

```python
# 4개의 변수 중 두 칼럼을 가져온 게 아니라 4개의 변수를 두개의 변수로 만든 것
df_pca
```

<pre>
array([[ 2.34385127, -0.92537628],
       [ 1.10856663, -0.62044625],
       [-2.01398871,  0.12647038],
       [ 2.3919495 , -1.68179115],
       [-1.65442026, -0.08828316],
       [ 3.2995882 ,  2.32059434],
       [-1.51683495, -0.38804455],
       [-0.94700923,  0.89533944],
       [-1.43114907,  0.23619506],
       [-0.91771171, -0.04537273],
       [-0.66284165,  0.1707149 ]])
</pre>

```python
cols = ["pc1", "pc2"]
df_pca = pd.DataFrame(data=df_pca, columns=cols)
df_pca
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
      <th>pc1</th>
      <th>pc2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.343851</td>
      <td>-0.925376</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.108567</td>
      <td>-0.620446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.013989</td>
      <td>0.126470</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.391949</td>
      <td>-1.681791</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.654420</td>
      <td>-0.088283</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.299588</td>
      <td>2.320594</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.516835</td>
      <td>-0.388045</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.947009</td>
      <td>0.895339</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.431149</td>
      <td>0.236195</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.917712</td>
      <td>-0.045373</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.662842</td>
      <td>0.170715</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 새로 만든 데이터에 선수 칼럼과 클러스터 번호 칼럼 추가
df_pca["name"] = df.선수
df_pca["target"] = df.cluster_id
```


```python
df_pca
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
      <th>pc1</th>
      <th>pc2</th>
      <th>name</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.343851</td>
      <td>-0.925376</td>
      <td>김연경</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.108567</td>
      <td>-0.620446</td>
      <td>박태환</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.013989</td>
      <td>0.126470</td>
      <td>박지성</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.391949</td>
      <td>-1.681791</td>
      <td>손흥민</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.654420</td>
      <td>-0.088283</td>
      <td>차두리</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.299588</td>
      <td>2.320594</td>
      <td>황희찬</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.516835</td>
      <td>-0.388045</td>
      <td>류현진</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.947009</td>
      <td>0.895339</td>
      <td>김민재</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.431149</td>
      <td>0.236195</td>
      <td>이윤열</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.917712</td>
      <td>-0.045373</td>
      <td>김병현</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.662842</td>
      <td>0.170715</td>
      <td>기성용</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### 시각화



```python
# 차원 축소를 통한 클러스터링 결과
# 세 가지 종류의 marker로 그래프 상에 좌표 표시

marker = ["^", "s", "o"]
for i, marker in enumerate(marker):
    x_val = df_pca[df_pca["target"]==i]["pc1"]
    y_val = df_pca[df_pca["target"]==i]["pc2"]
    plt.scatter(x_val, y_val, marker=marker)

# 한글 폰트 가져오기
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 선수 이름 표시
for i in range(11):
    plt.text(df_pca['pc1'][i]+0.05, df_pca['pc2'][i]+0.05, df_pca['name'][i])

plt.xlabel("pc1")
plt.ylabel("pc2")
```

<pre>
Text(0, 0.5, 'pc2')
</pre>
<img src="data:image/png;base64,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"/>


```python
# Euclidean Distance
import math
a=4
b=1
c=5
d=1
ED= math.sqrt((a-b)**2 +(c-d)**2)
ED
```

<pre>
5.0
</pre>

```python
# Manhattan Distance
a=4
b=1
c=5
d=1
MD= abs(a-b)+abs(c-d)
MD
```

<pre>
7
</pre>

```python
```
