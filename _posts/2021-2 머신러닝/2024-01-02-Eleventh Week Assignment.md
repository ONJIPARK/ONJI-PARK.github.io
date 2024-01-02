---
layout: single
title:  "Eleventh Week Assignment"
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


# 11주차 과제(단톡방)

- 60191315 박온지

- 아이리스 데이터를 활용하여 군집화해보시오.

- 별도로 다운받아도 되고 Seaborn 데이터셋에서 불러와도 됨

- 지도학습용 데이터에서 y변수를 생략해서 비지도학습용으로 사용할 수있음

- 적절한 군집의 개수를 정할 것(elbow point확인할 것)

- 스케일링을 수행할 것

- 차원축소를 통하여 클러스터링 결과를 그래프 상에 나타낼 것

- 실제 세 가지 품종과 클러스터링 결과가 얼마나 잘 맞는지 맞춘 비율을 계산하시오.



```python
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```


```python
# seaborn 패키지 제공 데이터
sns.get_dataset_names()
```

<pre>
['anagrams',
 'anscombe',
 'attention',
 'brain_networks',
 'car_crashes',
 'diamonds',
 'dots',
 'exercise',
 'flights',
 'fmri',
 'gammas',
 'geyser',
 'iris',
 'mpg',
 'penguins',
 'planets',
 'taxis',
 'tips',
 'titanic']
</pre>

```python
# 데이터 불러오기
df = sns.load_dataset('iris')
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>


### 스케일링



```python
# StandardScaler로 스케일링 수행 스키일링을 함으로 칼럼들 간의 값의 격차가 줄어듦
from sklearn.preprocessing import StandardScaler

# species 칼럼은 수치형 변수가 아니므로 드롭
df_scaled = StandardScaler().fit_transform(df.drop(['species'], axis=1))
df_scaled
```

<pre>
array([[-9.00681170e-01,  1.01900435e+00, -1.34022653e+00,
        -1.31544430e+00],
       [-1.14301691e+00, -1.31979479e-01, -1.34022653e+00,
        -1.31544430e+00],
       [-1.38535265e+00,  3.28414053e-01, -1.39706395e+00,
        -1.31544430e+00],
       [-1.50652052e+00,  9.82172869e-02, -1.28338910e+00,
        -1.31544430e+00],
       [-1.02184904e+00,  1.24920112e+00, -1.34022653e+00,
        -1.31544430e+00],
       [-5.37177559e-01,  1.93979142e+00, -1.16971425e+00,
        -1.05217993e+00],
       [-1.50652052e+00,  7.88807586e-01, -1.34022653e+00,
        -1.18381211e+00],
       [-1.02184904e+00,  7.88807586e-01, -1.28338910e+00,
        -1.31544430e+00],
       [-1.74885626e+00, -3.62176246e-01, -1.34022653e+00,
        -1.31544430e+00],
       [-1.14301691e+00,  9.82172869e-02, -1.28338910e+00,
        -1.44707648e+00],
       [-5.37177559e-01,  1.47939788e+00, -1.28338910e+00,
        -1.31544430e+00],
       [-1.26418478e+00,  7.88807586e-01, -1.22655167e+00,
        -1.31544430e+00],
       [-1.26418478e+00, -1.31979479e-01, -1.34022653e+00,
        -1.44707648e+00],
       [-1.87002413e+00, -1.31979479e-01, -1.51073881e+00,
        -1.44707648e+00],
       [-5.25060772e-02,  2.16998818e+00, -1.45390138e+00,
        -1.31544430e+00],
       [-1.73673948e-01,  3.09077525e+00, -1.28338910e+00,
        -1.05217993e+00],
       [-5.37177559e-01,  1.93979142e+00, -1.39706395e+00,
        -1.05217993e+00],
       [-9.00681170e-01,  1.01900435e+00, -1.34022653e+00,
        -1.18381211e+00],
       [-1.73673948e-01,  1.70959465e+00, -1.16971425e+00,
        -1.18381211e+00],
       [-9.00681170e-01,  1.70959465e+00, -1.28338910e+00,
        -1.18381211e+00],
       [-5.37177559e-01,  7.88807586e-01, -1.16971425e+00,
        -1.31544430e+00],
       [-9.00681170e-01,  1.47939788e+00, -1.28338910e+00,
        -1.05217993e+00],
       [-1.50652052e+00,  1.24920112e+00, -1.56757623e+00,
        -1.31544430e+00],
       [-9.00681170e-01,  5.58610819e-01, -1.16971425e+00,
        -9.20547742e-01],
       [-1.26418478e+00,  7.88807586e-01, -1.05603939e+00,
        -1.31544430e+00],
       [-1.02184904e+00, -1.31979479e-01, -1.22655167e+00,
        -1.31544430e+00],
       [-1.02184904e+00,  7.88807586e-01, -1.22655167e+00,
        -1.05217993e+00],
       [-7.79513300e-01,  1.01900435e+00, -1.28338910e+00,
        -1.31544430e+00],
       [-7.79513300e-01,  7.88807586e-01, -1.34022653e+00,
        -1.31544430e+00],
       [-1.38535265e+00,  3.28414053e-01, -1.22655167e+00,
        -1.31544430e+00],
       [-1.26418478e+00,  9.82172869e-02, -1.22655167e+00,
        -1.31544430e+00],
       [-5.37177559e-01,  7.88807586e-01, -1.28338910e+00,
        -1.05217993e+00],
       [-7.79513300e-01,  2.40018495e+00, -1.28338910e+00,
        -1.44707648e+00],
       [-4.16009689e-01,  2.63038172e+00, -1.34022653e+00,
        -1.31544430e+00],
       [-1.14301691e+00,  9.82172869e-02, -1.28338910e+00,
        -1.31544430e+00],
       [-1.02184904e+00,  3.28414053e-01, -1.45390138e+00,
        -1.31544430e+00],
       [-4.16009689e-01,  1.01900435e+00, -1.39706395e+00,
        -1.31544430e+00],
       [-1.14301691e+00,  1.24920112e+00, -1.34022653e+00,
        -1.44707648e+00],
       [-1.74885626e+00, -1.31979479e-01, -1.39706395e+00,
        -1.31544430e+00],
       [-9.00681170e-01,  7.88807586e-01, -1.28338910e+00,
        -1.31544430e+00],
       [-1.02184904e+00,  1.01900435e+00, -1.39706395e+00,
        -1.18381211e+00],
       [-1.62768839e+00, -1.74335684e+00, -1.39706395e+00,
        -1.18381211e+00],
       [-1.74885626e+00,  3.28414053e-01, -1.39706395e+00,
        -1.31544430e+00],
       [-1.02184904e+00,  1.01900435e+00, -1.22655167e+00,
        -7.88915558e-01],
       [-9.00681170e-01,  1.70959465e+00, -1.05603939e+00,
        -1.05217993e+00],
       [-1.26418478e+00, -1.31979479e-01, -1.34022653e+00,
        -1.18381211e+00],
       [-9.00681170e-01,  1.70959465e+00, -1.22655167e+00,
        -1.31544430e+00],
       [-1.50652052e+00,  3.28414053e-01, -1.34022653e+00,
        -1.31544430e+00],
       [-6.58345429e-01,  1.47939788e+00, -1.28338910e+00,
        -1.31544430e+00],
       [-1.02184904e+00,  5.58610819e-01, -1.34022653e+00,
        -1.31544430e+00],
       [ 1.40150837e+00,  3.28414053e-01,  5.35408562e-01,
         2.64141916e-01],
       [ 6.74501145e-01,  3.28414053e-01,  4.21733708e-01,
         3.95774101e-01],
       [ 1.28034050e+00,  9.82172869e-02,  6.49083415e-01,
         3.95774101e-01],
       [-4.16009689e-01, -1.74335684e+00,  1.37546573e-01,
         1.32509732e-01],
       [ 7.95669016e-01, -5.92373012e-01,  4.78571135e-01,
         3.95774101e-01],
       [-1.73673948e-01, -5.92373012e-01,  4.21733708e-01,
         1.32509732e-01],
       [ 5.53333275e-01,  5.58610819e-01,  5.35408562e-01,
         5.27406285e-01],
       [-1.14301691e+00, -1.51316008e+00, -2.60315415e-01,
        -2.62386821e-01],
       [ 9.16836886e-01, -3.62176246e-01,  4.78571135e-01,
         1.32509732e-01],
       [-7.79513300e-01, -8.22569778e-01,  8.07091462e-02,
         2.64141916e-01],
       [-1.02184904e+00, -2.43394714e+00, -1.46640561e-01,
        -2.62386821e-01],
       [ 6.86617933e-02, -1.31979479e-01,  2.51221427e-01,
         3.95774101e-01],
       [ 1.89829664e-01, -1.97355361e+00,  1.37546573e-01,
        -2.62386821e-01],
       [ 3.10997534e-01, -3.62176246e-01,  5.35408562e-01,
         2.64141916e-01],
       [-2.94841818e-01, -3.62176246e-01, -8.98031345e-02,
         1.32509732e-01],
       [ 1.03800476e+00,  9.82172869e-02,  3.64896281e-01,
         2.64141916e-01],
       [-2.94841818e-01, -1.31979479e-01,  4.21733708e-01,
         3.95774101e-01],
       [-5.25060772e-02, -8.22569778e-01,  1.94384000e-01,
        -2.62386821e-01],
       [ 4.32165405e-01, -1.97355361e+00,  4.21733708e-01,
         3.95774101e-01],
       [-2.94841818e-01, -1.28296331e+00,  8.07091462e-02,
        -1.30754636e-01],
       [ 6.86617933e-02,  3.28414053e-01,  5.92245988e-01,
         7.90670654e-01],
       [ 3.10997534e-01, -5.92373012e-01,  1.37546573e-01,
         1.32509732e-01],
       [ 5.53333275e-01, -1.28296331e+00,  6.49083415e-01,
         3.95774101e-01],
       [ 3.10997534e-01, -5.92373012e-01,  5.35408562e-01,
         8.77547895e-04],
       [ 6.74501145e-01, -3.62176246e-01,  3.08058854e-01,
         1.32509732e-01],
       [ 9.16836886e-01, -1.31979479e-01,  3.64896281e-01,
         2.64141916e-01],
       [ 1.15917263e+00, -5.92373012e-01,  5.92245988e-01,
         2.64141916e-01],
       [ 1.03800476e+00, -1.31979479e-01,  7.05920842e-01,
         6.59038469e-01],
       [ 1.89829664e-01, -3.62176246e-01,  4.21733708e-01,
         3.95774101e-01],
       [-1.73673948e-01, -1.05276654e+00, -1.46640561e-01,
        -2.62386821e-01],
       [-4.16009689e-01, -1.51316008e+00,  2.38717193e-02,
        -1.30754636e-01],
       [-4.16009689e-01, -1.51316008e+00, -3.29657076e-02,
        -2.62386821e-01],
       [-5.25060772e-02, -8.22569778e-01,  8.07091462e-02,
         8.77547895e-04],
       [ 1.89829664e-01, -8.22569778e-01,  7.62758269e-01,
         5.27406285e-01],
       [-5.37177559e-01, -1.31979479e-01,  4.21733708e-01,
         3.95774101e-01],
       [ 1.89829664e-01,  7.88807586e-01,  4.21733708e-01,
         5.27406285e-01],
       [ 1.03800476e+00,  9.82172869e-02,  5.35408562e-01,
         3.95774101e-01],
       [ 5.53333275e-01, -1.74335684e+00,  3.64896281e-01,
         1.32509732e-01],
       [-2.94841818e-01, -1.31979479e-01,  1.94384000e-01,
         1.32509732e-01],
       [-4.16009689e-01, -1.28296331e+00,  1.37546573e-01,
         1.32509732e-01],
       [-4.16009689e-01, -1.05276654e+00,  3.64896281e-01,
         8.77547895e-04],
       [ 3.10997534e-01, -1.31979479e-01,  4.78571135e-01,
         2.64141916e-01],
       [-5.25060772e-02, -1.05276654e+00,  1.37546573e-01,
         8.77547895e-04],
       [-1.02184904e+00, -1.74335684e+00, -2.60315415e-01,
        -2.62386821e-01],
       [-2.94841818e-01, -8.22569778e-01,  2.51221427e-01,
         1.32509732e-01],
       [-1.73673948e-01, -1.31979479e-01,  2.51221427e-01,
         8.77547895e-04],
       [-1.73673948e-01, -3.62176246e-01,  2.51221427e-01,
         1.32509732e-01],
       [ 4.32165405e-01, -3.62176246e-01,  3.08058854e-01,
         1.32509732e-01],
       [-9.00681170e-01, -1.28296331e+00, -4.30827696e-01,
        -1.30754636e-01],
       [-1.73673948e-01, -5.92373012e-01,  1.94384000e-01,
         1.32509732e-01],
       [ 5.53333275e-01,  5.58610819e-01,  1.27429511e+00,
         1.71209594e+00],
       [-5.25060772e-02, -8.22569778e-01,  7.62758269e-01,
         9.22302838e-01],
       [ 1.52267624e+00, -1.31979479e-01,  1.21745768e+00,
         1.18556721e+00],
       [ 5.53333275e-01, -3.62176246e-01,  1.04694540e+00,
         7.90670654e-01],
       [ 7.95669016e-01, -1.31979479e-01,  1.16062026e+00,
         1.31719939e+00],
       [ 2.12851559e+00, -1.31979479e-01,  1.61531967e+00,
         1.18556721e+00],
       [-1.14301691e+00, -1.28296331e+00,  4.21733708e-01,
         6.59038469e-01],
       [ 1.76501198e+00, -3.62176246e-01,  1.44480739e+00,
         7.90670654e-01],
       [ 1.03800476e+00, -1.28296331e+00,  1.16062026e+00,
         7.90670654e-01],
       [ 1.64384411e+00,  1.24920112e+00,  1.33113254e+00,
         1.71209594e+00],
       [ 7.95669016e-01,  3.28414053e-01,  7.62758269e-01,
         1.05393502e+00],
       [ 6.74501145e-01, -8.22569778e-01,  8.76433123e-01,
         9.22302838e-01],
       [ 1.15917263e+00, -1.31979479e-01,  9.90107977e-01,
         1.18556721e+00],
       [-1.73673948e-01, -1.28296331e+00,  7.05920842e-01,
         1.05393502e+00],
       [-5.25060772e-02, -5.92373012e-01,  7.62758269e-01,
         1.58046376e+00],
       [ 6.74501145e-01,  3.28414053e-01,  8.76433123e-01,
         1.44883158e+00],
       [ 7.95669016e-01, -1.31979479e-01,  9.90107977e-01,
         7.90670654e-01],
       [ 2.24968346e+00,  1.70959465e+00,  1.67215710e+00,
         1.31719939e+00],
       [ 2.24968346e+00, -1.05276654e+00,  1.78583195e+00,
         1.44883158e+00],
       [ 1.89829664e-01, -1.97355361e+00,  7.05920842e-01,
         3.95774101e-01],
       [ 1.28034050e+00,  3.28414053e-01,  1.10378283e+00,
         1.44883158e+00],
       [-2.94841818e-01, -5.92373012e-01,  6.49083415e-01,
         1.05393502e+00],
       [ 2.24968346e+00, -5.92373012e-01,  1.67215710e+00,
         1.05393502e+00],
       [ 5.53333275e-01, -8.22569778e-01,  6.49083415e-01,
         7.90670654e-01],
       [ 1.03800476e+00,  5.58610819e-01,  1.10378283e+00,
         1.18556721e+00],
       [ 1.64384411e+00,  3.28414053e-01,  1.27429511e+00,
         7.90670654e-01],
       [ 4.32165405e-01, -5.92373012e-01,  5.92245988e-01,
         7.90670654e-01],
       [ 3.10997534e-01, -1.31979479e-01,  6.49083415e-01,
         7.90670654e-01],
       [ 6.74501145e-01, -5.92373012e-01,  1.04694540e+00,
         1.18556721e+00],
       [ 1.64384411e+00, -1.31979479e-01,  1.16062026e+00,
         5.27406285e-01],
       [ 1.88617985e+00, -5.92373012e-01,  1.33113254e+00,
         9.22302838e-01],
       [ 2.49201920e+00,  1.70959465e+00,  1.50164482e+00,
         1.05393502e+00],
       [ 6.74501145e-01, -5.92373012e-01,  1.04694540e+00,
         1.31719939e+00],
       [ 5.53333275e-01, -5.92373012e-01,  7.62758269e-01,
         3.95774101e-01],
       [ 3.10997534e-01, -1.05276654e+00,  1.04694540e+00,
         2.64141916e-01],
       [ 2.24968346e+00, -1.31979479e-01,  1.33113254e+00,
         1.44883158e+00],
       [ 5.53333275e-01,  7.88807586e-01,  1.04694540e+00,
         1.58046376e+00],
       [ 6.74501145e-01,  9.82172869e-02,  9.90107977e-01,
         7.90670654e-01],
       [ 1.89829664e-01, -1.31979479e-01,  5.92245988e-01,
         7.90670654e-01],
       [ 1.28034050e+00,  9.82172869e-02,  9.33270550e-01,
         1.18556721e+00],
       [ 1.03800476e+00,  9.82172869e-02,  1.04694540e+00,
         1.58046376e+00],
       [ 1.28034050e+00,  9.82172869e-02,  7.62758269e-01,
         1.44883158e+00],
       [-5.25060772e-02, -8.22569778e-01,  7.62758269e-01,
         9.22302838e-01],
       [ 1.15917263e+00,  3.28414053e-01,  1.21745768e+00,
         1.44883158e+00],
       [ 1.03800476e+00,  5.58610819e-01,  1.10378283e+00,
         1.71209594e+00],
       [ 1.03800476e+00, -1.31979479e-01,  8.19595696e-01,
         1.44883158e+00],
       [ 5.53333275e-01, -1.28296331e+00,  7.05920842e-01,
         9.22302838e-01],
       [ 7.95669016e-01, -1.31979479e-01,  8.19595696e-01,
         1.05393502e+00],
       [ 4.32165405e-01,  7.88807586e-01,  9.33270550e-01,
         1.44883158e+00],
       [ 6.86617933e-02, -1.31979479e-01,  7.62758269e-01,
         7.90670654e-01]])
</pre>
### 적절한 클러스터 갯수 구하기 : 엘보우 메소드

- 오차제곱합(SSE)값이 inertia_에 저장되며 이 값을 이용하여 그래프 작성



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
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0,
       2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0,
       0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2])
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 6 columns</p>
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
      <th>cluster_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>108</th>
      <td>6.7</td>
      <td>2.5</td>
      <td>5.8</td>
      <td>1.8</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>6.6</td>
      <td>3.0</td>
      <td>4.4</td>
      <td>1.4</td>
      <td>versicolor</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>131</th>
      <td>7.9</td>
      <td>3.8</td>
      <td>6.4</td>
      <td>2.0</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>132</th>
      <td>6.4</td>
      <td>2.8</td>
      <td>5.6</td>
      <td>2.2</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>6.0</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.6</td>
      <td>versicolor</td>
      <td>2</td>
    </tr>
    <tr>
      <th>82</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>3.9</td>
      <td>1.2</td>
      <td>versicolor</td>
      <td>2</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.5</td>
      <td>2.4</td>
      <td>3.7</td>
      <td>1.0</td>
      <td>versicolor</td>
      <td>2</td>
    </tr>
    <tr>
      <th>106</th>
      <td>4.9</td>
      <td>2.5</td>
      <td>4.5</td>
      <td>1.7</td>
      <td>virginica</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 6 columns</p>
</div>



```python
# 세 가지 종류의 marker로 그래프 상에 좌표 표시

marker = ["^", "s", "o"]
for i, marker in enumerate(marker):
    x_val = df[df["cluster_id"]==i]["sepal_length"]
    y_val = df[df["cluster_id"]==i]["sepal_width"]
    plt.scatter(x_val, y_val, marker=marker)

    
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
```

<pre>
Text(0, 0.5, 'sepal_width')
</pre>
<img src="data:image/png;base64,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"/>

### 차원 축소- PCA(주성분 분석)



```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) # 차원 갯수 설정(그래프 상에 나타내기 위해서 2개로 설정)
pca.fit(df_scaled)

df_pca = pca.transform(df_scaled)

print("축소전:", df_scaled.shape)
print("축소후:", df_pca.shape)
```

<pre>
축소전: (150, 4)
축소후: (150, 2)
</pre>

```python
# 4개의 변수 중 두 칼럼을 가져온 게 아니라 4개의 변수를 두개의 변수로 만든 것
df_pca
```

<pre>
array([[-2.26470281,  0.4800266 ],
       [-2.08096115, -0.67413356],
       [-2.36422905, -0.34190802],
       [-2.29938422, -0.59739451],
       [-2.38984217,  0.64683538],
       [-2.07563095,  1.48917752],
       [-2.44402884,  0.0476442 ],
       [-2.23284716,  0.22314807],
       [-2.33464048, -1.11532768],
       [-2.18432817, -0.46901356],
       [-2.1663101 ,  1.04369065],
       [-2.32613087,  0.13307834],
       [-2.2184509 , -0.72867617],
       [-2.6331007 , -0.96150673],
       [-2.1987406 ,  1.86005711],
       [-2.26221453,  2.68628449],
       [-2.2075877 ,  1.48360936],
       [-2.19034951,  0.48883832],
       [-1.898572  ,  1.40501879],
       [-2.34336905,  1.12784938],
       [-1.914323  ,  0.40885571],
       [-2.20701284,  0.92412143],
       [-2.7743447 ,  0.45834367],
       [-1.81866953,  0.08555853],
       [-2.22716331,  0.13725446],
       [-1.95184633, -0.62561859],
       [-2.05115137,  0.24216355],
       [-2.16857717,  0.52714953],
       [-2.13956345,  0.31321781],
       [-2.26526149, -0.3377319 ],
       [-2.14012214, -0.50454069],
       [-1.83159477,  0.42369507],
       [-2.61494794,  1.79357586],
       [-2.44617739,  2.15072788],
       [-2.10997488, -0.46020184],
       [-2.2078089 , -0.2061074 ],
       [-2.04514621,  0.66155811],
       [-2.52733191,  0.59229277],
       [-2.42963258, -0.90418004],
       [-2.16971071,  0.26887896],
       [-2.28647514,  0.44171539],
       [-1.85812246, -2.33741516],
       [-2.5536384 , -0.47910069],
       [-1.96444768,  0.47232667],
       [-2.13705901,  1.14222926],
       [-2.0697443 , -0.71105273],
       [-2.38473317,  1.1204297 ],
       [-2.39437631, -0.38624687],
       [-2.22944655,  0.99795976],
       [-2.20383344,  0.00921636],
       [ 1.10178118,  0.86297242],
       [ 0.73133743,  0.59461473],
       [ 1.24097932,  0.61629765],
       [ 0.40748306, -1.75440399],
       [ 1.0754747 , -0.20842105],
       [ 0.38868734, -0.59328364],
       [ 0.74652974,  0.77301931],
       [-0.48732274, -1.85242909],
       [ 0.92790164,  0.03222608],
       [ 0.01142619, -1.03401828],
       [-0.11019628, -2.65407282],
       [ 0.44069345, -0.06329519],
       [ 0.56210831, -1.76472438],
       [ 0.71956189, -0.18622461],
       [-0.0333547 , -0.43900321],
       [ 0.87540719,  0.50906396],
       [ 0.35025167, -0.19631173],
       [ 0.15881005, -0.79209574],
       [ 1.22509363, -1.6222438 ],
       [ 0.1649179 , -1.30260923],
       [ 0.73768265,  0.39657156],
       [ 0.47628719, -0.41732028],
       [ 1.2341781 , -0.93332573],
       [ 0.6328582 , -0.41638772],
       [ 0.70266118, -0.06341182],
       [ 0.87427365,  0.25079339],
       [ 1.25650912, -0.07725602],
       [ 1.35840512,  0.33131168],
       [ 0.66480037, -0.22592785],
       [-0.04025861, -1.05871855],
       [ 0.13079518, -1.56227183],
       [ 0.02345269, -1.57247559],
       [ 0.24153827, -0.77725638],
       [ 1.06109461, -0.63384324],
       [ 0.22397877, -0.28777351],
       [ 0.42913912,  0.84558224],
       [ 1.04872805,  0.5220518 ],
       [ 1.04453138, -1.38298872],
       [ 0.06958832, -0.21950333],
       [ 0.28347724, -1.32932464],
       [ 0.27907778, -1.12002852],
       [ 0.62456979,  0.02492303],
       [ 0.33653037, -0.98840402],
       [-0.36218338, -2.01923787],
       [ 0.28858624, -0.85573032],
       [ 0.09136066, -0.18119213],
       [ 0.22771687, -0.38492008],
       [ 0.57638829, -0.1548736 ],
       [-0.44766702, -1.54379203],
       [ 0.25673059, -0.5988518 ],
       [ 1.84456887,  0.87042131],
       [ 1.15788161, -0.69886986],
       [ 2.20526679,  0.56201048],
       [ 1.44015066, -0.04698759],
       [ 1.86781222,  0.29504482],
       [ 2.75187334,  0.8004092 ],
       [ 0.36701769, -1.56150289],
       [ 2.30243944,  0.42006558],
       [ 2.00668647, -0.71143865],
       [ 2.25977735,  1.92101038],
       [ 1.36417549,  0.69275645],
       [ 1.60267867, -0.42170045],
       [ 1.8839007 ,  0.41924965],
       [ 1.2601151 , -1.16226042],
       [ 1.4676452 , -0.44227159],
       [ 1.59007732,  0.67624481],
       [ 1.47143146,  0.25562182],
       [ 2.42632899,  2.55666125],
       [ 3.31069558,  0.01778095],
       [ 1.26376667, -1.70674538],
       [ 2.0377163 ,  0.91046741],
       [ 0.97798073, -0.57176432],
       [ 2.89765149,  0.41364106],
       [ 1.33323218, -0.48181122],
       [ 1.7007339 ,  1.01392187],
       [ 1.95432671,  1.0077776 ],
       [ 1.17510363, -0.31639447],
       [ 1.02095055,  0.06434603],
       [ 1.78834992, -0.18736121],
       [ 1.86364755,  0.56229073],
       [ 2.43595373,  0.25928443],
       [ 2.30492772,  2.62632347],
       [ 1.86270322, -0.17854949],
       [ 1.11414774, -0.29292262],
       [ 1.2024733 , -0.81131527],
       [ 2.79877045,  0.85680333],
       [ 1.57625591,  1.06858111],
       [ 1.3462921 ,  0.42243061],
       [ 0.92482492,  0.0172231 ],
       [ 1.85204505,  0.67612817],
       [ 2.01481043,  0.61388564],
       [ 1.90178409,  0.68957549],
       [ 1.15788161, -0.69886986],
       [ 2.04055823,  0.8675206 ],
       [ 1.9981471 ,  1.04916875],
       [ 1.87050329,  0.38696608],
       [ 1.56458048, -0.89668681],
       [ 1.5211705 ,  0.26906914],
       [ 1.37278779,  1.01125442],
       [ 0.96065603, -0.02433167]])
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
      <td>-2.264703</td>
      <td>0.480027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.080961</td>
      <td>-0.674134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.364229</td>
      <td>-0.341908</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.299384</td>
      <td>-0.597395</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.389842</td>
      <td>0.646835</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>1.870503</td>
      <td>0.386966</td>
    </tr>
    <tr>
      <th>146</th>
      <td>1.564580</td>
      <td>-0.896687</td>
    </tr>
    <tr>
      <th>147</th>
      <td>1.521170</td>
      <td>0.269069</td>
    </tr>
    <tr>
      <th>148</th>
      <td>1.372788</td>
      <td>1.011254</td>
    </tr>
    <tr>
      <th>149</th>
      <td>0.960656</td>
      <td>-0.024332</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 2 columns</p>
</div>



```python
# 새로 만든 데이터에 species 칼럼과 클러스터 번호 칼럼 추가
df_pca["species"] = df.species
df_pca["target"] = df.cluster_id
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
      <th>species</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.264703</td>
      <td>0.480027</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.080961</td>
      <td>-0.674134</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.364229</td>
      <td>-0.341908</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.299384</td>
      <td>-0.597395</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.389842</td>
      <td>0.646835</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>1.870503</td>
      <td>0.386966</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>1.564580</td>
      <td>-0.896687</td>
      <td>virginica</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>1.521170</td>
      <td>0.269069</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>1.372788</td>
      <td>1.011254</td>
      <td>virginica</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>0.960656</td>
      <td>-0.024332</td>
      <td>virginica</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>


### 시각화



```python
# 세 가지 종류의 marker로 그래프 상에 좌표 표시

marker = ["^", "s", "o"]
for i, marker in enumerate(marker):
    x_val = df_pca[df_pca["target"]==i]["pc1"]
    y_val = df_pca[df_pca["target"]==i]["pc2"]
    plt.scatter(x_val, y_val, marker=marker)

    
plt.xlabel("pc1")
plt.ylabel("pc2")
```

<pre>
Text(0, 0.5, 'pc2')
</pre>
<img src="data:image/png;base64,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"/>

## 실제 위의 클러스터링 결과와 실제 품종과 일치하는 비율이 어느정돈지 계산



```python
df_pca['species_acc']= df_pca.species.map({'setosa':1, "versicolor":2, "virginica":0}) # 어떻게 정한겨
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
      <th>species</th>
      <th>target</th>
      <th>species_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.264703</td>
      <td>0.480027</td>
      <td>setosa</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.080961</td>
      <td>-0.674134</td>
      <td>setosa</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.364229</td>
      <td>-0.341908</td>
      <td>setosa</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.299384</td>
      <td>-0.597395</td>
      <td>setosa</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.389842</td>
      <td>0.646835</td>
      <td>setosa</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>1.870503</td>
      <td>0.386966</td>
      <td>virginica</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>1.564580</td>
      <td>-0.896687</td>
      <td>virginica</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>147</th>
      <td>1.521170</td>
      <td>0.269069</td>
      <td>virginica</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>1.372788</td>
      <td>1.011254</td>
      <td>virginica</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>0.960656</td>
      <td>-0.024332</td>
      <td>virginica</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>



```python
num=0
for i in range(len(df_pca)):
    if df_pca["species_acc"][i] == df_pca["target"][i]:
        num+=1
print(num)
```

<pre>
124
</pre>

```python
print(num/len(df_pca))
```

<pre>
0.8266666666666667
</pre>
# 11주차 과제(개인톡)

- 회귀모형을 적합시킨 결과 회귀식이 y=2x+50과 같이 구해졌다.

관측치가 (2,51), (3,56), (5,63), (6,60), (10, 72)일 때 각 관측치로

부터의 오차값을 이용하여 MAE, MSE, RMSE 값을 구하시오.



```python
yy=[51,56,63,60,72]
xx=[2,3,5,6,10]

a= [] # 오차값 저장할 리스트
for i in range(5):
    a.append(abs(yy[i]-(2*xx[i]+50)))
a
```

<pre>
[3, 0, 3, 2, 2]
</pre>

```python
import math

#MAE
mae = sum(a)/5

#MSE
m=0
for i in range(5):
    aa=a[i]**2
    m+=aa
mse=m/5

#RMSE
rmse=math.sqrt(mse)

print("MAE:", mae)
print("MSE:",mse)
print("RMSE:",rmse)
```

<pre>
MAE: 2.0
MSE: 5.2
RMSE: 2.280350850198276
</pre>

```python
```
