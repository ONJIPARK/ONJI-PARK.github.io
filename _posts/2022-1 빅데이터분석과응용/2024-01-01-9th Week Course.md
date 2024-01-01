---
layout: single
title:  "9th Week Course"
categories: coding
tag: [python, blog, jupyter, BeautifulSoup]
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


# 리뷰의 긍부정을 분류하는 모델을 만들어보자


## 네이버영화에서 영화 리뷰를 평점과 함께 크롤링



```python
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

pre = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=45290&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=sympathyScore'

review = []
rate = []

for i in range(1, 100): # 100페이지 가져오기
    url = pre + str(i)
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    
    id_list = []
    id_pre = '_filtered_ment_'
    
    for i in range(10):
        id_list.append(id_pre + str(i))
    
    for id in id_list:
        review.append(soup.find('span', {'id':id}).get_text().strip())
        
    rate_list = []
    rate_list = (soup.select('div.star_score > em'))
    
    for r in rate_list:
        r = int(re.sub('<.+?>', '', str(r)))
        rate.append(r)
```


```python
df1 = pd.DataFrame({'review' : review, 'rate': rate}) # 리뷰 1000개 정도 끌어오기
df1
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
      <th>review</th>
      <th>rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>인셉션은 대단하다 느꼈는데, 인터스텔라는 경이롭다고 느껴진다.</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>결론만 말하자면 대박이다 더이상 어떤단어로 칭찬해야하는지도모르겠다.약 3시간의 긴러...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>전율과 환희의 169분이였다. 그 어떤 영화도 시도한 적 없는 명석함과 감동이 담겨...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이 영화가 명량이나 도둑들보다 관객수가 적다면 진짜 부끄러울듯</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>팝콘, 콜라 사가지 마라.. 먹을시간 없다</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>985</th>
      <td>안보면 죽을지도 모릅니다... 꼭 봐야합니다...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>986</th>
      <td>무슨 말이 필요한가 ...나중에 커서 아들낳으면 아빠는 인터스텔라 영화관에서 봤어 ...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>987</th>
      <td>여친이 옆에서 자고 일어나더니, 평점 7점짜리 영화라네요. 잠깐 헤어질까 생각했습니다.</td>
      <td>10</td>
    </tr>
    <tr>
      <th>988</th>
      <td>우주인 인성검사의 중요성을 절실히 느낌</td>
      <td>10</td>
    </tr>
    <tr>
      <th>989</th>
      <td>이 영화를 보고 2번 놀랐다아니, 사람이 어떻게 이런걸 생각해 낼 수가 있지?아니,...</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>990 rows × 2 columns</p>
</div>


## 평점 높은 리뷰와 낮은 리뷰만 수집

- 위 100개 정도의 평점은, 좋은 리뷰 비중이 더 많으므로 공평하게 학습시키기 위해 평점 높은, 낮은 리뷰 모두 동일한 개수만큼 학습시키지 위함



```python
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

# 평점 높은 순, 낮은 순 url

pres = ["https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=45290&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=highest&page=",
       "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=45290&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=lowest&page="]

review = []
rate = []
target = []

for pre in pres:
    for i in range(1,200):
        url = pre + str(i)
        res = requests.get(url)

        soup = BeautifulSoup(res.content,'html.parser')

        id_list = []
        id_pre = "_filtered_ment_"

        for i in range(10):
            id_list.append(id_pre+str(i))

        for id in id_list:
            review.append(soup.find("span", {"id":id}).get_text().strip())

        rate_list = []
        rate_list = (soup.select("div.star_score > em"))

        for i in range(10):
            r = int(re.sub("<.+?>", "", str(rate_list[i])))
            rate.append(r)
            if r >= 8:
                target.append(1) # 평점이 8 이상인 경우 고득점인 1 지정
            elif r<=4:
                target.append(0)
            else:
                target.append(-1)

df = pd.DataFrame({"review" : review, "rate" : rate, "target":target})
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
      <th>review</th>
      <th>rate</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>브금 진짜 소름돋아요. 우주에 있는 것 처럼 .</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>내 인생 최고의 영화.딱 이거임</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3975</th>
      <td>과대포장은 인간의 뇌를 갉아먹는다!</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3976</th>
      <td>난해하고 억지스럽다. 이영화가 왜 이렇게 흥행하는지 이해할 수 없다. 사랑의 힘으로...</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3977</th>
      <td>이해력이  낮아  이해못했음</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3978</th>
      <td>심야시간에 보았다 + 졸렸다+볼거리 없다 왜케 평점이 높은건지 이해가 안간다(알바들...</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3979</th>
      <td>너무 기대를 해서 그런지....별로라는 생각이 드네요...약간 지루했습니다..</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3980 rows × 3 columns</p>
</div>



```python
df.rate.unique() # 평점 분류가 10,1,2,3 뿐임. 아무래도 인터스텔라는 평점이 높은 경우가 많아서 인 듯
```

<pre>
array([10,  1,  2,  3], dtype=int64)
</pre>
## 트레이닝/테스트셋 분리



```python
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(review, target, test_size=0.2, random_state=0)
```


```python
len(train_x), len(train_y)
```

<pre>
(3184, 3184)
</pre>

```python
len(test_x), len(test_y)
```

<pre>
(796, 796)
</pre>

```python
```

## 학습(적합)과 성능평가



```python
!pip install konlpy
```

<pre>
Requirement already satisfied: konlpy in c:\users\administrator\anaconda3\lib\site-packages (0.6.0)
Requirement already satisfied: lxml>=4.1.0 in c:\users\administrator\anaconda3\lib\site-packages (from konlpy) (4.6.3)
Requirement already satisfied: numpy>=1.6 in c:\users\administrator\anaconda3\lib\site-packages (from konlpy) (1.20.1)
Requirement already satisfied: JPype1>=0.7.0 in c:\users\administrator\anaconda3\lib\site-packages (from konlpy) (1.3.0)
</pre>

```python
# 텍스트 데이터의 벡터화

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
okt = Okt()

tfv = TfidfVectorizer(tokenizer=okt.morphs, ngram_range=(1,2), min_df=3,max_df=0.9)
tfv.fit(train_x) # 모델 적합(fitting)

tfv_train_x = tfv.transform(train_x)
tfv_train_x
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:489: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn("The parameter 'token_pattern' will not be used"
</pre>
<pre>
<3184x4868 sparse matrix of type '<class 'numpy.float64'>'
	with 60966 stored elements in Compressed Sparse Row format>
</pre>

```python
# 로지스틱회귀모형으로 모델 적합

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression(random_state = 0)
params = {'C': [1,3,5,7,9]}
grid_cv = GridSearchCV(clf, param_grid=params, cv=4, scoring='accuracy', verbose=1)
grid_cv.fit(tfv_train_x, train_y) # 모델 적합(fitting)
```

<pre>
Fitting 4 folds for each of 5 candidates, totalling 20 fits
</pre>
<pre>
GridSearchCV(cv=4, estimator=LogisticRegression(random_state=0),
             param_grid={'C': [1, 3, 5, 7, 9]}, scoring='accuracy', verbose=1)
</pre>

```python
# 테스트 데이터를 통한 예측 정확도 산출
tfv_test_x = tfv.transform(test_x)
grid_cv.best_estimator_.score(tfv_test_x, test_y)
```

<pre>
0.878140703517588
</pre>

```python
# 영화리뷰에 대한 긍부정 분류

my_review = tfv.transform([input()])
if(grid_cv.best_estimator_.predict(my_review)):
    print("긍정 리뷰")
else:
    print("부정 리뷰")
```

<pre>
정말 시간 가는 줄 모르고 봤다
긍정 리뷰
</pre>

```python
```


```python
```

# 9주차 개인톡 과제(60191315 박온지)

- 자신이 좋아하는 영화를 임의로 선정하여 리뷰를 수집하고 모델을 학습시키되 학습시키는 리뷰의 수가 많은 모델과 적은 모델을 만들고나서 각각 아래 예시 문장에 대해 몇 개나 잘 분류하는지 성능을 검증하시오.

1. 시간 아깝네요.

2. 스토리 전개가 고구마 먹는 듯 답답

3. 마지막까지 손에 땀을 쥐며 잘 봤습니다.

4. 도무지 주제도 모르겠고 멀 전달하려는지 모르겠음

5. 별로 추천하고 싶진 않네요.

6. 몰입도가 높은 영화입니다. 시간 가는 줄 몰랐네요.

7. 곱씹을수록 잘 만들어진 영화인 것 같아요. 다시 볼 겁니다. 

8. 전개가 어이가 없네

9. 평점 알바 왜이렇게 많냐

10. 일상의 고단함을 잊게 해주었던 인생영화


## 적은 리뷰: 영화 마음이 리뷰 수 4000개 



```python
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

# 평점 높은 순, 낮은 순 url

pres = ["https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=42809&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=highest&page=",
       "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=42809&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=lowest&page="]

review = []
rate = []
target = []

for pre in pres:
    for i in range(1,200):
        url = pre + str(i)
        res = requests.get(url)

        soup = BeautifulSoup(res.content,'html.parser')

        id_list = []
        id_pre = "_filtered_ment_"

        for i in range(10):
            id_list.append(id_pre+str(i))

        for id in id_list:
            review.append(soup.find("span", {"id":id}).get_text().strip())

        rate_list = []
        rate_list = (soup.select("div.star_score > em"))

        for i in range(10):
            r = int(re.sub("<.+?>", "", str(rate_list[i])))
            rate.append(r)
            if r >= 6:
                target.append(1) # 평점이 6 이상인 경우 고득점인 1 지정
            elif r<=5:
                target.append(0)

df = pd.DataFrame({"review" : review, "rate" : rate, "target":target})
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
      <th>review</th>
      <th>rate</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>이 영화를 볼때 하얀강아지를 안고들어온 부모님 영화에서 장군이라는 새끼강아지가 나올...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ㅠㅠ봤던 영화중에 손꼽히게 슬퍼요ㅠㅠ 내내 울었어요 ㅠㅠ 어휴ㅠㅠ</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>강아지 키우는 입장으로 마음이 역 강아지가 너무 불쌍한듯 맞는 씬 이나 투견씬은 실...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>초딩때 친구랑 영화관에서 본 것..엄청 울었음 예고편만 봐도 슬픔 ㅠㅠ</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3975</th>
      <td>진짜 캐강추</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3976</th>
      <td>아직도 마음속의 감동이 가시질 않네</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3977</th>
      <td>마음이~ 사랑해요~ 죽지는 않았겠죠? 저 많이 울었는데...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3978</th>
      <td>마음이...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3979</th>
      <td>저 언제 영화볼때 시지브이가서 울었음ㅋㅋㅋ</td>
      <td>10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3980 rows × 3 columns</p>
</div>



```python
df.rate.unique() # 평점 분류가 골고루 있음
```

<pre>
array([10,  1,  2,  3,  4,  5,  6,  7,  8,  9], dtype=int64)
</pre>
## 트레이닝/테스트셋 분리



```python
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(review, target, test_size=0.2, random_state=0)
```


```python
len(train_x), len(train_y)
```

<pre>
(3184, 3184)
</pre>

```python
len(test_x), len(test_y)
```

<pre>
(796, 796)
</pre>

```python
# 텍스트 데이터의 벡터화

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
okt = Okt()

tfv = TfidfVectorizer(tokenizer=okt.morphs, ngram_range=(1,2), min_df=3,max_df=0.9)
tfv.fit(train_x) # 모델 적합(fitting)

tfv_train_x = tfv.transform(train_x)
tfv_train_x
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:489: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn("The parameter 'token_pattern' will not be used"
</pre>
<pre>
<3184x3080 sparse matrix of type '<class 'numpy.float64'>'
	with 40563 stored elements in Compressed Sparse Row format>
</pre>

```python
# 로지스틱회귀모형으로 모델 적합

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression(random_state = 0)
params = {'C': [1,3,5,7,9]}
grid_cv = GridSearchCV(clf, param_grid=params, cv=4, scoring='accuracy', verbose=1)
grid_cv.fit(tfv_train_x, train_y) # 모델 적합(fitting)
```

<pre>
Fitting 4 folds for each of 5 candidates, totalling 20 fits
</pre>
<pre>
GridSearchCV(cv=4, estimator=LogisticRegression(random_state=0),
             param_grid={'C': [1, 3, 5, 7, 9]}, scoring='accuracy', verbose=1)
</pre>

```python
# 테스트 데이터를 통한 예측 정확도 산출
tfv_test_x = tfv.transform(test_x)
grid_cv.best_estimator_.score(tfv_test_x, test_y)
```

<pre>
0.9371859296482412
</pre>

```python
# 영화리뷰에 대한 긍부정 분류

rv=["시간 아깝네요.","스토리 전개가 고구마 먹는 듯 답답","마지막까지 손에 땀을 쥐며 잘 봤습니다.",
    "도무지 주제도 모르겠고 멀 전달하려는지 모르겠음","별로 추천하고 싶진 않네요.",
    "몰입도가 높은 영화입니다. 시간 가는 줄 몰랐네요.","곱씹을수록 잘 만들어진 영화인 것 같아요. 다시 볼 겁니다.",
    "전개가 어이가 없네","평점 알바 왜이렇게 많냐","일상의 고단함을 잊게 해주었던 인생영화"]

my_review = tfv.transform(rv)
grid_cv.best_estimator_.predict(my_review)
```

<pre>
array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1])
</pre>
1. 시간 아깝네요.                                    

2. 스토리 전개가 고구마 먹는 듯 답답

3. 마지막까지 손에 땀을 쥐며 잘 봤습니다.

4. 도무지 주제도 모르겠고 멀 전달하려는지 모르겠음

5. 별로 추천하고 싶진 않네요.

6. 몰입도가 높은 영화입니다. 시간 가는 줄 몰랐네요.

7. 곱씹을수록 잘 만들어진 영화인 것 같아요. 다시 볼 겁니다. 

8. 전개가 어이가 없네

9. 평점 알바 왜이렇게 많냐

10. 일상의 고단함을 잊게 해주었던 인생영화



#### 실제 답: 0, 0, 1, 0, 0, 1, 1, 0, 0, 1

#### 모델 답: 1, 1, 1, 1, 0, 1, 1, 1, 0, 1



### -> 실제 정답률은 60% 정도


 


 


## 많은 리뷰: 영화 과속스캔들 리뷰 수 31000개 



```python
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

# 평점 높은 순, 낮은 순 url

pres = ["https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=51143&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=highest&page=",
       "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=51143&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=lowest&page="]

review = []
rate = []
target = []

for pre in pres:
    for i in range(1,200):
        url = pre + str(i)
        res = requests.get(url)

        soup = BeautifulSoup(res.content,'html.parser')

        id_list = []
        id_pre = "_filtered_ment_"

        for i in range(10):
            id_list.append(id_pre+str(i))

        for id in id_list:
            review.append(soup.find("span", {"id":id}).get_text().strip())

        rate_list = []
        rate_list = (soup.select("div.star_score > em"))

        for i in range(10):
            r = int(re.sub("<.+?>", "", str(rate_list[i])))
            rate.append(r)
            if r >= 6:
                target.append(1) # 평점이 6 이상인 경우 고득점인 1 지정
            elif r<=5:
                target.append(0)

df = pd.DataFrame({"review" : review, "rate" : rate, "target":target})
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
      <th>review</th>
      <th>rate</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>어렸을 때 봤는데 가끔 생각나서 보러 올때마다 여운이 남는 영화네요 ㅎㅎ 연말에 가...</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>재미+웃음+감동 3박자가 고루 갖춰진명가족영화!!</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>박보영 팬이 되게 해준 행복한 영화</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>너무 좋은영화, 이 시대에도 이런 영화 다시 보고파용</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>지금봐도 손색없네.군더더기 줄이는데 성공.깔끔하다.</td>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3975</th>
      <td>솔직히 웃음코드가 달라서그런지모르겟지만,,그저그런영화였슴,,</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3976</th>
      <td>그저그럼</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3977</th>
      <td>에에.. 뭐랄까.. 솔직히 말해서 돈 내고 볼 정도는 아니라고 생각되네요..</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3978</th>
      <td>기대가 너무 컸었나 .</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3979</th>
      <td>재미와 감동이 적절히 버무려진 영화 .. 나도 전문가 되고픔 ㅠㅠ</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3980 rows × 3 columns</p>
</div>



```python
df.rate.unique() 
```

<pre>
array([10,  1,  2,  3,  4,  5], dtype=int64)
</pre>
## 트레이닝/테스트셋 분리



```python
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(review, target, test_size=0.2, random_state=0)
```


```python
len(train_x), len(train_y)
```

<pre>
(3184, 3184)
</pre>

```python
len(test_x), len(test_y)
```

<pre>
(796, 796)
</pre>

```python
# 텍스트 데이터의 벡터화

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
okt = Okt()

tfv = TfidfVectorizer(tokenizer=okt.morphs, ngram_range=(1,2), min_df=3,max_df=0.9)
tfv.fit(train_x) # 모델 적합(fitting)

tfv_train_x = tfv.transform(train_x)
tfv_train_x
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\sklearn\feature_extraction\text.py:489: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
  warnings.warn("The parameter 'token_pattern' will not be used"
</pre>
<pre>
<3184x2641 sparse matrix of type '<class 'numpy.float64'>'
	with 31789 stored elements in Compressed Sparse Row format>
</pre>

```python
# 로지스틱회귀모형으로 모델 적합

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression(random_state = 0)
params = {'C': [1,3,5,7,9]}
grid_cv = GridSearchCV(clf, param_grid=params, cv=4, scoring='accuracy', verbose=1)
grid_cv.fit(tfv_train_x, train_y) # 모델 적합(fitting)
```

<pre>
Fitting 4 folds for each of 5 candidates, totalling 20 fits
</pre>
<pre>
GridSearchCV(cv=4, estimator=LogisticRegression(random_state=0),
             param_grid={'C': [1, 3, 5, 7, 9]}, scoring='accuracy', verbose=1)
</pre>

```python
# 테스트 데이터를 통한 예측 정확도 산출
tfv_test_x = tfv.transform(test_x)
grid_cv.best_estimator_.score(tfv_test_x, test_y)
```

<pre>
0.8844221105527639
</pre>

```python
# 영화리뷰에 대한 긍부정 분류

rv=["시간 아깝네요.","스토리 전개가 고구마 먹는 듯 답답","마지막까지 손에 땀을 쥐며 잘 봤습니다.",
    "도무지 주제도 모르겠고 멀 전달하려는지 모르겠음","별로 추천하고 싶진 않네요.",
    "몰입도가 높은 영화입니다. 시간 가는 줄 몰랐네요.","곱씹을수록 잘 만들어진 영화인 것 같아요. 다시 볼 겁니다.",
    "전개가 어이가 없네","평점 알바 왜이렇게 많냐","일상의 고단함을 잊게 해주었던 인생영화"]

my_review = tfv.transform(rv)
grid_cv.best_estimator_.predict(my_review)
```

<pre>
array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
</pre>
1. 시간 아깝네요.                                    

2. 스토리 전개가 고구마 먹는 듯 답답

3. 마지막까지 손에 땀을 쥐며 잘 봤습니다.

4. 도무지 주제도 모르겠고 멀 전달하려는지 모르겠음

5. 별로 추천하고 싶진 않네요.

6. 몰입도가 높은 영화입니다. 시간 가는 줄 몰랐네요.

7. 곱씹을수록 잘 만들어진 영화인 것 같아요. 다시 볼 겁니다. 

8. 전개가 어이가 없네

9. 평점 알바 왜이렇게 많냐

10. 일상의 고단함을 잊게 해주었던 인생영화



#### 실제 답: 0, 0, 1, 0, 0, 1, 1, 0, 0, 1

#### 모델 답: 0, 0, 1, 1, 0, 1, 1, 0, 0, 1



### -> 실제 정답률은 90% 정도. 리뷰 수가 많은 영화는 정확도 측면에서 더 우수한 수치를 보여준다



```python
```
