---
layout: single
title:  "12th Week Course-4"
categories: coding
tag: [python, blog, jupyter, Word2Vec ]
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


# 최적의 토픽 갯수 구하기

- 그룹을 유사한 것끼리 나눈다고 할 때. 그룹 내에서는 서로 유사도가 높을수록 좋고 토픽 사이에는 이질적일수록 좋다


#### review 데이터 수집 및 전처리



```python
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

df = pd.read_csv('data/review.csv', encoding='utf-8') # 데이터 위치 주의

df
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\gensim\similarities\__init__.py:15: UserWarning: The gensim.similarities.levenshtein submodule is disabled, because the optional Levenshtein package <https://pypi.org/project/python-Levenshtein/> is unavailable. Install Levenhstein (e.g. `pip install python-Levenshtein`) to suppress this warning.
  warnings.warn(msg)
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
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>인셉션은 대단하다 느꼈는데, 인터스텔라는 경이롭다고 느껴진다.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>결론만 말하자면 대박이다 더이상 어떤단어로 칭찬해야하는지도모르겠다.약 3시간의 긴러...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>전율과 환희의 169분이였다. 그 어떤 영화도 시도한 적 없는 명석함과 감동이 담겨...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>이 영화가 명량이나 도둑들보다 관객수가 적다면 진짜 부끄러울듯</td>
    </tr>
    <tr>
      <th>4</th>
      <td>팝콘, 콜라 사가지 마라.. 먹을시간 없다</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>3267</th>
      <td>인류는 걍 다 d져야한다....쓰레기들 살아서 머하니? 전 우주를 다 쓰레기통으로만...</td>
    </tr>
    <tr>
      <th>3268</th>
      <td>초반에 너무 밑밥을 마니깔고 지루했음 전체적으로 작품성은 높음 2시간 30 .</td>
    </tr>
    <tr>
      <th>3269</th>
      <td>과대포장 풀기운동 동참 영환재밋게봄 리뷰다읽고보면 졸게됩니다</td>
    </tr>
    <tr>
      <th>3270</th>
      <td>솔직히 9점대는 아니다 7점정도.. 고로 1점준다</td>
    </tr>
    <tr>
      <th>3271</th>
      <td>완전 재미없음 ㅠㅠ 돈 아까워 그래비티가 백배는 재미있음 다시는 놀란 감독 영화 안...</td>
    </tr>
  </tbody>
</table>
<p>3272 rows × 1 columns</p>
</div>



```python
df = df.dropna(how = 'any') # 결측치 제거
df['review'] = df['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

okt = Okt()
tokenized_data = []

for sentence in df['review']:
    temp_X = okt.nouns(sentence) # 명사만 추출
    temp_X = [word for word in temp_X if len(word)>1] # 2글자 이상만 추출
    tokenized_data.append(temp_X)

pd.DataFrame(tokenized_data).to_csv('data/review_tokenized.csv')
```

<pre>
<ipython-input-3-a752e1ee651a>:2: FutureWarning: The default value of regex will change from True to False in a future version.
  df['review'] = df['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
</pre>

```python
pd.DataFrame(tokenized_data)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>인셉션</td>
      <td>인터스텔라</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>결론</td>
      <td>대박</td>
      <td>이상</td>
      <td>단어</td>
      <td>칭찬</td>
      <td>시간</td>
      <td>러닝</td>
      <td>타임</td>
      <td>시간</td>
      <td>상황</td>
      <td>...</td>
      <td>명작</td>
      <td>몇번</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>전율</td>
      <td>환희</td>
      <td>영화</td>
      <td>시도</td>
      <td>감동</td>
      <td>영화</td>
      <td>놀란</td>
      <td>야심</td>
      <td>능력</td>
      <td>존경</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>영화</td>
      <td>명량</td>
      <td>도둑</td>
      <td>관객수</td>
      <td>진짜</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>팝콘</td>
      <td>콜라</td>
      <td>시간</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
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
      <th>3267</th>
      <td>인류</td>
      <td>쓰레기</td>
      <td>우주</td>
      <td>쓰레기통</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3268</th>
      <td>초반</td>
      <td>밑밥</td>
      <td>전체</td>
      <td>작품</td>
      <td>성은</td>
      <td>시간</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3269</th>
      <td>과대</td>
      <td>포장</td>
      <td>풀기</td>
      <td>운동</td>
      <td>동참</td>
      <td>환재</td>
      <td>리뷰</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3270</th>
      <td>정도</td>
      <td>고로</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3271</th>
      <td>완전</td>
      <td>그래비티</td>
      <td>백배</td>
      <td>놀란</td>
      <td>감독</td>
      <td>영화</td>
      <td>시간</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>3272 rows × 24 columns</p>
</div>


#### 시각화를 위한 pyLDAvis 설치



```python
!pip install pyLDAvis
```

<pre>
Requirement already satisfied: pyLDAvis in c:\users\administrator\anaconda3\lib\site-packages (3.3.1)
Requirement already satisfied: numpy>=1.20.0 in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (1.20.1)
Requirement already satisfied: jinja2 in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (2.11.3)
Requirement already satisfied: pandas>=1.2.0 in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (1.2.4)
Requirement already satisfied: future in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (0.18.2)
Requirement already satisfied: funcy in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (1.17)
Requirement already satisfied: gensim in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (4.0.0)
Requirement already satisfied: setuptools in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (52.0.0.post20210125)
Requirement already satisfied: scikit-learn in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (0.24.1)
Requirement already satisfied: joblib in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (1.0.1)
Requirement already satisfied: numexpr in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (2.7.3)
Requirement already satisfied: sklearn in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (0.0)
Requirement already satisfied: scipy in c:\users\administrator\anaconda3\lib\site-packages (from pyLDAvis) (1.6.2)
Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\administrator\anaconda3\lib\site-packages (from pandas>=1.2.0->pyLDAvis) (2.8.1)
Requirement already satisfied: pytz>=2017.3 in c:\users\administrator\anaconda3\lib\site-packages (from pandas>=1.2.0->pyLDAvis) (2021.1)
Requirement already satisfied: six>=1.5 in c:\users\administrator\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas>=1.2.0->pyLDAvis) (1.15.0)
Requirement already satisfied: Cython==0.29.21 in c:\users\administrator\anaconda3\lib\site-packages (from gensim->pyLDAvis) (0.29.21)
Requirement already satisfied: smart-open>=1.8.1 in c:\users\administrator\anaconda3\lib\site-packages (from gensim->pyLDAvis) (6.0.0)
Requirement already satisfied: MarkupSafe>=0.23 in c:\users\administrator\anaconda3\lib\site-packages (from jinja2->pyLDAvis) (1.1.1)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\administrator\anaconda3\lib\site-packages (from scikit-learn->pyLDAvis) (2.1.0)
</pre>
#### 토픽 갯수별 응집도 구하기

- 토픽의 개수를 늘려가며 토픽을 몇 개로 했을 때 가장 응집도가 좋은지를 보는 것



```python
from tqdm import tqdm
from gensim.models.ldamodel import LdaModel 
from gensim.models.callbacks import CoherenceMetric 
from gensim import corpora 
from gensim.models.callbacks import PerplexityMetric 
import logging 
import pyLDAvis.gensim_models
from gensim.models.coherencemodel import CoherenceModel 
import matplotlib.pyplot as plt 

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3): # 응집도 계산
    coherence_values = [] 
    model_list = [] 
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model) 
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence()) 
    return model_list, coherence_values 

def find_optimal_number_of_topics(dictionary, corpus, processed_data): # 토픽갯수별 응집도 그래프
    limit = 50; #토픽 마지막갯수
    start = 2; #토픽 시작갯수
    step = 6; 
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, 
                                                            texts=processed_data, start=start, limit=limit, step=step) 
    x = range(start, limit, step) 
    plt.plot(x, coherence_values) 
    plt.xlabel("Num Topics") 
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best') 
    plt.show() 


processed_data = [sent.strip().split(",") for sent in tqdm(open('./data/review_tokenized.csv', 'r', encoding='utf-8').readlines())]
dictionary = corpora.Dictionary(processed_data) 
dictionary.filter_extremes(no_below=10, no_above=0.05) # 출현빈도가 적거나 자주 등장하는 단어는 제거 
corpus = [dictionary.doc2bow(text) for text in processed_data]
print('Number of unique tokens: %d' % len(dictionary)) 
print('Number of documents: %d' % len(corpus))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
    
# 최적의 토픽 수 찾기 
find_optimal_number_of_topics(dictionary, corpus, processed_data)
```

<pre>
100%|██████████████████████████████████████████████████████████████████████████| 3273/3273 [00:00<00:00, 328192.33it/s]
2022-05-22 01:03:11,086 : INFO : using symmetric alpha at 0.5
2022-05-22 01:03:11,087 : INFO : using symmetric eta at 0.5
2022-05-22 01:03:11,088 : INFO : using serial LDA version on this node
2022-05-22 01:03:11,089 : INFO : running online (single-pass) LDA training, 2 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:11,089 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:11,091 : INFO : PROGRESS: pass 0, at document #2000/3273
</pre>
<pre>
Number of unique tokens: 246
Number of documents: 3273
</pre>
<pre>
2022-05-22 01:03:12,016 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:12,017 : INFO : topic #0 (0.500): 0.025*"감독" + 0.021*"인생" + 0.020*"다시" + 0.019*"상상력" + 0.018*"평점" + 0.016*"과학" + 0.016*"이해" + 0.015*"인터스텔라" + 0.015*"스토리" + 0.014*"사람"
2022-05-22 01:03:12,018 : INFO : topic #1 (0.500): 0.025*"보고" + 0.021*"그냥" + 0.020*"이해" + 0.019*"처음" + 0.019*"과학" + 0.018*"대박" + 0.017*"인터스텔라" + 0.016*"사람" + 0.016*"경이" + 0.015*"사랑"
2022-05-22 01:03:12,019 : INFO : topic diff=0.701074, rho=1.000000
2022-05-22 01:03:12,472 : INFO : -5.547 per-word bound, 46.8 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:12,472 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:12,830 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:12,832 : INFO : topic #0 (0.500): 0.031*"감독" + 0.028*"평점" + 0.020*"상상력" + 0.019*"다시" + 0.019*"인생" + 0.018*"사람" + 0.015*"스토리" + 0.015*"재미" + 0.015*"중간" + 0.014*"내용"
2022-05-22 01:03:12,833 : INFO : topic #1 (0.500): 0.028*"보고" + 0.024*"이해" + 0.022*"그냥" + 0.019*"처음" + 0.019*"정도" + 0.019*"사람" + 0.017*"소름" + 0.015*"경이" + 0.015*"사랑" + 0.015*"표현"
2022-05-22 01:03:12,834 : INFO : topic diff=0.340773, rho=0.707107
2022-05-22 01:03:12,835 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=2, decay=0.5, chunksize=2000) in 1.75s', 'datetime': '2022-05-22T01:03:12.835307', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:12,837 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:15,361 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:15,381 : INFO : accumulated word occurrence stats for 1746 virtual documents
2022-05-22 01:03:15,507 : INFO : using symmetric alpha at 0.125
2022-05-22 01:03:15,514 : INFO : using symmetric eta at 0.125
2022-05-22 01:03:15,515 : INFO : using serial LDA version on this node
2022-05-22 01:03:15,517 : INFO : running online (single-pass) LDA training, 8 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:15,517 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:15,519 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:16,001 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:16,002 : INFO : topic #6 (0.125): 0.053*"다시" + 0.034*"과학" + 0.033*"영화관" + 0.030*"감독" + 0.027*"스토리" + 0.025*"경이" + 0.024*"처음" + 0.023*"한번" + 0.022*"사람" + 0.020*"로움"
2022-05-22 01:03:16,003 : INFO : topic #4 (0.125): 0.047*"평점" + 0.036*"감독" + 0.036*"보고" + 0.029*"아이맥스" + 0.026*"처음" + 0.023*"작품" + 0.022*"역시" + 0.020*"표현" + 0.017*"이해" + 0.017*"느낌"
2022-05-22 01:03:16,003 : INFO : topic #7 (0.125): 0.049*"이해" + 0.043*"과학" + 0.037*"사람" + 0.021*"다시" + 0.018*"인간" + 0.016*"감독" + 0.015*"차원" + 0.014*"경이" + 0.014*"한번" + 0.014*"그래비티"
2022-05-22 01:03:16,004 : INFO : topic #3 (0.125): 0.040*"보고" + 0.040*"인생" + 0.031*"사람" + 0.028*"인터스텔라" + 0.028*"명작" + 0.026*"감독" + 0.026*"우리" + 0.023*"대박" + 0.020*"표현" + 0.020*"이상"
2022-05-22 01:03:16,004 : INFO : topic #0 (0.125): 0.044*"그냥" + 0.036*"처음" + 0.026*"과학" + 0.026*"후회" + 0.026*"차원" + 0.026*"이영화" + 0.026*"추천" + 0.025*"인생" + 0.020*"극장" + 0.020*"대박"
2022-05-22 01:03:16,005 : INFO : topic diff=3.316885, rho=1.000000
2022-05-22 01:03:16,396 : INFO : -6.157 per-word bound, 71.4 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:16,398 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:16,667 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:16,669 : INFO : topic #7 (0.125): 0.066*"사람" + 0.066*"이해" + 0.043*"과학" + 0.027*"그래비티" + 0.025*"차원" + 0.021*"내내" + 0.019*"초반" + 0.017*"이야기" + 0.015*"작품" + 0.015*"사랑"
2022-05-22 01:03:16,670 : INFO : topic #0 (0.125): 0.075*"그냥" + 0.047*"처음" + 0.043*"후회" + 0.042*"안보" + 0.034*"이영화" + 0.032*"추천" + 0.026*"평점" + 0.022*"차원" + 0.021*"인생" + 0.019*"액션"
2022-05-22 01:03:16,670 : INFO : topic #3 (0.125): 0.065*"보고" + 0.038*"명작" + 0.036*"완전" + 0.033*"스케일" + 0.033*"사람" + 0.031*"느낌" + 0.029*"인생" + 0.026*"최고다" + 0.021*"미래" + 0.020*"우리"
2022-05-22 01:03:16,671 : INFO : topic #2 (0.125): 0.054*"소름" + 0.042*"말로" + 0.042*"대박" + 0.036*"정도" + 0.035*"인셉션" + 0.033*"표현" + 0.030*"이건" + 0.026*"올해" + 0.025*"그냥" + 0.023*"하나"
2022-05-22 01:03:16,672 : INFO : topic #1 (0.125): 0.059*"상상력" + 0.055*"재미" + 0.037*"정도" + 0.023*"반전" + 0.022*"조금" + 0.022*"이론" + 0.019*"물리학" + 0.019*"동안" + 0.019*"이해" + 0.018*"한번"
2022-05-22 01:03:16,673 : INFO : topic diff=0.567638, rho=0.707107
2022-05-22 01:03:16,674 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=8, decay=0.5, chunksize=2000) in 1.16s', 'datetime': '2022-05-22T01:03:16.674447', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:16,679 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:19,754 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:19,777 : INFO : accumulated word occurrence stats for 2347 virtual documents
2022-05-22 01:03:20,284 : INFO : using symmetric alpha at 0.07142857142857142
2022-05-22 01:03:20,285 : INFO : using symmetric eta at 0.07142857142857142
2022-05-22 01:03:20,286 : INFO : using serial LDA version on this node
2022-05-22 01:03:20,288 : INFO : running online (single-pass) LDA training, 14 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:20,288 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:20,289 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:20,726 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:20,728 : INFO : topic #2 (0.071): 0.037*"사랑" + 0.034*"우리" + 0.029*"중간" + 0.022*"인간" + 0.019*"인생" + 0.019*"재미" + 0.019*"블랙홀" + 0.017*"한번" + 0.017*"과학" + 0.017*"상상"
2022-05-22 01:03:20,731 : INFO : topic #0 (0.071): 0.031*"상상" + 0.031*"느낌" + 0.027*"이해" + 0.027*"동안" + 0.023*"존경" + 0.023*"지식" + 0.023*"대해" + 0.023*"감독" + 0.023*"처음" + 0.023*"상상력"
2022-05-22 01:03:20,732 : INFO : topic #8 (0.071): 0.087*"그냥" + 0.052*"감독" + 0.035*"인터스텔라" + 0.024*"작품" + 0.020*"미래" + 0.018*"차원" + 0.018*"인생" + 0.018*"상상력" + 0.018*"정도" + 0.018*"평점"
2022-05-22 01:03:20,733 : INFO : topic #13 (0.071): 0.050*"표현" + 0.046*"인생" + 0.031*"감독" + 0.029*"그냥" + 0.028*"정도" + 0.028*"상상" + 0.024*"이해" + 0.021*"장면" + 0.021*"전율" + 0.018*"인터스텔라"
2022-05-22 01:03:20,735 : INFO : topic #10 (0.071): 0.036*"이상" + 0.035*"스토리" + 0.031*"상상력" + 0.025*"상미" + 0.025*"아이맥스" + 0.022*"정도" + 0.017*"상대성" + 0.017*"한번" + 0.017*"작품" + 0.017*"보고"
2022-05-22 01:03:20,736 : INFO : topic diff=7.204914, rho=1.000000
2022-05-22 01:03:21,090 : INFO : -6.481 per-word bound, 89.3 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:21,091 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:21,321 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:21,322 : INFO : topic #2 (0.071): 0.063*"중간" + 0.049*"사랑" + 0.033*"마지막" + 0.032*"블랙홀" + 0.028*"재미" + 0.026*"지구" + 0.024*"우주여행" + 0.023*"우리" + 0.020*"상상" + 0.019*"지루"
2022-05-22 01:03:21,323 : INFO : topic #11 (0.071): 0.065*"보고" + 0.061*"천재" + 0.045*"한번" + 0.041*"명작" + 0.040*"초반" + 0.039*"크리스토퍼" + 0.032*"추천" + 0.030*"평점" + 0.028*"감독" + 0.027*"처음"
2022-05-22 01:03:21,324 : INFO : topic #7 (0.071): 0.053*"사람" + 0.047*"인셉션" + 0.045*"정도" + 0.042*"차원" + 0.041*"후회" + 0.039*"인터스텔라" + 0.029*"보기" + 0.027*"머리" + 0.024*"이해" + 0.023*"그래비티"
</pre>
<pre>
2022-05-22 01:03:21,325 : INFO : topic #13 (0.071): 0.075*"표현" + 0.050*"정도" + 0.040*"장면" + 0.038*"재미" + 0.036*"실망" + 0.030*"나중" + 0.028*"인생" + 0.025*"스케일" + 0.023*"아빠" + 0.022*"전율"
2022-05-22 01:03:21,327 : INFO : topic #12 (0.071): 0.041*"소름" + 0.040*"계속" + 0.040*"평가" + 0.027*"어디" + 0.026*"관람" + 0.025*"이건" + 0.023*"인류" + 0.023*"내용" + 0.022*"이후" + 0.021*"전율"
2022-05-22 01:03:21,329 : INFO : topic diff=0.566551, rho=0.707107
2022-05-22 01:03:21,330 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=14, decay=0.5, chunksize=2000) in 1.04s', 'datetime': '2022-05-22T01:03:21.329560', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:21,334 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:23,868 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:23,902 : INFO : accumulated word occurrence stats for 2464 virtual documents
2022-05-22 01:03:24,723 : INFO : using symmetric alpha at 0.05
2022-05-22 01:03:24,724 : INFO : using symmetric eta at 0.05
2022-05-22 01:03:24,725 : INFO : using serial LDA version on this node
2022-05-22 01:03:24,727 : INFO : running online (single-pass) LDA training, 20 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:24,728 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:24,729 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:25,169 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:25,170 : INFO : topic #5 (0.050): 0.049*"차원" + 0.047*"경이" + 0.046*"정도" + 0.029*"그냥" + 0.027*"한번" + 0.024*"인터스텔라" + 0.020*"내용" + 0.020*"전혀" + 0.016*"부분" + 0.016*"형제"
2022-05-22 01:03:25,171 : INFO : topic #18 (0.050): 0.078*"보고" + 0.039*"다시" + 0.031*"전율" + 0.027*"마지막" + 0.027*"작품" + 0.023*"여운" + 0.020*"정도" + 0.020*"초반" + 0.020*"그냥" + 0.020*"갈수록"
2022-05-22 01:03:25,172 : INFO : topic #8 (0.050): 0.048*"인생" + 0.045*"경이" + 0.040*"로움" + 0.037*"감독" + 0.027*"이해" + 0.025*"차원" + 0.023*"스토리" + 0.020*"반전" + 0.020*"역시" + 0.020*"대작"
2022-05-22 01:03:25,174 : INFO : topic #3 (0.050): 0.059*"스토리" + 0.029*"아주" + 0.029*"동안" + 0.029*"상상력" + 0.029*"추천" + 0.026*"보고" + 0.022*"인간" + 0.018*"이상" + 0.018*"인생" + 0.015*"존경"
2022-05-22 01:03:25,175 : INFO : topic #11 (0.050): 0.038*"경이" + 0.038*"정도" + 0.029*"인간" + 0.029*"이해" + 0.029*"로움" + 0.024*"그냥" + 0.019*"부분" + 0.019*"혼자" + 0.019*"초반" + 0.019*"화면"
2022-05-22 01:03:25,175 : INFO : topic diff=11.656754, rho=1.000000
2022-05-22 01:03:25,548 : INFO : -6.696 per-word bound, 103.6 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:25,549 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:25,803 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:25,805 : INFO : topic #17 (0.050): 0.164*"이해" + 0.073*"지금" + 0.042*"상대성이론" + 0.037*"재미" + 0.034*"작품" + 0.033*"사람" + 0.028*"이야기" + 0.021*"정도" + 0.021*"무엇" + 0.021*"세계"
2022-05-22 01:03:25,806 : INFO : topic #7 (0.050): 0.116*"완전" + 0.092*"대박" + 0.043*"지구" + 0.040*"조금" + 0.038*"집중" + 0.032*"매우" + 0.029*"우주여행" + 0.029*"가족" + 0.027*"간만" + 0.027*"의미"
2022-05-22 01:03:25,806 : INFO : topic #4 (0.050): 0.084*"기대" + 0.051*"작품" + 0.044*"한번" + 0.044*"감독" + 0.038*"올해" + 0.034*"상상력" + 0.031*"역시" + 0.031*"사람" + 0.027*"모두" + 0.020*"만점"
2022-05-22 01:03:25,807 : INFO : topic #10 (0.050): 0.079*"인생" + 0.055*"인셉션" + 0.051*"인터스텔라" + 0.043*"이영화" + 0.039*"인간" + 0.034*"존재" + 0.033*"다크나이트" + 0.030*"그냥" + 0.027*"대한" + 0.026*"극장"
2022-05-22 01:03:25,809 : INFO : topic #6 (0.050): 0.177*"평점" + 0.050*"평가" + 0.040*"여운" + 0.040*"처음" + 0.031*"그냥" + 0.026*"표현" + 0.025*"언제" + 0.024*"영상" + 0.020*"느낌" + 0.020*"상상력"
2022-05-22 01:03:25,810 : INFO : topic diff=0.558434, rho=0.707107
2022-05-22 01:03:25,810 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=20, decay=0.5, chunksize=2000) in 1.08s', 'datetime': '2022-05-22T01:03:25.810530', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:25,819 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:28,663 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:28,704 : INFO : accumulated word occurrence stats for 2556 virtual documents
2022-05-22 01:03:29,872 : INFO : using symmetric alpha at 0.038461538461538464
2022-05-22 01:03:29,873 : INFO : using symmetric eta at 0.038461538461538464
2022-05-22 01:03:29,874 : INFO : using serial LDA version on this node
2022-05-22 01:03:29,876 : INFO : running online (single-pass) LDA training, 26 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:29,877 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:29,878 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:30,336 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:30,338 : INFO : topic #15 (0.038): 0.055*"평점" + 0.046*"감독" + 0.035*"표현" + 0.035*"스토리" + 0.035*"다시" + 0.034*"명작" + 0.028*"이건" + 0.025*"상상력" + 0.025*"우리" + 0.025*"보고"
2022-05-22 01:03:30,338 : INFO : topic #17 (0.038): 0.049*"과학" + 0.042*"눈물" + 0.035*"한번" + 0.028*"처음" + 0.021*"존재" + 0.021*"안보" + 0.021*"초월" + 0.021*"하나" + 0.021*"사랑" + 0.021*"아이맥스"
2022-05-22 01:03:30,340 : INFO : topic #4 (0.038): 0.049*"보고" + 0.041*"사랑" + 0.041*"처음" + 0.041*"대박" + 0.033*"사람" + 0.033*"스토리" + 0.025*"현실" + 0.025*"마지막" + 0.025*"평점" + 0.025*"영화관"
2022-05-22 01:03:30,340 : INFO : topic #14 (0.038): 0.061*"그냥" + 0.054*"존재" + 0.054*"인간" + 0.038*"상상력" + 0.031*"우리" + 0.028*"차원" + 0.023*"자체" + 0.023*"지구" + 0.023*"블랙홀" + 0.023*"조금"
2022-05-22 01:03:30,341 : INFO : topic #10 (0.038): 0.144*"인생" + 0.061*"사람" + 0.032*"정도" + 0.024*"영화관" + 0.021*"대작" + 0.021*"극장" + 0.021*"동안" + 0.021*"사랑" + 0.018*"한번" + 0.016*"크리스토퍼"
2022-05-22 01:03:30,342 : INFO : topic diff=16.343201, rho=1.000000
2022-05-22 01:03:30,741 : INFO : -6.929 per-word bound, 121.8 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:30,741 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:30,993 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:30,994 : INFO : topic #6 (0.038): 0.210*"역시" + 0.091*"이상" + 0.072*"크리스토퍼" + 0.054*"느낌" + 0.047*"감독" + 0.038*"관심" + 0.020*"간만" + 0.019*"의미" + 0.018*"저런" + 0.017*"과학"
2022-05-22 01:03:30,995 : INFO : topic #10 (0.038): 0.183*"인생" + 0.096*"사람" + 0.068*"대작" + 0.031*"영화관" + 0.029*"개봉" + 0.027*"다운" + 0.025*"동안" + 0.023*"정도" + 0.022*"인터스텔라" + 0.021*"보고"
2022-05-22 01:03:30,996 : INFO : topic #5 (0.038): 0.114*"재미" + 0.103*"반전" + 0.080*"영화관" + 0.068*"가장" + 0.056*"감독" + 0.051*"천재" + 0.021*"재밋" + 0.020*"중간" + 0.017*"몰입" + 0.017*"완전"
2022-05-22 01:03:30,999 : INFO : topic #14 (0.038): 0.083*"인간" + 0.054*"지구" + 0.054*"존재" + 0.046*"상상력" + 0.041*"후반" + 0.041*"자체" + 0.040*"그냥" + 0.038*"차원" + 0.035*"우리" + 0.035*"순간"
2022-05-22 01:03:31,000 : INFO : topic #24 (0.038): 0.068*"감독" + 0.065*"과학" + 0.048*"장면" + 0.047*"후회" + 0.042*"아이맥스" + 0.039*"크리스토퍼" + 0.036*"몰입" + 0.030*"상상력" + 0.027*"인터스텔라" + 0.025*"다크나이트"
2022-05-22 01:03:31,001 : INFO : topic diff=0.534097, rho=0.707107
2022-05-22 01:03:31,002 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=26, decay=0.5, chunksize=2000) in 1.13s', 'datetime': '2022-05-22T01:03:31.002413', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
</pre>
<pre>
2022-05-22 01:03:31,008 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:33,503 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:33,555 : INFO : accumulated word occurrence stats for 2581 virtual documents
2022-05-22 01:03:34,939 : INFO : using symmetric alpha at 0.03125
2022-05-22 01:03:34,939 : INFO : using symmetric eta at 0.03125
2022-05-22 01:03:34,939 : INFO : using serial LDA version on this node
2022-05-22 01:03:34,939 : INFO : running online (single-pass) LDA training, 32 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:34,946 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:34,947 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:35,298 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:35,298 : INFO : topic #29 (0.031): 0.042*"감독" + 0.035*"스토리" + 0.035*"전율" + 0.035*"처음" + 0.028*"상상력" + 0.021*"평가" + 0.021*"내내" + 0.021*"대박" + 0.021*"한번" + 0.021*"상영"
2022-05-22 01:03:35,298 : INFO : topic #20 (0.031): 0.060*"이건" + 0.053*"보고" + 0.030*"이야기" + 0.030*"인터스텔라" + 0.023*"뭔가" + 0.023*"한번" + 0.023*"인셉션" + 0.023*"역시" + 0.023*"러닝" + 0.023*"대작"
2022-05-22 01:03:35,302 : INFO : topic #8 (0.031): 0.060*"과학" + 0.053*"평점" + 0.053*"그냥" + 0.053*"완전" + 0.045*"명작" + 0.045*"대박" + 0.030*"하나" + 0.030*"이상" + 0.023*"이해" + 0.015*"평론가"
2022-05-22 01:03:35,303 : INFO : topic #16 (0.031): 0.067*"정도" + 0.061*"사랑" + 0.041*"평점" + 0.041*"세상" + 0.031*"집중" + 0.031*"과학" + 0.031*"가장" + 0.031*"인생" + 0.021*"감정" + 0.021*"작품"
2022-05-22 01:03:35,304 : INFO : topic #1 (0.031): 0.059*"인생" + 0.043*"사람" + 0.037*"과학" + 0.032*"인간" + 0.027*"경이" + 0.027*"눈물" + 0.027*"로움" + 0.021*"하나" + 0.020*"정도" + 0.019*"아이맥스"
2022-05-22 01:03:35,305 : INFO : topic diff=21.822063, rho=1.000000
2022-05-22 01:03:35,611 : INFO : -7.079 per-word bound, 135.2 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:35,611 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:35,836 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:35,838 : INFO : topic #9 (0.031): 0.077*"이론" + 0.056*"뭔가" + 0.051*"인셉션" + 0.050*"상상" + 0.040*"감독" + 0.039*"상상력" + 0.039*"이건" + 0.027*"그냥" + 0.025*"다시" + 0.025*"상대성"
2022-05-22 01:03:35,839 : INFO : topic #6 (0.031): 0.082*"크리스토퍼" + 0.079*"표현" + 0.048*"경이" + 0.047*"이상" + 0.043*"말로" + 0.042*"중간" + 0.042*"평점" + 0.040*"필요" + 0.031*"만점" + 0.030*"작품"
2022-05-22 01:03:35,840 : INFO : topic #1 (0.031): 0.177*"인생" + 0.048*"사람" + 0.045*"인간" + 0.037*"로움" + 0.033*"경이" + 0.032*"쿠퍼" + 0.030*"눈물" + 0.025*"별로" + 0.021*"연기" + 0.021*"취향"
2022-05-22 01:03:35,841 : INFO : topic #30 (0.031): 0.088*"동안" + 0.080*"부분" + 0.076*"약간" + 0.062*"이해" + 0.040*"감탄" + 0.038*"내내" + 0.029*"장면" + 0.026*"이영화" + 0.024*"초반" + 0.024*"집중"
2022-05-22 01:03:35,842 : INFO : topic #28 (0.031): 0.133*"그냥" + 0.090*"초반" + 0.066*"마지막" + 0.055*"절대" + 0.049*"지금" + 0.040*"중간" + 0.039*"후반" + 0.029*"정도" + 0.026*"존재" + 0.026*"이번"
2022-05-22 01:03:35,843 : INFO : topic diff=0.517331, rho=0.707107
2022-05-22 01:03:35,844 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=32, decay=0.5, chunksize=2000) in 0.90s', 'datetime': '2022-05-22T01:03:35.844409', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:35,854 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:38,254 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:38,300 : INFO : accumulated word occurrence stats for 2589 virtual documents
2022-05-22 01:03:39,870 : INFO : using symmetric alpha at 0.02631578947368421
2022-05-22 01:03:39,870 : INFO : using symmetric eta at 0.02631578947368421
2022-05-22 01:03:39,870 : INFO : using serial LDA version on this node
2022-05-22 01:03:39,870 : INFO : running online (single-pass) LDA training, 38 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
2022-05-22 01:03:39,870 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:39,875 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:40,261 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:40,261 : INFO : topic #6 (0.026): 0.060*"집중" + 0.048*"처음" + 0.048*"영화관" + 0.036*"자체" + 0.036*"경이" + 0.036*"로움" + 0.036*"대박" + 0.036*"정도" + 0.031*"아이맥스" + 0.024*"이해"
2022-05-22 01:03:40,261 : INFO : topic #1 (0.026): 0.063*"감독" + 0.063*"인터스텔라" + 0.042*"부분" + 0.031*"정도" + 0.031*"중간" + 0.031*"보고" + 0.021*"전혀" + 0.021*"대작" + 0.021*"사람" + 0.021*"동안"
2022-05-22 01:03:40,267 : INFO : topic #21 (0.026): 0.077*"한번" + 0.048*"후회" + 0.048*"차원" + 0.039*"영화관" + 0.039*"영상" + 0.039*"스토리" + 0.029*"부분" + 0.025*"처음" + 0.025*"감독" + 0.019*"역시"
2022-05-22 01:03:40,267 : INFO : topic #15 (0.026): 0.053*"보고" + 0.042*"과학" + 0.032*"그냥" + 0.026*"상상력" + 0.026*"경이" + 0.021*"아버지" + 0.021*"이유" + 0.021*"오늘" + 0.021*"현실" + 0.021*"블랙홀"
2022-05-22 01:03:40,268 : INFO : topic #30 (0.026): 0.063*"느낌" + 0.047*"과학" + 0.038*"인생" + 0.033*"대박" + 0.027*"감독" + 0.027*"그냥" + 0.024*"가족" + 0.022*"물리" + 0.022*"다시" + 0.022*"사랑"
2022-05-22 01:03:40,269 : INFO : topic diff=27.123056, rho=1.000000
2022-05-22 01:03:40,568 : INFO : -7.204 per-word bound, 147.5 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:40,569 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:40,789 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:40,789 : INFO : topic #9 (0.026): 0.142*"상상" + 0.060*"물리학" + 0.047*"중간" + 0.038*"깊이" + 0.038*"사람" + 0.030*"내용" + 0.029*"장면" + 0.024*"이상" + 0.023*"뭔가" + 0.020*"감독"
2022-05-22 01:03:40,796 : INFO : topic #19 (0.026): 0.094*"안보" + 0.083*"액션" + 0.063*"아주" + 0.060*"올해" + 0.052*"관객" + 0.051*"영화인" + 0.046*"중간" + 0.043*"명작" + 0.039*"스페이스" + 0.035*"보고"
2022-05-22 01:03:40,798 : INFO : topic #10 (0.026): 0.098*"처음" + 0.070*"완전" + 0.064*"장르" + 0.056*"기분" + 0.049*"평점" + 0.046*"동안" + 0.038*"그냥" + 0.035*"우주여행" + 0.031*"인간" + 0.026*"설명"
2022-05-22 01:03:40,798 : INFO : topic #21 (0.026): 0.110*"후회" + 0.073*"안보" + 0.072*"영상" + 0.067*"한번" + 0.055*"스토리" + 0.047*"차원" + 0.045*"이야기" + 0.045*"부분" + 0.035*"이건" + 0.034*"그대로"
2022-05-22 01:03:40,799 : INFO : topic #13 (0.026): 0.191*"그래비티" + 0.080*"계속" + 0.078*"정도" + 0.044*"가족" + 0.042*"보고" + 0.031*"느낌" + 0.027*"재미" + 0.022*"평점" + 0.021*"이해" + 0.021*"다시"
2022-05-22 01:03:40,800 : INFO : topic diff=0.516262, rho=0.707107
2022-05-22 01:03:40,801 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=38, decay=0.5, chunksize=2000) in 0.93s', 'datetime': '2022-05-22T01:03:40.801318', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:40,809 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:42,930 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:43,005 : INFO : accumulated word occurrence stats for 2612 virtual documents
2022-05-22 01:03:44,990 : INFO : using symmetric alpha at 0.022727272727272728
2022-05-22 01:03:44,990 : INFO : using symmetric eta at 0.022727272727272728
2022-05-22 01:03:44,990 : INFO : using serial LDA version on this node
2022-05-22 01:03:44,992 : INFO : running online (single-pass) LDA training, 44 topics, 1 passes over the supplied corpus of 3273 documents, updating model once every 2000 documents, evaluating perplexity every 3273 documents, iterating 50x with a convergence threshold of 0.001000
</pre>
<pre>
2022-05-22 01:03:44,995 : WARNING : too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
2022-05-22 01:03:44,997 : INFO : PROGRESS: pass 0, at document #2000/3273
2022-05-22 01:03:45,405 : INFO : merging changes from 2000 documents into a model of 3273 documents
2022-05-22 01:03:45,407 : INFO : topic #37 (0.023): 0.049*"한번" + 0.037*"계속" + 0.037*"감탄" + 0.037*"크리스토퍼" + 0.037*"상상" + 0.037*"과학" + 0.025*"전혀" + 0.025*"후회" + 0.025*"더욱" + 0.025*"순간"
2022-05-22 01:03:45,408 : INFO : topic #41 (0.023): 0.083*"경이" + 0.072*"감독" + 0.052*"상상력" + 0.031*"자체" + 0.031*"이론" + 0.021*"존경" + 0.021*"그냥" + 0.021*"대해" + 0.021*"시나리오" + 0.021*"대박"
2022-05-22 01:03:45,408 : INFO : topic #42 (0.023): 0.051*"과학" + 0.038*"내용" + 0.038*"작품" + 0.038*"인생" + 0.038*"한번" + 0.038*"소름" + 0.025*"조금" + 0.025*"상상" + 0.025*"눈물" + 0.025*"상상력"
2022-05-22 01:03:45,409 : INFO : topic #28 (0.023): 0.049*"상상" + 0.033*"대해" + 0.033*"스토리" + 0.033*"평점" + 0.033*"이영화" + 0.033*"시공간" + 0.033*"오늘" + 0.033*"이론" + 0.033*"다시" + 0.017*"극장"
2022-05-22 01:03:45,411 : INFO : topic #13 (0.023): 0.053*"보고" + 0.032*"사람" + 0.032*"평점" + 0.032*"그냥" + 0.032*"지구" + 0.032*"가족" + 0.021*"실제" + 0.021*"마지막" + 0.021*"상상력" + 0.021*"영화로"
2022-05-22 01:03:45,412 : INFO : topic diff=32.544582, rho=1.000000
2022-05-22 01:03:45,823 : INFO : -7.322 per-word bound, 160.0 perplexity estimate based on a held-out corpus of 1273 documents with 3086 words
2022-05-22 01:03:45,823 : INFO : PROGRESS: pass 0, at document #3273/3273
2022-05-22 01:03:46,108 : INFO : merging changes from 1273 documents into a model of 3273 documents
2022-05-22 01:03:46,108 : INFO : topic #31 (0.023): 0.075*"후회" + 0.071*"이해" + 0.056*"집중" + 0.054*"개인" + 0.042*"보고" + 0.039*"그래비티" + 0.035*"아이맥스" + 0.035*"초반" + 0.032*"지루" + 0.032*"로움"
2022-05-22 01:03:46,108 : INFO : topic #34 (0.023): 0.073*"내내" + 0.059*"러닝" + 0.059*"타임" + 0.048*"경험" + 0.034*"느낌" + 0.034*"이영화" + 0.030*"과학" + 0.028*"아빠" + 0.026*"사랑" + 0.026*"해도"
2022-05-22 01:03:46,112 : INFO : topic #26 (0.023): 0.094*"거의" + 0.085*"차원" + 0.060*"상대성이론" + 0.047*"이번" + 0.044*"관람" + 0.037*"가장" + 0.033*"인생" + 0.033*"이해" + 0.032*"결말" + 0.028*"소름"
2022-05-22 01:03:46,114 : INFO : topic #32 (0.023): 0.146*"하나" + 0.116*"명작" + 0.073*"보고" + 0.031*"다른" + 0.031*"그대로" + 0.031*"액션" + 0.028*"전율" + 0.028*"내내" + 0.027*"인터스텔라" + 0.026*"가족"
2022-05-22 01:03:46,115 : INFO : topic #16 (0.023): 0.114*"스케일" + 0.081*"반전" + 0.050*"매우" + 0.050*"후반" + 0.044*"존재" + 0.041*"마지막" + 0.041*"도대체" + 0.038*"인간" + 0.031*"과학" + 0.031*"상상력"
2022-05-22 01:03:46,116 : INFO : topic diff=0.481883, rho=0.707107
2022-05-22 01:03:46,118 : INFO : LdaModel lifecycle event {'msg': 'trained LdaModel(num_terms=246, num_topics=44, decay=0.5, chunksize=2000) in 1.13s', 'datetime': '2022-05-22T01:03:46.118133', 'gensim': '4.0.0', 'python': '3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.19041-SP0', 'event': 'created'}
2022-05-22 01:03:46,125 : INFO : using ParallelWordOccurrenceAccumulator(processes=7, batch_size=64) to estimate probabilities from sliding windows
2022-05-22 01:03:48,187 : INFO : 7 accumulators retrieved from output queue
2022-05-22 01:03:48,233 : INFO : accumulated word occurrence stats for 2609 virtual documents
</pre>
<img src="data:image/png;base64,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"/>


```python
```
