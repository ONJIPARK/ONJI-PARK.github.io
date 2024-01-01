---
layout: single
title:  "3rd Week Course"
categories: coding
tag: [python, blog, jupyter, KoNLPy Task]
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



```python
!pip install JPype1-1.3.0-cp310-cp310-win_amd64.whl
!pip install konlpy
```

<pre>
WARNING: Requirement 'JPype1-1.3.0-cp310-cp310-win_amd64.whl' looks like a filename, but the file does not exist
ERROR: JPype1-1.3.0-cp310-cp310-win_amd64.whl is not a supported wheel on this platform.
</pre>
<pre>
Requirement already satisfied: konlpy in c:\users\administrator\anaconda3\lib\site-packages (0.6.0)
Requirement already satisfied: lxml>=4.1.0 in c:\users\administrator\anaconda3\lib\site-packages (from konlpy) (4.6.3)
Requirement already satisfied: JPype1>=0.7.0 in c:\users\administrator\anaconda3\lib\site-packages (from konlpy) (1.3.0)
Requirement already satisfied: numpy>=1.6 in c:\users\administrator\anaconda3\lib\site-packages (from konlpy) (1.20.1)
</pre>
 


해당 에러 발생시 아래 링크 참조

에러 메시지: No JVM shared library file (jvm.dll) found. Try setting up the JAVA_HOME environment variable properly.



1. 출처: https://stricky.tistory.com/398 [The DataBase that i am good at]

2. https://mola23.tistory.com/84 



```python
from konlpy.tag import Twitter

twitter = Twitter()  #트위터라는 형태소 분석기

print(twitter.pos('빅데이터 수업 너무 재미있다.')) # pos는 품사
```

<pre>
C:\Users\Administrator\anaconda3\lib\site-packages\konlpy\tag\_okt.py:17: UserWarning: "Twitter" has changed to "Okt" since KoNLPy v0.4.5.
  warn('"Twitter" has changed to "Okt" since KoNLPy v0.4.5.')
</pre>
<pre>
[('빅데이터', 'Noun'), ('수업', 'Noun'), ('너무', 'Adverb'), ('재미있다', 'Adjective'), ('.', 'Punctuation')]
</pre>

```python
from konlpy.tag import Twitter

twitter = Twitter()  #트위터라는 형태소 분석기

print(twitter.pos('아버지가방에들어가신다.')) # pos는 품사
```

<pre>
[('아버지', 'Noun'), ('가방', 'Noun'), ('에', 'Josa'), ('들어가신다', 'Verb'), ('.', 'Punctuation')]
</pre>

```python
```
