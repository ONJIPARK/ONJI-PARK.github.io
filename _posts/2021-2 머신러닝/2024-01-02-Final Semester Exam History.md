---
layout: single
title:  "Final Semester Exam History"
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



```python
import math
```

### 성별 지니지수 세 개



```python
(10/20)*(1-10/20)+(10/20)*(1-10/20)
```

<pre>
0.5
</pre>

```python
(8/9)*(1-8/9)+(1/9)*(1-1/9)
```

<pre>
0.1975308641975309
</pre>

```python
(2/11)*(1-2/11)+(9/11)*(1-9/11)
```

<pre>
0.2975206611570248
</pre>
### 분할 전 엔트로피



```python
분할전=(-10/20*math.log2(10/20)-10/20*math.log2(10/20))
분할전
```

<pre>
1.0
</pre>
### 엔트로피값(남자)



```python
남자_엔트로피=(-8/9*math.log2(8/9)-1/9*math.log2(1/9))
남자_엔트로피
```

<pre>
0.5032583347756457
</pre>
### 엔트로피값(여자)



```python
여자_엔트로피=(-2/11*math.log2(2/11)-9/11*math.log2(9/11))
여자_엔트로피
```

<pre>
0.6840384356390417
</pre>
### 평균 엔트로피(성별로 분할)



```python
성별_분할=9/20*(-8/9*math.log2(8/9)-1/9*math.log2(1/9))+11/20*(-2/11*math.log2(2/11)-9/11*math.log2(9/11))
성별_분할
```

<pre>
0.6026873902505135
</pre>
### 정보이득값(성별)



```python
정보이득=분할전-성별_분할
정보이득
```

<pre>
0.39731260974948646
</pre>
### 탑승항구 지니지수 세 개 



```python
(10/20)*(1-10/20)+(10/20)*(1-10/20)
```

<pre>
0.5
</pre>

```python
(4/9)*(1-4/9)+(5/9)*(1-5/9)
```

<pre>
0.49382716049382713
</pre>

```python
(6/11)*(1-6/11)+(5/11)*(1-5/11)
```

<pre>
0.49586776859504134
</pre>
### 분할 전 엔트로피



```python
분할전=(-10/20*math.log2(10/20)-10/20*math.log2(10/20))
분할전
```

<pre>
1.0
</pre>
### 탑승항구 S 엔트로피



```python
s_엔트로피=(-4/9*math.log2(4/9)-5/9*math.log2(5/9))
s_엔트로피
```

<pre>
0.9910760598382222
</pre>
### 탑승항구 Q 엔트로피



```python
q_엔트로피=(-6/11*math.log2(6/11)-5/11*math.log2(5/11))
q_엔트로피
```

<pre>
0.9940302114769565
</pre>
### 평균 엔트로피(탑승항구로 분할)



```python
탑승항구_분할=9/20*(-4/9*math.log2(4/9)-5/9*math.log2(5/9))+11/20*(-6/11*math.log2(6/11)-5/11*math.log2(5/11))
탑승항구_분할
```

<pre>
0.9927008432395261
</pre>
### 정보이득



```python
정보이득=분할전-탑승항구_분할
정보이득
```

<pre>
0.007299156760473879
</pre>

```python
import math
```


```python
(6/7)*(1-6/7)+(1/7)*(1-1/7)
```

<pre>
0.24489795918367352
</pre>

```python
(7/7)*(1-7/7)+(0/7)*(1-0/7)
```

<pre>
0.0
</pre>

```python
-0.5*math.log2(0.5)-0.5*math.log2(0.5)
```

<pre>
1.0
</pre>

```python
-1.0*math.log2(1)-0*math.log2(1)
```

<pre>
-0.0
</pre>

```python
#분할 전 엔트로피
분할전=-9/14*math.log2(9/14)-5/14*math.log2(5/14)
분할전
```

<pre>
0.9402859586706311
</pre>

```python
outlook분할=5/14*(-2/5*math.log2(2/5)-3/5*math.log2(3/5))+4/14*(-4/4*math.log2(4/4)-0/4*math.log2(1))+5/14*(-3/5*math.log2(3/5)-2/5*math.log2(2/5))
outlook분할
```

<pre>
0.6935361388961918
</pre>

```python
정보이득=분할전-outlook분할
정보이득
```

<pre>
0.24674981977443933
</pre>

```python
humidity분할=7/14*(-3/7*math.log2(3/7)-4/7*math.log2(4/7))+7/14*(-6/7*math.log2(6/7)-1/7*math.log2(1/7))
humidity분할
```

<pre>
0.7884504573082896
</pre>

```python
정보이득=분할전-humidity분할
정보이득
```

<pre>
0.15183550136234159
</pre>

```python
temperature분할=4/14*(-2/4*math.log2(2/4)-2/4*math.log2(2/4))+6/14*(-4/6*math.log2(4/6)-2/6*math.log2(2/6))+4/14*(-3/4*math.log2(3/4)-1/4*math.log2(1/4))
temperature분할
```

<pre>
0.9110633930116763
</pre>

```python
정보이득=분할전-temperature분할
정보이득
```

<pre>
0.02922256565895487
</pre>

```python
windy분할=8/14*(-6/8*math.log2(6/8)-2/8*math.log2(2/8))+6/14*(-3/6*math.log2(3/6)-3/6*math.log2(3/6))
windy분할
```

<pre>
0.8921589282623617
</pre>

```python
정보이득=분할전-windy분할
정보이득
```

<pre>
0.04812703040826949
</pre>
# 6주차 개인톡 과제 지니지수, 엔트로피, 정보이득값 구하기

- 60191315 박온지

- 통신사에서 충성고객과 탈퇴고객을 구분하는 규칙을 만들고자 한다.

- 10명의 고객을 대상으로 직업유무와 결혼여부 중 어느 변수가 더 분류 를 잘하는 변수인지 찾고 분류규칙을 찾고자 함

- 두 분류 기준에 따른 지니지수, 엔트로피, 정보이득값을 계산하시오.


### 지니지수 (직업유무 지니지수 값 세 개 )



```python
(5/10)*(1-5/10)+(5/10)*(1-5/10)
```

<pre>
0.5
</pre>

```python
(5/6)*(1-5/6)+(1/6)*(1-1/6)
```

<pre>
0.2777777777777778
</pre>

```python
(0/4)*(1-0/4)+(4/4)*(1-4/4)
```

<pre>
0.0
</pre>
### 분할 전 엔트로피값



```python
분할전_엔트로피=(-5/10*math.log2(5/10)-5/10*math.log2(5/10))
분할전_엔트로피
```

<pre>
1.0
</pre>
### 엔트로피 (직업 유)



```python
(-5/6*math.log2(5/6)-1/6*math.log2(1/6))
```

<pre>
0.6500224216483541
</pre>
### 엔트로피 (직업 무)



```python
(-0/4*math.log2(1)-4/4*math.log2(4/4))
```

<pre>
0.0
</pre>
### 평균 엔트로피 (직업유무로 분할할 경우)



```python
직업유무_분할=6/10*(-5/6*math.log2(5/6)-1/6*math.log2(1/6))+4/10*(-0/4*math.log2(1)-4/4*math.log2(4/4))
직업유무_분할
```

<pre>
0.39001345298901247
</pre>
### 정보이득값



```python
정보이득=분할전_엔트로피-직업유무_분할
정보이득
```

<pre>
0.6099865470109875
</pre>
# 


### 지니지수 (결혼여부 지니지수 값 세 개 )



```python
(5/10)*(1-5/10)+(5/10)*(1-5/10)
```

<pre>
0.5
</pre>

```python
(2/5)*(1-2/5)+(3/5)*(1-3/5)
```

<pre>
0.48
</pre>

```python
(3/5)*(1-3/5)+(2/5)*(1-2/5)
```

<pre>
0.48
</pre>
### 분할 전 엔트로피값



```python
분할전_엔트로피=(-5/10*math.log2(5/10)-5/10*math.log2(5/10))
분할전_엔트로피
```

<pre>
1.0
</pre>
### 엔트로피값(기혼)



```python
(-2/5*math.log2(2/5)-3/5*math.log2(3/5))
```

<pre>
0.9709505944546686
</pre>
### 엔트로피값(미혼)



```python
(-3/5*math.log2(3/5)-2/5*math.log2(2/5))
```

<pre>
0.9709505944546686
</pre>
### 평균 엔트로피(결혼여부로 분할할 경우)



```python
결혼여부_분할=5/10*(-2/5*math.log2(2/5)-3/5*math.log2(3/5))+5/10*(-3/5*math.log2(3/5)-2/5*math.log2(2/5))
결혼여부_분할
```

<pre>
0.9709505944546686
</pre>
### 정보이득값



```python
정보이득=분할전_엔트로피-결혼여부_분할
정보이득
```

<pre>
0.02904940554533142
</pre>
## 기말고사 족보 13번


### 분할 전 엔트로피값



```python
분할전_엔트로피=(-4/8*math.log2(4/8)-4/8*math.log2(4/8))
분할전_엔트로피
```

<pre>
1.0
</pre>
### 평균 엔트로피값(흡연으로 분할)



```python
흡연_분할=4/8*(-3/4*math.log2(3/4)-1/4*math.log2(1/4))+4/8*(-1/4*math.log2(1/4)-3/4*math.log2(3/4))
흡연_분할
```

<pre>
0.8112781244591328
</pre>
### 평균 엔트로피값(음주로 분할)



```python
음주_분할=6/8*(-2/6*math.log2(2/6)-4/6*math.log2(4/6))+2/8*(-2/2*math.log2(2/2)-0/2*math.log2(1))
음주_분할
```

<pre>
0.6887218755408672
</pre>
### 정보이득값



```python
정보이득=분할전_엔트로피-흡연_분할
정보이득
```

<pre>
0.18872187554086717
</pre>

```python
정보이득=분할전_엔트로피-음주_분할
정보이득
```

<pre>
0.31127812445913283
</pre>
### -> 음주를 기준으로 분할했을 때 분할 된 노드들의 평균 엔트로피값이 가장 크게 감소하므로 음주를 선택하는 것이 정보이득이 가장 높음



```python
```
