---
layout: single
title:  "Sixth Week Personal Assignment"
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


# 6주차 개인톡 과제 지니지수, 엔트로피, 정보이득값 구하기

- 60191315 박온지

- 통신사에서 충성고객과 탈퇴고객을 구분하는 규칙을 만들고자 한다.

- 10명의 고객을 대상으로 직업유무와 결혼여부 중 어느 변수가 더 분류 를 잘하는 변수인지 찾고 분류규칙을 찾고자 함

- 두 분류 기준에 따른 지니지수, 엔트로피, 정보이득값을 계산하시오.



```python
import math as np
```

# 직업유무

## 1. 지니지수



```python
#직업유무 지니지수(클래스 종류: 충성, 탈퇴)
(5/10)*(1-5/10) + (5/10)*(1-5/10)
```

<pre>
0.5
</pre>

```python
#직업:유 지니지수(클래스 종류: 충성, 탈퇴)
(5/6)*(1-5/6) + (1/6)*(1-1/6)
```

<pre>
0.2777777777777778
</pre>

```python
#직업:무 지니지수(클래스 종류: 충성, 탈퇴)
(0/4)*(1-0/4) + (4/4)*(1-4/4)
```

<pre>
0.0
</pre>
## 2. 분할전 엔트로피값



```python
#분할전 직업유무 엔트로피값(충성, 탈퇴)
before= - (5/10)*np.log2(5/10)-(5/10)*np.log2(5/10)
before
```

<pre>
1.0
</pre>

```python
#직업: 유 기준 엔트로피값(충성, 탈퇴)
-(5/6)*np.log2(5/6)-(1/6)*np.log2(1/6)
```

<pre>
0.6500224216483541
</pre>

```python
#직업: 무 기준 엔트로피값(충성, 탈퇴)
(-0/4*np.log2(1)-4/4*np.log2(4/4))
```

<pre>
0.0
</pre>
## 3. 평균 엔트로피값



```python
#직업여부 기준 평균 엔트로피값(충성, 탈퇴)
job_mean= 6/10 * (-(5/6)*np.log2(5/6)-(1/6)*np.log2(1/6)) + 4/10 * (-(0/4)*np.log2(1)-(4/4)*np.log2(4/4))
job_mean
```

<pre>
0.39001345298901247
</pre>
## 4. 정도이득값 



```python
#정보이득값 
before-job_mean
```

<pre>
0.6099865470109875
</pre>
# 결혼여부

## 1. 지니지수



```python
#결혼여부 지니지수(클래스 종류: 충성, 탈퇴)
(5/10)*(5/10) + (5/10)*(5/10)
```

<pre>
0.5
</pre>

```python
#결혼:기혼 지니지수(클래스 종류: 충성, 탈퇴)
(2/5)*(3/5) + (3/5)*(2/5)
```

<pre>
0.48
</pre>

```python
#결혼:미혼 지니지수(클래스 종류: 충성, 탈퇴)
(3/5)*(2/5) + (2/5)*(3/5)
```

<pre>
0.48
</pre>
## 2. 분활전 엔트로피값



```python
#분활전 결혼여부 엔트로피값(충성, 탈퇴)
marry_before= - (5/10)*np.log2(5/10)-(5/10)*np.log2(5/10)
marry_before
```

<pre>
1.0
</pre>

```python
#결혼: 기혼 기준 엔트로피값(충성, 탈퇴)
-(2/5)*np.log2(2/5)-(3/5)*np.log2(3/5)
```

<pre>
0.9709505944546686
</pre>

```python
#결혼: 미혼 기준 엔트로피값(충성, 탈퇴)
-(3/5)*np.log2(3/5)-(2/5)*np.log2(2/5)
```

<pre>
0.9709505944546686
</pre>
## 3. 평균 엔트로피값



```python
#결혼여부 기준 평균 엔트로피값(충성, 탈퇴)
marry_mean= 5/10 * (-(2/5)*np.log2(2/5)-(3/5)*np.log2(3/5)) + 5/10 * (-(3/5)*np.log2(3/5)-(2/5)*np.log2(2/5))
marry_mean
```

<pre>
0.9709505944546686
</pre>
## 4. 정보이득값



```python
#정보이득값
marry_before-marry_mean
```

<pre>
0.02904940554533142
</pre>

```python
```
