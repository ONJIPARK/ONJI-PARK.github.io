---
layout: single
title:  "Second Course Assignment"
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


과제1



```python

S='So Jesus said to the Jews who had believed him, "If you abide in my word, you are truly my disciples, and you will know the truth, and the truth will set you free."'
S2=S[135:-2]
print(S2+"\n")
print(S2+"\n")
print(S2+"\n")
```

<pre>
the truth will set you free

the truth will set you free

the truth will set you free

</pre>
과제2



```python
review="이 강의가 제일 좋아!"
review.split()
if len(review)>=2:
    print("정상 리뷰입니다.")
else:
    print("비정상 리뷰입니다.")
```

<pre>
정상 리뷰입니다.
</pre>
과제3



```python
review="이 수업 너무 지겨워!"
if "수업" in review:
    review=review.replace("수업", "강의")
if "지겨워" in review:
    review=review.replace("지겨워", "즐거워")
print(review)
```

<pre>
이 강의 너무 즐거워!
</pre>
과제4



```python
과일=[]
과일.extend(["사과"])
과일.extend(["딸기"])
과일.extend(["포도"])
print(과일)
```

<pre>
['사과', '딸기', '포도']
</pre>

```python
fruit=input()
if fruit in 과일:
        print("yes")
else:
        print("no")
```

<pre>
사과
yes
</pre>
과제5



```python
n=int(input())
for i in range(n):
    print("hi")
```

<pre>
5
hi
hi
hi
hi
hi
</pre>
과제6



```python
N=int(input())
```

<pre>
5
</pre>

```python
start=0
while start<N:
    start+=1
    print(start)
```

<pre>
1
2
3
4
5
</pre>
과제7



```python
단=int(input("단을 입력하시오: "))
```

<pre>
단을 입력하시오: 3
</pre>

```python
for i in range(1,10):
        print("{0} X {1} = {2}".format(단,i,단*i))
```

<pre>
3 X 1 = 3
3 X 2 = 6
3 X 3 = 9
3 X 4 = 12
3 X 5 = 15
3 X 6 = 18
3 X 7 = 21
3 X 8 = 24
3 X 9 = 27
</pre>
과제8



```python
for j in range(2,8):
    for k in range(1,10):
        print(j*k, end=" ")
    print("")
```

<pre>
2 4 6 8 10 12 14 16 18 
3 6 9 12 15 18 21 24 27 
4 8 12 16 20 24 28 32 36 
5 10 15 20 25 30 35 40 45 
6 12 18 24 30 36 42 48 54 
7 14 21 28 35 42 49 56 63 
</pre>

```python
```
