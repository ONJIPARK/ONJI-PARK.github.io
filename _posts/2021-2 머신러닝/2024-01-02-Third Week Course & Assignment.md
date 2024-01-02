---
layout: single
title:  "Third Week Course & Assignment"
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


리턴값이 없는 함수



```python
def sum(a, b):
    print("%d, %d의 합은 %d입니다." %(a, b, a+b))

sum(3, 4)

s=sum(3, 4)
print(s)
```

<pre>
3, 4의 합은 7입니다.
3, 4의 합은 7입니다.
None
</pre>
입력값도 리턴값도 없는 함수



```python
def say():
    print("Hello")

say()
```

<pre>
Hello
</pre>
입력값이 몇 개가 될 지 모르는 경우



```python
def sum_many(*args):
    sum=0
    for i in args:
        sum=sum+i
    return sum
result=sum_many(1, 2, 3, 4,5,6,7,8,9,10)
print(result)
```

<pre>
55
</pre>
입력값이 몇 개가 될 지 모르는 경우 (arg만이 인수로 올 수 있는 것은 아니다)



```python
def sum_mul(choice, *args):
    if choice=="sum":
        result=0
        for i in args:
            result=result+i
    elif choice=="mul":
        result=1
        for i in args:
            result=result*i
    return result

result=sum_mul("sum", 1,2,3,4,5)
print(result)
result=sum_mul("mul", 1,2,3,4,5)
print(result)
```

<pre>
15
120
</pre>
함수의 리턴갑은 언제나 하나



```python
def sum_and_mul(a, b):
    return a+b, a*b

#a에 7과 12가 반환됨
#여러 리턴값이 하나의 변수에 할당되는 경우에는 자동으로 튜플에 할당해준다!
a=sum_and_mul(3,4)
print(a)

#sum하고 mul에 각각 값을 반환
sum,mul=sum_and_mul(3,4)
print(sum)
print(mul)
```

<pre>
(7, 12)
7
12
</pre>

```python
type(a)
```

<pre>
tuple
</pre>
입력 인자의 초기값 설정하기



```python
def say_myself(name, old, gender=1):
    print("나의 이름은 %s입니다." %name)
    print("나이는 %d입니다."%old)
    if gender:
        print("남자입니다")
    else:
        print("여자입니다")
        
say_myself("홍길동", 27)
say_myself("홍길동", 27, 1)
say_myself("홍길동", 27, 0)
```

<pre>
나의 이름은 홍길동입니다.
나이는 27입니다.
남자입니다
나의 이름은 홍길동입니다.
나이는 27입니다.
남자입니다
나의 이름은 홍길동입니다.
나이는 27입니다.
여자입니다
</pre>
함수 내에서 선언된 변수의 효력 범위



```python
a=1
def vartest(a):
    a=a+1
vartest(a)
print(a)
```

<pre>
1
</pre>

```python
def vartest(b):
    b=b+1
vartest(3)
print(b)
```

함수예제



```python
def calculator(cal,a,b):
    if cal=="sum":
        result=a+b
    elif cal=="sub":
        result=a-b
    elif cal=="mul":
        result=a*b
    elif cal=="div":
        result=a/b
    return result

x=int(input())
y=int(input())
calculator("sum",x,y)
```

<pre>
3
4
</pre>
<pre>
7
</pre>
함수예제 (최솟값 찾기)



```python
nums=[2,4,1,3,5]
def min_num():
    min=1000
    for i in nums:
        if i<min:
            min=i
    return min
print("최솟값은 %s입니다."%min_num())
```

<pre>
최솟값은 1입니다.
</pre>
람다함수



```python
f=lambda x:x**2
print(f(2))

print((lambda x:x+1)(5))
```

<pre>
4
6
</pre>

```python
even_power=lambda x:x**2 if x%2==0 else x
print(even_power(2))
print(even_power(3))
```

<pre>
4
3
</pre>

```python
ex=[1,2,3,4,5]
f=lambda x:x**2
print(list(map(f,ex)))
```

<pre>
[1, 4, 9, 16, 25]
</pre>

```python
ex2=[1,2,3,4,5]
f_2=lambda x,y:x+y
print(list(map(f_2,ex2,ex2)))
```

<pre>
[2, 4, 6, 8, 10]
</pre>

```python
def add(n,m):
    return n+m
print(add(2,3))
```

<pre>
5
</pre>

```python
print((lambda n,m:n+m)(2,3))

람다=lambda n,m:n+m
print(람다(2,3))
```

<pre>
5
5
</pre>

```python
h=lambda n: -abs(n)
print(h(5))
```

<pre>
-5
</pre>

```python
j=lambda n, m:n if n>m else m
j(2,3)
```

<pre>
3
</pre>
과제(단톡방)

-60191315 박온지



```python
import random

#무작위 6개 번호
six_random=[]

#1~45 난수 추출 함수
def random_no():
    return random.randrange(1,46)

while len(six_random)<6:
    s=random_no()
    if s not in six_random:  #6개의 번호가 겹치면 안되니까
        six_random.append(s)  


six_random.sort()
print(six_random)
```

<pre>
[2, 18, 28, 31, 39, 43]
</pre>
과제(개인톡 #1)

60191315 박온지



```python
num=[]
while len(num)<6:
    r=random.randrange(1,16)
    if r not in num:
        num.append(r)
        

count=0
for i in range(6):
    num_input=int(input("1-45 사이의 여섯 개의 번호를 입력하시오: "))
    if num_input in num:
        count+=1
        
num.sort()
print(num)

if count==6:
    print(str(count)+"개"+"-1등")
elif count==5:
    print(str(count)+"개"+"-2등")
elif count==4:
    print(str(count)+"개"+"-3등")
elif count==3:
    print(str(count)+"개"+"-4등")
elif count==2:
    print(str(count)+"개"+"-5등")
elif count==1:
    print(str(count)+"개"+"-6등")
else:
    print("아무것도 맞추지 못했습니다")
```

<pre>
1-45 사이의 여섯 개의 번호를 입력하시오: 1
1-45 사이의 여섯 개의 번호를 입력하시오: 10
1-45 사이의 여섯 개의 번호를 입력하시오: 11
1-45 사이의 여섯 개의 번호를 입력하시오: 15
1-45 사이의 여섯 개의 번호를 입력하시오: 2
1-45 사이의 여섯 개의 번호를 입력하시오: 7
[2, 6, 9, 10, 14, 15]
3개-4등
</pre>
과제(개인톡 #2)

60191315 박온지



```python
five_eng=[]
five_mat=[]
while len(five_eng)<5:
    eng_input=int(input("다섯 명의 영어점수를 입력하시오: "))
    five_eng.append(eng_input)
    
while len(five_mat)<5:
    mat_input=int(input("다섯 명의 수학점수를 입력하시오: "))
    five_mat.append(mat_input)


list1=list(map(lambda x: x+5 if x>=90 else x, five_eng))
list2=list(map(lambda x: x+5 if x>=90 else x, five_mat))

for x,y in zip(list1,list2):
    print(x," ",y)
```

<pre>
다섯 명의 영어점수를 입력하시오: 80
다섯 명의 영어점수를 입력하시오: 90
다섯 명의 영어점수를 입력하시오: 85
다섯 명의 영어점수를 입력하시오: 75
다섯 명의 영어점수를 입력하시오: 55
다섯 명의 수학점수를 입력하시오: 95
다섯 명의 수학점수를 입력하시오: 75
다섯 명의 수학점수를 입력하시오: 70
다섯 명의 수학점수를 입력하시오: 40
다섯 명의 수학점수를 입력하시오: 80
80   100
95   75
85   70
75   40
55   80
</pre>

```python
```
