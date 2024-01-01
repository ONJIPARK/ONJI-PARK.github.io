---
layout: single
title:  "8th Week Course"
categories: coding
tag: [python, blog, jupyter, Selenium]
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


## 7주차 개인톡 과제풀이


#### 정규식에서 그룹 사용하기

- 패턴 안에서 정규표현식을 ()로 묶음



```python
import re

std_no = '''
60163456
60191234
'''
```


```python
# 학번8자리를 2자리, 2자리, 4자리의 세 그룹으로 분할
pat2 = re.compile("(\d{2})(\d{2})(\d{4})")
```


```python
# 학번만 가져와보자

print(pat2.sub("\g<2>", std_no)) # 입학년도는 두번째 그룹이지
```

<pre>

16
19

</pre>

```python
# 입학년도를 별표로 바꿔보자

print(pat2.sub("\g<1>**\g<3>", std_no))
```

<pre>

60**3456
60**1234

</pre>
### 불필요한 문자/문자열 제거



```python
import re
text = 'I love you. \t사랑해.'

eng = "[^a-zA-Z ]" # 영문이 아닌 것들
kor = "[^가-힣ㄱ-ㅎㅏ-ㅢ ]" # 한글이 아닌 것들. 끝에 한 칸 공백을 주면 글자 사이에 공백도 포함된다

r1 = re.sub(eng, '', text) # 영문이 아닌 것을 만났을 때 공백으로 치환
r2 = re.sub(kor, '', text) # 한글이 아닌 것을 만났을 때 공백으로 치환
```


```python
print(r1)
print(r2.strip())
```

<pre>
I love you 
사랑해
</pre>

```python
text = '저의 이메일 주소는 hjlee1609@gmail.com입니다.'

sub_word = '<E-mail>'

p = re.compile('[a-zA-Z0-9\-_.]+@[a-zA-Z0-9\-_.]+') # 아이디에 알파벳,숫자,하이픈, 점 등이 들어갈 수 있지

r = re.sub(p, sub_word, text)
```


```python
print(r)
```

<pre>
저의 이메일 주소는 <E-mail>입니다.
</pre>

```python
text = '명지대학교 경영대(경영정보학과, 경영학과, 경상통계학과)의 취업률이 예년 대비 3.5% 상승하였습니다.'

c = re.compile('\(+[가-힣0-9, ]+\)') # 괄호 안에 있는 한글, 숫자, 쉼표, 공백 등 1개 이상
r = re.sub(c, '', text) # 정의한 문자열을 만나면 공백으로 치환
print(r)
```

<pre>
명지대학교 경영대의 취업률이 예년 대비 3.5% 상승하였습니다.
</pre>
# Selenium

- https://stackoverflow.com/questions/17361742/download-image-with-selenium-python



```python
!pip install selenium
```

<pre>
Collecting selenium
  Downloading selenium-4.1.3-py3-none-any.whl (968 kB)
Collecting trio-websocket~=0.9
  Downloading trio_websocket-0.9.2-py3-none-any.whl (16 kB)
Collecting trio~=0.17
  Downloading trio-0.20.0-py3-none-any.whl (359 kB)
Requirement already satisfied: urllib3[secure,socks]~=1.26 in c:\users\administrator\anaconda3\lib\site-packages (from selenium) (1.26.4)
Requirement already satisfied: idna in c:\users\administrator\anaconda3\lib\site-packages (from trio~=0.17->selenium) (2.10)
Requirement already satisfied: async-generator>=1.9 in c:\users\administrator\anaconda3\lib\site-packages (from trio~=0.17->selenium) (1.10)
Requirement already satisfied: attrs>=19.2.0 in c:\users\administrator\anaconda3\lib\site-packages (from trio~=0.17->selenium) (20.3.0)
Requirement already satisfied: sniffio in c:\users\administrator\anaconda3\lib\site-packages (from trio~=0.17->selenium) (1.2.0)
Collecting outcome
  Downloading outcome-1.1.0-py2.py3-none-any.whl (9.7 kB)
Requirement already satisfied: sortedcontainers in c:\users\administrator\anaconda3\lib\site-packages (from trio~=0.17->selenium) (2.3.0)
Requirement already satisfied: cffi>=1.14 in c:\users\administrator\anaconda3\lib\site-packages (from trio~=0.17->selenium) (1.14.5)
Requirement already satisfied: pycparser in c:\users\administrator\anaconda3\lib\site-packages (from cffi>=1.14->trio~=0.17->selenium) (2.20)
Collecting wsproto>=0.14
  Downloading wsproto-1.1.0-py3-none-any.whl (24 kB)
Requirement already satisfied: cryptography>=1.3.4 in c:\users\administrator\anaconda3\lib\site-packages (from urllib3[secure,socks]~=1.26->selenium) (3.4.7)
Requirement already satisfied: certifi in c:\users\administrator\anaconda3\lib\site-packages (from urllib3[secure,socks]~=1.26->selenium) (2020.12.5)
Requirement already satisfied: pyOpenSSL>=0.14 in c:\users\administrator\anaconda3\lib\site-packages (from urllib3[secure,socks]~=1.26->selenium) (20.0.1)
Requirement already satisfied: PySocks!=1.5.7,<2.0,>=1.5.6 in c:\users\administrator\anaconda3\lib\site-packages (from urllib3[secure,socks]~=1.26->selenium) (1.7.1)
Requirement already satisfied: six>=1.5.2 in c:\users\administrator\anaconda3\lib\site-packages (from pyOpenSSL>=0.14->urllib3[secure,socks]~=1.26->selenium) (1.15.0)
Collecting h11<1,>=0.9.0
  Downloading h11-0.13.0-py3-none-any.whl (58 kB)
Installing collected packages: outcome, h11, wsproto, trio, trio-websocket, selenium
Successfully installed h11-0.13.0 outcome-1.1.0 selenium-4.1.3 trio-0.20.0 trio-websocket-0.9.2 wsproto-1.1.0
</pre>

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib
```


```python
driver = webdriver.Chrome() # 내 코드로 인해 제어되는 크롬창 띄우기

url = "http://naver.com"
driver.get(url) # 네이버에 접속

# 검색창에 명지대 mcc 검색. input text는 검색창 class먕
driver.find_element_by_class_name('input_text').send_keys('명지대 mcc') 

# 검색창에 입력하고 엔터를 쳐라
driver.find_element_by_class_name('input_text').send_keys(Keys.ENTER) 

# 이미지 탭의 검사-우클릭-copy-copy selector 해서 나오는 경로 복사/ -> 이미지탭 클릭
driver.find_element_by_css_selector('#lnb > div.lnb_group > div > ul > li:nth-child(2) > a').click()

# 이미지 클릭 사이에 딜레이를 줘야 함. 2초 딜레이
time.sleep(2)


# _image _listImage 의 공백은 . 점으로 치환해줘야 함. elements는 해당하는 엘리먼트는 다 가져오는 것.
# [0] 은 가장 첫 번째 사진을 클릭하겠다는 것.
driver.find_elements_by_class_name('_image._listImage')[0].click()


# get the image source
img = driver.find_element_by_class_name('_image') # 이미지 class명
src = img.get_attribute('src') # 이미지 태그의 속성

# download the image
# image 폴더를 만들어서(안 만드면 오류남) 그 안에 mcc.jpg로 가져온 이미지 저장
urllib.request.urlretrieve(src, "image/mcc.jpg") 
```

<pre>
<ipython-input-51-c08962e0fcbf>:7: DeprecationWarning: find_element_by_class_name is deprecated. Please use find_element(by=By.CLASS_NAME, value=name) instead
  driver.find_element_by_class_name('input_text').send_keys('명지대 mcc')
<ipython-input-51-c08962e0fcbf>:10: DeprecationWarning: find_element_by_class_name is deprecated. Please use find_element(by=By.CLASS_NAME, value=name) instead
  driver.find_element_by_class_name('input_text').send_keys(Keys.ENTER)
<ipython-input-51-c08962e0fcbf>:13: DeprecationWarning: find_element_by_css_selector is deprecated. Please use find_element(by=By.CSS_SELECTOR, value=css_selector) instead
  driver.find_element_by_css_selector('#lnb > div.lnb_group > div > ul > li:nth-child(2) > a').click()
<ipython-input-51-c08962e0fcbf>:21: DeprecationWarning: find_elements_by_class_name is deprecated. Please use find_elements(by=By.CLASS_NAME, value=name) instead
  driver.find_elements_by_class_name('_image._listImage')[0].click()
<ipython-input-51-c08962e0fcbf>:25: DeprecationWarning: find_element_by_class_name is deprecated. Please use find_element(by=By.CLASS_NAME, value=name) instead
  img = driver.find_element_by_class_name('_image') # 이미지 class명
</pre>
<pre>
('image/mcc.jpg', <http.client.HTTPMessage at 0x14f00647250>)
</pre>
 


# 8주차 단톡 과제(60191315 박온지)

- Selenium을 활용하여 임의의 사이트에서 이미지들을 가져와 저장하시오.

- Ex. SNS, 쇼핑몰, 포털(네이버 제외)

- 검색어를 입력하도록 할 것

- 반복문을 사용하여 이미지를 30장 이상 가져올 것 • 이미지들이 담긴 폴더를 캡쳐하여 제출



```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib

driver = webdriver.Chrome() # 내 코드로 인해 제어되는 크롬창 띄우기

url = "http://google.com/"
driver.get(url) # 네이버에 접속

# 검색창에 귀여운 고양이 검색. gLFyf gsfi는 검색창 class먕
driver.find_element_by_class_name('gLFyf.gsfi').send_keys('귀여운 고양이') 

# 검색창에 입력하고 엔터를 쳐라
driver.find_element_by_class_name('gLFyf.gsfi').send_keys(Keys.ENTER) 

# 이미지 탭의 검사-우클릭-copy-copy selector 해서 나오는 경로 복사/ -> 이미지탭 클릭
driver.find_element_by_css_selector('#hdtb-msb > div:nth-child(1) > div > div:nth-child(2) > a').click()

# 이미지 클릭 사이에 딜레이를 줘야 함. 2초 딜레이
time.sleep(2)

# get the image source
for i in range(35):
    driver.find_elements_by_class_name('rg_i.Q4LuWd')[i].click() # 개별 이미지 클릭
    img = driver.find_element_by_class_name('n3VNCb') #  원본이미지 
    src = img.get_attribute('src') # 이미지 태그의 속성
    # download the image
    # image 폴더를 만들어서(안 만드면 오류남) 그 안에 cat.jpg로 가져온 이미지 저장
    urllib.request.urlretrieve(src, "image/cat{}.jpg".format(i)) 
    
    driver.execute_script("window.scrollTo(0,1)") # 스크롤을 내려주면서 다운로드
```

<pre>
<ipython-input-76-24ed2b67fe13>:12: DeprecationWarning: find_element_by_class_name is deprecated. Please use find_element(by=By.CLASS_NAME, value=name) instead
  driver.find_element_by_class_name('gLFyf.gsfi').send_keys('귀여운 고양이')
<ipython-input-76-24ed2b67fe13>:15: DeprecationWarning: find_element_by_class_name is deprecated. Please use find_element(by=By.CLASS_NAME, value=name) instead
  driver.find_element_by_class_name('gLFyf.gsfi').send_keys(Keys.ENTER)
<ipython-input-76-24ed2b67fe13>:18: DeprecationWarning: find_element_by_css_selector is deprecated. Please use find_element(by=By.CSS_SELECTOR, value=css_selector) instead
  driver.find_element_by_css_selector('#hdtb-msb > div:nth-child(1) > div > div:nth-child(2) > a').click()
<ipython-input-76-24ed2b67fe13>:32: DeprecationWarning: find_elements_by_class_name is deprecated. Please use find_elements(by=By.CLASS_NAME, value=name) instead
  driver.find_elements_by_class_name('rg_i.Q4LuWd')[i].click() # 개별 이미지 클릭
<ipython-input-76-24ed2b67fe13>:33: DeprecationWarning: find_element_by_class_name is deprecated. Please use find_element(by=By.CLASS_NAME, value=name) instead
  img = driver.find_element_by_class_name('n3VNCb') #  원본이미지
</pre>

```python
```
