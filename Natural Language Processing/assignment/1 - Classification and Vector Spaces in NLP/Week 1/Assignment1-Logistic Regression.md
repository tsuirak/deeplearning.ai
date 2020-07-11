# Assignment1:Logistic Regression

## Task:sentiment analysis

## Specifically you will:

* Learn how to extract features for logistic regression given some text
* Implement logistic regression from scratch
* Apply logistic regression on a natural language processing task
* Test using your logistic regression
* Perform error analysis



## 1.文本预处理

##### 1.1处理文本

```python
import nltk
from nltk.corpus import stopwors,twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string

def process_text(tweet):
  
  # 将单个单词处理成为词干
  stemmer=PorterStemmer()
  # 英语中的停顿标点符号
  stopwords_english=stopwords.words('english')
  
  # 使用正则表达式
  
  # 去除股票符号比如 $GE
  tweet = re.sub(r'\$\w*', '', tweet)
  # 去除旧样式转发文字比如 "RT"
  tweet = re.sub(r'^RT[\s]+', '', tweet)
  # 去除超链接
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
  # 去除#
  tweet = re.sub(r'#', '', tweet)
  
  
  tokenizer=TweetTokenizer(preserve_case=True,strip_handles=True,
                           reduce_len=True)
  
  # 将句子处理为单个单词
  tweet_tokens=tokenizer.tokenize(tweet)
  
  tweets_clean=[]
  
  for word in tweet_tokens:
    if word not in stopwords_english and\
    	word not in string.punctuation:
        stem_word=stemmer.stem(word)
        tweets_clean.append(word)
        
  return tweets_clean
```

##### 1.2统计词频

```python
import numpy as np

def build_freqs(tweets,ys):
  # 将label扁平化后转化为列表
  ys_list=np.squeeze(ys).tolist()
  freqs={}
  
  # 统计词频
  for y,tweet in zip(ys_list,tweets):
    for word in process_tweet(twees):
      pair=(word,y)
      if pair in freqs:
        freqs[pair]+=1
      else:
        freqs[pair]=1
  return freqs
```

##### 1.3提取特征

```python
def extract_features(tweet,freqs):
  word_list=process_tweet(tweet)
  x=np.zeros((1,3))
  
  x[0,0]=1
  
  # 正样本 label=1
  # 负样本 label=0
  for word in word_list:
    x[0,1]+=freqs.get((word,1.),0)
    x[0,2]+=freqs.get((word,0.),0)
    
  assert(x.shape==(1,3))
  
  return x
```



## 2.逻辑斯谛回归模型

##### 2.1 sigmoid



```python
import numpy as np

def sigmoid(z):
  return 1/(1+np.exp(-z))
```

##### 2.2 梯度下降

```python
def gradientDescent(x,y,theta,alpha,num_iters):
  m=x.shape[0]
  
  for i in range(num_iters):
    z=np.dot(x,theta)
    h=sigmoid(z)
    J=-1./m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))
    theta=theta-(alpha/m)*np.dot(x.transpose(),(h-y))
    
  J=float(J)
  
  return J,theta
```

##### 2.3 训练模型

```python
X=np.zeros((len(train_x),3))

for i in range(len(train_x)):
  X[i,:]=extract_features(train_x[i],freqs)

Y=train_y


J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
```



## 3.测试模型

##### 3.1预测

```python
def predict_tweet(tweet,freqs,theta):
  
  x=extract_features(tweet,freqs)
  y_pred=sigmoid(np.dot(x,theta))
  
  return y_pred 
```

##### 3.2在测试集上测试模型

```python
def test_logistic_regression(test_x,test_y,freqs,theta):
  y_hat=[]
  
  for tweet in test_x:
    y_pred=predict_tweet(tweet,freqs,theta)
    
    if y_pred>0.5:
      y_hat.append(1)
     else:
      y_hat.append(0)
 	accuracy = (y_hat==np.squeezeu(test_y)).sum().len(test_x)
	return accuracy
```

