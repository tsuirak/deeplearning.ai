# Assignment2:Naive Bayes

## Task:sentiment analysis

## Specifically you will:

* Train a naive bayes model on a sentiment analysis task
* Test using your model
* Compute ratios of positive words to negative words
* Do some error analysis
* Predict on your own tweet



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

def count_tweets(result,tweets,ys):
  # 将label扁平化后转化为列表
  ys_list=np.squeeze(ys).tolist()

  
  # 统计词频
  for y,tweet in zip(ys_list,tweets):
    for word in process_tweet(twees):
      pair=(word,y)
      if pair in freqs:
        result[pair]+=1
      else:
        result[pair]=1
  return result
```

## 2.贝叶斯模型

##### 2.1建立词频率字典

```python
freqs = count_tweets({}, train_x, train_y)
```

##### 2.2查询单词频率

```python
def lookup(freqs,word,labels):
  n=0
  pair=(word,label)
  if pair in freqs:
    n=freqs[pairs]
    
  return n
```



##### 2.3训练贝叶斯模型

```python
def train_naive_bayes(freqs,train_x,train_y):
	loglikelihood={}
  logprior=0
  
  # 计算词库中不同的单词数目
  vocab=set([pair[0] for pair in freqs.keys()])
  V=0
  
  # 计算正样本的单词在词频率中的总数
  # 计算负样本的单词在词频率中的总数
  N_pos=N_neg=0
  
  for pair in freqs.keys():
    if pair[1]>0:
      N_pos+=freqs[pair]
    else:
      N_neg+=freqs[pair]
      

  
  # 计算正样本的总数
  # 计算副样本的总数
  D_pos=(len(list(filter(lambda x:x>0,train_y))))
  D_neg=(len(list(filter(lambda x:x<=0,train_y))))
  
  # 计算先验概率
  logprior=np.log(D_pos)-np.log(D_neg)
  
  for word in vocab:
    
    # 查询该单词在正样本的出现频率
    freq_pos=lookup(freqs,word,1)
    # 查询该单词在负样本的出现频率
    freq_neg=lookup(freqs,word,0)
    
    p_w_pos=(freq_pos+1)/(N_pos+V)
    p_w_neg=(freq_neg+1)/(N_neg+V)
    
    # 计算loglikelihood
    loglikelihood[word]=np.log(p_w_pos/p_w_neg)
    
	return logprior,loglikelihood
```



## 3.测试模型

##### 3.1预测

```python
def naive_bayes_predict(tweet,logprior,loglikelihood):
  word_l=process_tweet(tweet)
  
  p=0
  
  p+=logprior
  
  for word in word_l:
    if word in loglikelihood:
      p+=loglikelihood[word]
  return p
```

##### 3.2在测试集上测试模型

```python
def test_naive_bayes(test_x,test_y,logprior,loglikelihood):
  accuracy=0
  
  y_hats=[]
  
  for tweet in test_x:
    if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
      y_hat_i=1
    else:
      y_hat_i=0
      
    y_hats.append(y_hat_i)
   
	error=np.mean(np.absolute(y_hats-test_y))
     
  accuracy=1-errror
  
  return accuracy
```

## 4.改进模型

Some words have more positive counts than others, and can be considered "more positive". Likewise, some words can be considered more negative than others.

##### 4.1计算比率

```python
def get_ratio(freqs,word):
  pos_neg_ratio={'positive':0,'negative':0,'ratio':0.0}
  
  pos_neg_ratio['positive']=lookup(freqs,word,1)
  
  pos_neg_ratio['negative']=lookup(freqs,word,0)
  
  pos_neg_ratio['ratio']=(pos_neg_ratio['positive'] + 1)/(pos_neg_ratio['negative'] + 1)
  
  return pos_neg_ratio
```

##### 4.2阈值过滤

```python
def get_words_by_threshold(freqs,label,threshold):
  word_list={}
  
  for key in freqs.keys():
    word,_=key
    
    pos_neg_ratio=get_ratio(freqs,word)
    
    if label==1 and pos_neg_ratio['ratio'] >=threshold:
      word_list[word] = pos_neg_ratio
		elif label==0 and pos_neg_ratio['ratio'] <=threshold:
      word_list[word] = pos_neg_ratio
      
  return word_list
```

