# Assignment3:Hello Vectors

## task:Predict the Countries from Capitals

## Specifically you will:

- Predict analogies between words.
- Use PCA to reduce the dimensionality of the word embeddings and plot them in two dimensions.
- Compare word embeddings by using a similarity measure (the cosine similarity).
- Understand how these vector space models work.



###### 注:使用word_embeddings_subset.p

```python
import pickle

word_embeddings=pickle.load(open('./word_embeddings_subset.p','rb'))
```

​	

## 1.词嵌入模型

##### 1.1余弦相似度

$$
cos(\theta)=\frac{AB}{||A||\space||B||}
$$



```python
import numpy as np
def cosine_similarity(A,B):
  
  dot=np.dot(A,B)
  normA=np.sqrt((np.dot(A,A)))
  normB=np.sqrt((np.dot(B,B)))
  cos=dot/(norma*normb)
  
  return cos
```

##### 1.2欧式距离

$$
d(A,B)=d(B,A)=\sqrt{\sum_{i=1}^{n}(A_{i}-B_{i})^{2}}
$$



```python
def eucidean(A,B):
  return np.linalg.norm(A-B)
```

##### 1.3计算国家之间的距离

$$
formula:King-Man+Woman=Queen
$$



```python
def get_country(city1,country1,city2,country2,word_embeddings):
  group=set((city1,country1,city2))
  
  # 获得city1的词嵌入值
  city1_emb=word_embeddings[city1]
  
	# 获得country1的词嵌入值
  country1_emb=word_embeddings[country1]
  
  # 获得country1的词嵌入值
  city2_emb=word_embeddings[city2]  
 
	# 公式
  vec=country1_emb-city1_emb+city2_emb
  
  similarity=-1
  country=''
  
  # 遍历词嵌入模型
  for word in embeddings.keys():
    if word not in group:
      word_emb=word_embeddings[word]
      cur_similarity=cosine_similarity(vec,word_emb)
      
      if cur_similarity>similarity:
        similarity=cur_similarity
        country=(word,similarity)
        
  return country
```

## 2.测试模型

##### 2.1预测

```python
def get_accuracy(word_embeddings,data):
  num_correct=0
  
  # 遍历数据
  for i,row in data.iterrows():
    city1=row['city1']
    country1=row['country1']
    city2=row['city2']
    country2=row['country2']
    # 预测数据
    predicted_country2,_=get_country(city1,country1,city2,word_embeddings)
    
    if predicted_country2==country2:
      num_correct+=1
      
  # 计算准确度
	accuracy=num_correct/len(data)
  
  return accuracy
```

##### 2.2主成分分析PCA

###### 将数据降到2维 即可可视化数据

```python
def compute_pca(X,n_components=2):
  # 数据均值处理
  X_demeaned=X-np.mean(X,axis=0)
  
  # 计算协方差矩阵
  covariance_matrix = np.cov(X_demeaned, rowvar=False)

  # 计算协方差矩阵的特征向量/特征值
  eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')

  # 特征值按照升序排序
  idx_sorted = np.argsort(eigen_vals)

  # 从高到低排序
  idx_sorted_decreasing = idx_sorted[::-1]

  # 按照idx对特征值排序
  eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

  # 使用idx_sorted_decreasing索引对特征向量进行排序
  eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]
 
  # 选择前n个特征向量
  # n_components:所需缩放的尺寸
  eigen_vecs_subset = eigen_vecs_sorted[:,0:n_components]

	# 通过乘以特征向量的转置来达到降维的作用
  X_reduced = np.dot(eigen_vecs_subset.transpose(),X_demeaned.transpose()).transpose()
  
  return X_reduced
```

