# Assignment4:Naive Machine Translation and LSH

## task:machine translation system

## Specifically you will:

- The word embeddings data for English and French words
- Translations
- LSH and document search



###### 注:使用en_embeddings.p/fr_embeddings.p

## 1.词嵌入模型

##### 1.1加载模型

```python
en_embeddings_subset=pickle.load(open("en_embeddings.p","rb"))
fr_embeddings_subset=pickle.load(open("fr_embeddings.p","rb"))
```

##### 1.2读入数据并存为字典形式

```python
import pandas as pd

def get_dict(file_name):
  my_file=pd.read_csv(file_name,delimitet=' ')
  
  # 字典 {"英语":"法语"}
  en_to_fr={}
  
  for i in range(len(my_file)):
    en=my_file.iloc[i][0]
    fr=my_file.iloc[i][1]
    en_to_fr[en]=fr
    
 	return en_to_fr
```

##### 1.3生成嵌入模型矩阵和转化矩阵

```python
import numpy as np

def get_matrices(en_to_fr,french_vecs,enlish_vecs):
	"""
    Input:
        en_fr: 英语转法语字典
        french_vecs: 法语对应的词嵌入
        english_vecs: 英语对应的词嵌入
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
  """
  # 存入英语单词向量
  X_l=list()
  # 存入法语单词向量
  Y_l=list()
  
  # 将英语单词存入
  enlish_set=enlish_vecs.keys()
  # 将法语单词存入
  french_set=french_vecs.keys()
  

  french_words=set(en_to_fr.values())
  
  for en_word,fr_word in en_to_fr.items():
    if fr_word in french_set and\
    	en_word in enlish.set:
        en_vec=english_vecs[en_word]
        fr_vec=french_vecs[fr_word]
       	
        X_l.append(en_vec)
        Y_l.append(fr_vec)
        
  # 堆叠
	X=np.vstack(X_l)
  Y=np.vstack(Y_l)
  
  return X,Y
```

##### 1.4计算损失值

$$
formula=\frac{1}{m}||XR-Y||^{2}_{F}
$$



```python
def compute_loss(X,Y,R):
	'''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
  '''
  
  # X的行数
  m=X.shape[0]
  
  # 公式
  diff=np.dot(X,R)-Y	
  sum_diff_squared=np.sum(diff**2)  
  loss=sum_diff_squared/m
  
  return loss
```

##### 1.5计算梯度值

$$
\frac{d}{dR}𝐿(𝑋,𝑌,𝑅)=\frac{d}{dR}\Big(\frac{1}{m}\| X R -Y\|_{F}^{2}\Big) = \frac{2}{m}X^{T} (X R - Y)
$$



```python
def compute_gradient(X,Y,R):
	'''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
  '''
  m=X.shape[0]
  
  gradient=np.dot(X.T,np.dot(X,R)-Y)*(2/m)
```

##### 1.6梯度下降求最优R

$$
R_{n}=R_{old}-\alpha g
$$



```python
def align_embeddings(X,Y,train_steps=100,learning_rate=0.0003):
  np.random.seed(129)
  
  R=np.random.rand(X.shape[1],X.shape[1])
  
  for i in range(train_steps):
    if i%25==0:
      print("loss at iteration {} is {:.4f}".format(i,compute_loss(X,Y,R)))
      
      gradient=compute_loss(X,Y,R)
      
      R-=learning_rate*gradient
      
	return R
```

## 2.测试模型

##### 2.1KNN

```python
def cosine_similarity(A,B):
  dot=np.dot(A,B)
  norma=np.linalg.norm(A)
  normb=np.linalg.norm(B)
  
  return dot/(norma*normb)

def knn(v,candidates,k=1):
  
  similarity_l=[]
  
  for row in candidates:
    cos_similarity=cosine_similarity(v,row)
    similarity_l.append(cos_similarity)
 	
  sorted_ids=np.argsort(similarity_l)
  
  k_idx=sorted_ids[-k:]
  
  return k_idx
```

##### 2.2计算准确性

```python
def test_vocabulary(X,Y,R):
  pred=np.dot(X,R)
  num_correct=0
  
  for i in range(len(pred)):
    pred_idx=knn(pred[i],Y)
    
    if pred_idx==i:
      num_correct+=1
  accuracy=num_correct/len(pred)
  
  return accuracy
```



----





## 1.词袋模型BOW

##### 1.1预处理文本

```python
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
import re

def process_tweet(tweet):
    stemmer=PorterStemmer()
    stopwords_english=stopwords.words('english')
    
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    
    tokenizer=TweetTokenizer(preserve_case=False,strip_handles=True,
                             reduce_len=True)
    
    tweet_tokens=tokenizer.tokenize(tweet)
    
    tweets_clean=[]
    
    for word in tweet_tokens:
        if word not in stopwords_english and\
            word not in string.punctuation:
            stem_word=stemmer.stem(word)
            tweets_clean.append(stem_word)
    
    return tweets_clean
```

##### 1.2文件嵌入

```python
def get_document_embedding(tweet,en_embeddings):
	'''
    Input:
        - tweet: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
  '''
  doc_embedding=np.zeros(300)
  
  # 预处理
  processed_doc=process_tweet(tweet)
  
	# 统计词频率 label=0
  for word in processed_doc:
    doc_embedding+=en_embeddings.get(word,0)
    
	return doc_embedding
```

##### 1.3存入字典

```python
def get_document_vecs(all_docs,en_embeddings):
  '''
    Input:
        - all_docs: list of strings - all tweets in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of tweet embeddings.
        - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
  '''
    
  
  ind2Doc_dict={}
  
  document_vec_l=[]
  
  # 遍历文件
  for i,doc in enumerate(all_docs):
		# 统计词频率
    doc_embedding=get_document_embedding(doc,en_embeddings)
    # 字典记录
    ind2Doc_dict[doc]=doc_embedding
    # 添加入列表
    document_vec_l.append(doc_embedding)
    
 	document_vec_matrix=np.vstack(document_vec_l)
  
  return document_vec_matrix,ind2Doc_dict
```

##### 1.4局部敏感哈希

$$
hash=\sum_{i=0}^{N-1}{2^{i}\space h_{i}}
$$



```python
def hash_value_of_vector(v, planes):
    """Create a hash for a vector; hash_id says which random hash to use.
    Input:
        - v:  vector of tweet. It's dimension is (1, N_DIMS)
        - planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
    Output:
        - res: a number which is used as a hash for your vector

    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # for the set of planes,
    # calculate the dot product between the vector and the matrix containing the planes
    # remember that planes has shape (300, 10)
    # The dot product will have the shape (1,10)
    dot_product = np.dot(v,planes)
    
    # get the sign of the dot product (1,10) shaped vector
    sign_of_dot_product = np.sign(dot_product)
    
    # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    h = sign_of_dot_product>=0

    # remove extra un-used dimensions (convert this from a 2D to a 1D array)
    h = np.squeeze(h)

    # initialize the hash value to 0
    hash_value = 0

    n_planes = planes.shape[1]
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += np.power(2,i)*h[i]
    ### END CODE HERE ###

    # cast hash_value as an integer
    hash_value = int(hash_value)

    return hash_value

```

##### 1.5哈希表

```python
def make_hash_table(vecs, planes):
    """
    Input:
        - vecs: list of vectors to be hashed.
        - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
    Output:
        - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
        - id_table: dictionary - keys are hashes, values are list of vectors id's
                            (it's used to know which tweet corresponds to the hashed vector)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    num_buckets = 2**num_of_planes

    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i:[] for i in range(num_buckets)}

    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i:[] for i in range(num_buckets)}

    # for each vector in 'vecs'
    for i, v in enumerate(vecs):

        # calculate the hash value for the vector
        h = hash_value_of_vector(v,planes)
        #print(h)
        #print('******')
        # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h].append(v)

        # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h].append(i)

    ### END CODE HERE ###

    return hash_table, id_table
```

##### 1.6Approximate K-NN

```python
def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):
    """Search for k-NN using hashes."""
    assert num_universes_to_use <= N_UNIVERSES

    # Vectors that will be checked as p0ossible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        # remove the id of the document that we're searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]
                

                # append the new_id (the index for the document) to the list of ids to consider
                vecs_to_consider_l.append(document_vector_at_i)
                ids_to_consider_l.append(new_id)
                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)

        ### END CODE HERE ###

    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = knn(v, vecs_to_consider_arr, k=k)
    print(nearest_neighbor_idx_l)
    print(ids_to_consider_l)
    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids
```

