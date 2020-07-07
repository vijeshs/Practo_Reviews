#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from gensim import corpora, models, similarities, matutils
import gensim
import dexplot as dxp
from collections import Counter
from matplotlib import rcParams
from gensim import corpora, models, similarities
from itertools import chain
from sklearn.cluster import MeanShift


# In[2]:


pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None) 


# In[3]:


cd desktop


# In[4]:


data=pd.read_csv("processed_reviews_practo.csv") #importing the processed reviews


# In[5]:


data=data.dropna(how='any') #dropping all missing reviews


# In[6]:


len(data)


# In[7]:


data.info()


# In[8]:


data.columns=['reviews', 'recommendation', 'location', 'text_clean', 'sentiment(Recommendation)',
       'reviews_stop_words', 'reviews_stop_words_1']


# ## Distribution of recommendations

# In[9]:


dxp.aggplot(agg="recommendation",data=data,figsize=(4, 4)) 


# ## Distribution of reviews on location

# In[10]:


dxp.aggplot(agg="location", data=data,figsize=(4, 4))


# ## Function to extract sentiment from the reviews

# In[11]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
def sentiment_scores(sentence): 
  
    sid_obj = SentimentIntensityAnalyzer() 
  
    sentiment_dict = sid_obj.polarity_scores(str(sentence)) #finding polarity score
 
    if sentiment_dict['compound'] >= 0.05 :     #score is greater than 0.05 then the tweet is positive
        return "Positive"
  
    elif sentiment_dict['compound'] <= - 0.05 : #score  less than -0.05 is negative
        return "Negative"
  
    else : 
        return "Neutral"


# In[12]:


data['sentiment_vader']=data.apply(lambda row:sentiment_scores(row['reviews']),axis=1) #calling function


# In[115]:


dxp.aggplot(agg="sentiment_vader", data=data,figsize=(4, 4))


# ## Recommendations vs reviews sentiment

# In[13]:


data.groupby('sentiment_vader')['sentiment(Recommendation)']


# In[14]:


pd.crosstab(data['sentiment_vader'],data['sentiment(Recommendation)']).apply(lambda x:x*100/len(data),axis=1)


# In[15]:


positive_reviews=data.loc[data['sentiment(Recommendation)']=='Positive','reviews_stop_words_1'].tolist()#list of positive reviews
negative_reviews=data.loc[data['sentiment(Recommendation)']=='Negative','reviews_stop_words_1'].tolist()#list of negative reviews


# # Recommended Reviews

# ## Function to get LDA topics

# In[16]:


def topics(List):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer.fit(List)
    counts = count_vectorizer.transform(List).transpose()
    corpus = matutils.Sparse2Corpus(counts)
    id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
    lda = models.LdaMulticore(corpus=corpus, num_topics=10, id2word=id2word, passes=1)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=5,passes=10,alpha='auto',per_word_topics=True)
    return lda.print_topics()


# ## Most common words

# In[17]:


# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    
    sns.set_style('whitegrid')
    get_ipython().run_line_magic('matplotlib', 'inline')
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words in not recommended reviews')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(positive_reviews)
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# In[18]:


import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 30
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print(colored("Topics found via LDA:",'green'))
print_topics(lda, count_vectorizer, number_words)

# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components=number_topics)
nmf_model.fit(count_data)
print('\n')
print (colored("Topics found via nmf_Z:",'green'))
print_topics(nmf_model, count_vectorizer, number_words)

print('\n')
print (colored("Topics found via lsi_Z:",'green'))
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=number_topics)
lsi_Z = lsi_model.fit(count_data)
print_topics(lsi_Z, count_vectorizer, number_words)


# In[19]:


topics(positive_reviews) #lda for recommended reviews


# ## K-Means clustering for recommended reviews

# In[20]:


tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), min_df=25, max_df=0.98)
tf_idf_vecs = tf_idf.fit_transform(positive_reviews)

lsa = TruncatedSVD(100, algorithm='arpack')
lsa_vecs = lsa.fit_transform(tf_idf_vecs)
lsa_vecs = Normalizer(copy=False).fit_transform(lsa_vecs)
feature_names = tf_idf.get_feature_names()
lsa_df = pd.DataFrame(lsa.components_.round(5), columns=feature_names)

km = KMeans(n_clusters=5, init='k-means++')
km.fit(lsa_vecs)
clusters = km.predict(lsa_vecs)

km.cluster_centers_.shape

original_space_centroids = lsa.inverse_transform(km.cluster_centers_)
original_space_centroids.shape

order_centroids = original_space_centroids.argsort()[:, ::-1]
order_centroids.shape

for cluster in range(5):
    features = order_centroids[cluster,0:5]
    print('Cluster {}\n'.format(cluster))
    for feature in features:
        print(feature_names[feature])
    print('\n')


# ## Not recommended Reviews

# In[21]:


count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(negative_reviews)
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# In[22]:


# Tweak the two parameters below
number_topics = 5
number_words = 30
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print(colored("Topics found via LDA:",'green'))
print_topics(lda, count_vectorizer, number_words)

# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components=number_topics)
nmf_model.fit(count_data)
print('\n')
print (colored("Topics found via nmf_Z:",'green'))
print_topics(nmf_model, count_vectorizer, number_words)

print('\n')
print (colored("Topics found via lsi_Z:",'green'))
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=number_topics)
lsi_Z = lsi_model.fit(count_data)
print_topics(lsi_Z, count_vectorizer, number_words)


# ## K-Means Clustering for not recommended reviews

# In[23]:


tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), min_df=25, max_df=0.98)
tf_idf_vecs = tf_idf.fit_transform(negative_reviews)

lsa = TruncatedSVD(100, algorithm='arpack')
lsa_vecs = lsa.fit_transform(tf_idf_vecs)
lsa_vecs = Normalizer(copy=False).fit_transform(lsa_vecs)
feature_names = tf_idf.get_feature_names()
lsa_df = pd.DataFrame(lsa.components_.round(5), columns=feature_names)

km = KMeans(n_clusters=5, init='k-means++')
km.fit(lsa_vecs)
clusters = km.predict(lsa_vecs)

km.cluster_centers_.shape

original_space_centroids = lsa.inverse_transform(km.cluster_centers_)
original_space_centroids.shape

order_centroids = original_space_centroids.argsort()[:, ::-1]
order_centroids.shape

for cluster in range(5):
    features = order_centroids[cluster,0:10]
    print('Cluster {}\n'.format(cluster))
    for feature in features:
        print(feature_names[feature])
    print('\n')


# In[24]:


#recommended negative reviews
List_=data.loc[(data['sentiment(Recommendation)']=='Positive')&(data['sentiment_vader']=='Negative'),'reviews_stop_words_1'].tolist()


# In[25]:


#not recommended positive reviews
List=data.loc[(data['sentiment(Recommendation)']=='Negative')&(data['sentiment_vader']=='Positive'),'reviews_stop_words_1'].tolist() 


# In[26]:


len(List)


# ## Not recommended Positive reviews

# In[27]:


tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), min_df=25, max_df=0.98)
tf_idf_vecs = tf_idf.fit_transform(List)

lsa = TruncatedSVD()
lsa_vecs = lsa.fit_transform(tf_idf_vecs)
lsa_vecs = Normalizer(copy=False).fit_transform(lsa_vecs)
feature_names = tf_idf.get_feature_names()
lsa_df = pd.DataFrame(lsa.components_.round(5), columns=feature_names)

km = KMeans(n_clusters=3, init='k-means++')
km.fit(lsa_vecs)
clusters = km.predict(lsa_vecs)

km.cluster_centers_.shape

original_space_centroids = lsa.inverse_transform(km.cluster_centers_)
original_space_centroids.shape

order_centroids = original_space_centroids.argsort()[:, ::-1]
order_centroids.shape





for cluster in range(3):
    features = order_centroids[cluster,0:10]
    print('Cluster {}\n'.format(cluster))
    for feature in features:
        print(feature_names[feature])
    print('\n')


# In[28]:


X_dist = km.transform(lsa_vecs)**2

# do something useful...
import pandas as pd
df = pd.DataFrame(X_dist.sum(axis=1).round(2), columns=['sqdist'])
df['label'] = km.labels_


df['reviews']=List
df.head()


# ## Function to create Bigrams

# In[29]:


def commonwords(review,top):
    reviews=" ".join(review)
    tokenised_reviews=reviews.split(" ")
    
    
    freq_counter=Counter(tokenised_reviews) #counts the occurance
    return freq_counter.most_common(top)


recommended_negative_sentiment=commonwords(List_,30) #top 30 positive reviews
Not_recommended_positive_sentiment=commonwords(List,30) #top 30 negative reviews

def plotCommonWords(reviews,top,title,color="blue",axis=None):
    top_words=commonwords(reviews,top=top)
    data=pd.DataFrame()
    data['words']=[val[0] for val in top_words]
    data['freq']=[val[1] for val in top_words]
    if axis!=None:
        sns.barplot(y='words',x='freq',data=data,color=color,ax=axis).set_title(title+" top "+str(top))
    else:
        sns.barplot(y='words',x='freq',data=data,color=color).set_title(title+" top "+str(top))

def generateNGram(text,n=3):  #bigram
    tokens=text.split(" ")
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return ["_".join(ngram) for ngram in ngrams] #joins two consecutive words

positive_tweets_bigrams=[" ".join(generateNGram(review)) for review in List_] 
negative_tweets_bigrams=[" ".join(generateNGram(review)) for review in List] 


rcParams['figure.figsize'] = 10,10
fig,ax=plt.subplots(1,2)
fig.subplots_adjust(wspace=1) #Adjusts the space between the two plots
plotCommonWords(positive_tweets_bigrams,20,"Recommended negative sentiment",axis=ax[0])

plotCommonWords(negative_tweets_bigrams,20,"Not recommended positive sentiment",color="red",axis=ax[1])


# In[30]:


documents=data['reviews_stop_words_1'] #All reviews


# # Analysis using all the reviews

# ## LDA topic Modelling

# In[31]:


stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

# Create Dictionary.
id2word = corpora.Dictionary(texts)
# Creates the Bag of Word corpus.
mm = [id2word.doc2bow(text) for text in texts]

# Trains the LDA models.
lda = models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=5)

# Prints the topics.
for top in lda.print_topics():
    print(top)
print

# Assigns the topics to the documents in corpus
lda_corpus = lda[mm]


# In[32]:


all_topics_csr = gensim.matutils.corpus2csc(lda_corpus) 
topic_score = all_topics_csr.T.toarray().tolist() #probabilty scores for each cluster


# In[33]:


cluster_1=[]
cluster_2=[]
cluster_3=[]
cluster_4=[]
cluster_5=[]

#splitting the respective cluster scores

for i in topic_score:
    cluster_1.append(i[0])
    cluster_2.append(i[1])
    cluster_3.append(i[2])
    cluster_4.append(i[3])
    cluster_5.append(i[4])
 


# In[34]:


Topics_data = pd.DataFrame()
Topics_data['Text']=documents
Topics_data['cluster_1']=cluster_1
Topics_data['cluster_2']=cluster_2
Topics_data['cluster_3']=cluster_3
Topics_data['cluster_4']=cluster_4
Topics_data['cluster_5']=cluster_5


# In[35]:


Topics_data.head() 


# ## K-Means Clustering

# In[114]:


tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 5), min_df=25, max_df=0.98)
tf_idf_vecs = tf_idf.fit_transform(documents)

lsa = TruncatedSVD(n_components=500)
lsa_vecs = lsa.fit_transform(tf_idf_vecs)
lsa_vecs = Normalizer(copy=False).fit_transform(lsa_vecs)
feature_names = tf_idf.get_feature_names()
lsa_df = pd.DataFrame(lsa.components_.round(5), columns=feature_names)

km = KMeans(n_clusters=5)
km.fit(lsa_vecs)
clusters = km.predict(lsa_vecs)

km.cluster_centers_.shape

original_space_centroids = lsa.inverse_transform(km.cluster_centers_)
original_space_centroids.shape

order_centroids = original_space_centroids.argsort()[:, ::-1]
order_centroids.shape

clusters=[]
topics=[]



for cluster in range(5):
    features = order_centroids[cluster,0:10]
    print('Cluster {}\n'.format(cluster))
    for feature in features:
        print(feature_names[feature])
        topics.append(feature_names[feature])
    clusters.append(topics)
    topics=[]
    print('\n')


# ## determining the cluster distance

# In[37]:


sq_distance=km.transform(lsa_vecs)**2
# do something useful...
import pandas as pd
df = pd.DataFrame(sq_distance.sum(axis=1).round(2), columns=['sqdist'])
df['cluster'] = km.labels_


df['reviews']=data['reviews_stop_words_1']
df['recom']=data['sentiment(Recommendation)']
df['sentiment_vader']=data['sentiment_vader']
df=df.join(pd.DataFrame({'cluster 0': sq_distance[:, 0], 'cluster 1': sq_distance[:, 1],'cluster 2': sq_distance[:, 2],'cluster 3': sq_distance[:, 3],'cluster 4': sq_distance[:, 4]}))


# In[38]:


df.head()


# In[ ]:





# In[ ]:





# ## Tagging the cluster Name

# In[101]:


def cluster(row):
    cluster_no=row['cluster']
    if cluster_no==0:
        return 'nature of treatment'
    if cluster_no==1:
        return 'doctor-interaction'
    if cluster_no==2:
        return 'pre-booking'
    if cluster_no==3:
        return 'post-treatment'
    if cluster_no==4:
        return 'medicine'


# In[102]:


df['topics']=df.apply(lambda row:cluster(row),axis=1)


# In[103]:


df.head()


# In[104]:


df[df.topics=='pre-booking'].head(5)


# In[106]:


df[df.topics=='doctor-interaction'].head(5)


# In[66]:


df[df.topics=='nature of treatment'].head(5)


# In[67]:


df[df.topics=='post-treatment'].head(5)


# In[68]:


df[df.topics=='medicine'].head(5)


# In[108]:


doctor_interaction=df[df['topics']=='doctor-interaction']
medicine=df[df['topics']=='medicine']
nature=df[df['topics']=='nature of treatment']
post_treatment=df[df['topics']=='post-treatment']
pre_booking=df[df['topics']=='pre-booking']


# In[96]:


len(doctor_interaction)/14651


# In[97]:


len(medicine)/14651


# In[98]:


len(nature)/14651


# In[99]:


len(post_treatment)/14651


# In[100]:


len(pre_booking)/14651


# In[95]:


len(df)


# In[ ]:





# In[ ]:





# In[110]:


doctor_interaction.groupby(['topics','sentiment_vader'])['recom'].value_counts().transform(lambda x: x/len(doctor_interaction)*100).unstack()


# In[84]:


medicine.groupby(['topics','sentiment_vader'])['recom'].value_counts().transform(lambda x: x/len(medicine)*100).unstack()


# In[85]:


nature.groupby(['topics','sentiment_vader'])['recom'].value_counts().transform(lambda x: x/len(nature)*100).unstack()


# In[86]:


post_treatment.groupby(['topics','sentiment_vader'])['recom'].value_counts().transform(lambda x: x/len(post_treatment)*100).unstack()


# In[89]:


pre_booking.groupby(['topics','sentiment_vader'])['recom'].value_counts().transform(lambda x: x/len(pre_booking)*100).unstack()


# In[ ]:





# In[111]:


df.groupby(['topics','sentiment_vader'])['recom'].value_counts().transform(lambda x: x/len(df)*100).unstack()


# In[ ]:





# In[70]:


dxp.aggplot(agg="topics", data=df)


# In[71]:


dxp.aggplot(agg="topics", data=df,hue='recom')


# In[72]:


dxp.aggplot(agg="topics", data=df,hue='sentiment_vader')


# In[73]:


positive_not_recommended_reviews=df[(df['sentiment_vader']=='Positive')&(df['recom']=='Negative')]


# In[74]:


dxp.aggplot(agg="topics", data=positive_not_recommended_reviews)


# In[75]:


Negative_recommended_reviews=df[(df['sentiment_vader']=='Negative')&(df['recom']=='Positive')]


# In[76]:


dxp.aggplot(agg="topics", data=Negative_recommended_reviews)


# In[55]:


import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 5
number_words = 30
count_data = count_vectorizer.fit_transform(data['reviews_stop_words_1'])
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print(colored("Topics found via LDA:",'green'))
print_topics(lda, count_vectorizer, number_words)

# Build a Non-Negative Matrix Factorization Model
nmf_model = NMF(n_components=number_topics)
nmf_model.fit(count_data)
print('\n')
print (colored("Topics found via nmf_Z:",'green'))
print_topics(nmf_model, count_vectorizer, number_words)

print('\n')
print (colored("Topics found via lsi_Z:",'green'))
# Build a Latent Semantic Indexing Model
lsi_model = TruncatedSVD(n_components=number_topics)
lsi_Z = lsi_model.fit(count_data)
print_topics(lsi_Z, count_vectorizer, number_words)


# In[ ]:





# In[ ]:





# In[56]:


#from sklearn.cluster import AgglomerativeClustering


# In[57]:


#import numpy as np

#clustering = AgglomerativeClustering().fit(lsa_vecs)


# In[58]:


#from sklearn.metrics.pairwise import cosine_similarity
#dist = 1 - cosine_similarity(lsa_vecs)


# In[59]:


#from scipy.cluster.hierarchy import ward, dendrogram

#linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

#fig, ax = plt.subplots(figsize=(100, 100)) # set size
#ax = dendrogram(linkage_matrix, orientation="right");

#plt.show()

