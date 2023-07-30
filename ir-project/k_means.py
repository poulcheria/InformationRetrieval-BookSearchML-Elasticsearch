from cgi import print_directory
from dataclasses import field
from gensim.models.keyedvectors import Vocab
from nltk.tokenize.api import TokenizerI
from nltk.util import pad_sequence
from numpy.core.defchararray import array
from numpy.lib.function_base import append
from numpy.lib.shape_base import column_stack
from pandas.core import series
import requests
import numpy as np
import pandas as pd
import json 
import nltk
#nltk.download()
#import nltk
#nltk.download('punkt')
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from tensorflow.python.keras.backend import constant
from tensorflow.python.keras.layers import embeddings 
warnings.filterwarnings(action = 'ignore')
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import numpy as np

from sklearn.cluster import KMeans



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)


#----------------------------------------PandasCsvtoDF
booksdf = pd.read_csv("Data/BX-Books.csv") 
ratingsdf = pd.read_csv("Data/BX-Book-Ratings.csv") 
usersdf = pd.read_csv("Data/BX-Users.csv") 


#---------------------------------------Dropping columns we don't need
booksdf.drop(['book_title','book_author','year_of_publication','publisher'],inplace=True,axis=1)
booksdf=pd.merge(booksdf,ratingsdf,how='left', left_on='isbn',right_on='isbn')
booksdf=pd.merge(booksdf,usersdf,how='left', left_on='uid',right_on='uid')

booksdf=booksdf.iloc[:13800]
booksvectors = booksdf['summary'].values.tolist()

def wordembeddings(lines):
      review_lines=list()

      for line in lines:
            tokens=word_tokenize(line)
            #convert to lower case
            tokens=[w.lower() for w in tokens]
            #remove punctuation from each word
            table=str.maketrans('','', string.punctuation)
            stripped=[w.translate(table) for w in tokens]
            #remove remaining tokens that are not alphabetic
            words=[word for word in stripped if word.isalpha()]
            #filter out stop words
            stop_words=set(stopwords.words('english'))
            words=[w for  w in words if not w in stop_words]
            review_lines.append(words)
      num_words=len(review_lines)


      #print(review_lines)
    
      #--------------------------------------Word embendings word2vec CBOW model
      model1 = Word2Vec(review_lines,vector_size=10, window=5,workers=3, min_count=1)

      word_em=[]
      summary_sum=[]
      arr=[]
      templist=[]
      newlist=[]
      final=[]
      from numpy import sum
      #----------------------getting the sum of words word embedings for each summary
      # print(review_lines)
      for line in review_lines:
            for word in line:
                  summary_sum.append(model1.wv[word])
                  
            templist.append(summary_sum)
            temp=np.array(templist,dtype="float32")
            newlist.append(temp)
            templist.clear()
            summary_sum.clear()
            del temp

      count=0
      # print(len(newlist))
      #column_sums=np.zeros((num_words,10))
      #temp=np.zeros((1,10))
      for line in newlist:
            temp=np.zeros((len(line),10))
            #print(temp)
            for word in line:
                  #print(word)
                  temp=np.sum(word,axis=0)
            #temp=column_sums+word
            final.append(temp)
      #print(templist[0:2])
      print(len(final))
      #final=np.array(final,dtype="float32")
      return(final)


bookz = wordembeddings(booksvectors)


#----------------------Kmeans on dataset
df_m00n=[]
k=0
for line in bookz:
      # print(line)
 
      print(line/np.linalg.norm(line, ord = 2))
      if line.all(0)!=0.0:
            print(line)
            df_m00n.append(line/np.linalg.norm(line, ord = 2))
      print(k)
      k=k+1
#print(df_m00n)
kmeans = KMeans(n_clusters=10).fit(df_m00n)
centroids = kmeans.cluster_centers_
print ("\n\n------Centroids--------\n")
print(centroids)

booksdf['cluster'] = kmeans.labels_


#print(booksdf)
import math
for i in range(10):
      newdf=booksdf.loc[booksdf['cluster']==i]
      print('cluster:',i)
      #print(newdf)
      # newdf=pd.DataFrame()
      Totalage=newdf['age'].sum(skipna=True)
      cnt=newdf['age']
      cnt=cnt.count()
      #print(cnt)
      #print(Totalage)
      mesosoroshlikia=Totalage/cnt
      if(math.isnan(mesosoroshlikia)==False):
            mesosoroshlikia=round(mesosoroshlikia)
      print('Mesos oros hlikias:',mesosoroshlikia)
      locationdf=newdf['location'].mode()
      #locationdf=locationdf.dropna()
      print('Most frequent Location:')
      print(locationdf)
      Totalrating=newdf['rating'].sum(skipna=True)
      ratingcnt=newdf['rating']
      ratingcnt=ratingcnt.count()
      MOrating=Totalrating/ratingcnt
      MOrating=round(MOrating,2)
      print('mesos oros vathmologias:')
      print(MOrating)
      mostfreq=newdf['category'].mode()
      print('Most frequent category:')
      print(mostfreq)
      print('=================================')
      # del tempdf
      # del newdf
      # del cnt
