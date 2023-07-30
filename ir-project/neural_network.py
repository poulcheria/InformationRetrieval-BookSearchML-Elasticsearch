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
ratingsdf=pd.read_csv("Data/BX-Book-Ratings.csv")

#---------------------------------------Dropping columns we don't need
booksdf.drop(booksdf.iloc[:,1:5],inplace=True,axis=1)
booksdf.drop(booksdf.columns[[2]],axis=1,inplace=True)

#---------------------------------------Merging booksdf with ratingsdf
mergedf=pd.merge(booksdf,ratingsdf,how='left', left_on='isbn',right_on='isbn')
#print( mergedf.iloc[:20])
uid = int(input("unique identifier:"))

for line in mergedf:
      newuserdf=mergedf.where(mergedf['uid']==uid).dropna()
      predictdf=mergedf.where(mergedf['uid']==11676).dropna()

size=len(newuserdf)
print(size)


predictdf=predictdf.iloc[:100]

lines=newuserdf['summary'].values.tolist()
ratings=newuserdf['rating']
prediction=predictdf['summary'].values.tolist()


def wordembeddings(lines):
      review_lines=list()

      for line in lines:
            tokens=word_tokenize(line)
            #convert to lower case
            tokens=[w.lower() for w in tokens]
            #remove punctuation from each word
            table=str.maketrans('','', string.punctuation)
            #table=str.maketrans('''\'t'', string.punctuation)
            stripped=[w.translate(table) for w in tokens]
            #remove remaining tokens that are not alphabetic
            words=[word for word in stripped if word.isalpha()]
            #filter out stop words
            stop_words=set(stopwords.words('english'))
            words=[w for  w in words if not w in stop_words]
            review_lines.append(words)
      num_words=len(review_lines)


      #print(review_lines)
      print("---------------------------------------------------------------------")
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



#---------------------- nn input and output

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


final=wordembeddings(lines)
predict=wordembeddings(prediction)


df_predict=pd.DataFrame(predict)
df_books=pd.DataFrame(predict)


df_in = pd.DataFrame(final)

X=df_in
y=ratings

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)

#--------Building
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))



#-------compile keras model
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['accuracy'])    



model.fit(X_train,y_train, batch_size=16,epochs=1000)      #fit model on the dataset
# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
predictions = model.predict(X_test)
predictions2 = model.predict(df_predict)
# round predictions 
rounded = [round(x[0]) for x in predictions]
rounded2 = [round(x[0]) for x in predictions2]

# print("train ratings:")
# print(y_train)
print("prediction for test ratings:")
print(rounded)
# print("actual ratings:")
# print(y)
print("prediction for books that have not been rated:")
print(rounded2)
