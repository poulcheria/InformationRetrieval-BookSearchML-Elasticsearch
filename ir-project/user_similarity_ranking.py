from elasticsearch import Elasticsearch
from pandas.core import series
import requests
import numpy as np
import pandas as pd
import json 



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)



#----------------------------------------PandasCsvtoDF
usersdf = pd.read_csv("Data/BX-Users.csv") 
booksdf = pd.read_csv("Data/BX-Books.csv") 
ratingsdf = pd.read_csv("Data/BX-Book-Ratings.csv") 
testdf = pd.DataFrame()
#----------------------------------------ElasticSearch
es = Elasticsearch()


checkthis = input("Enter search:")
uid = int(input("unique identifier:"))


url = "http://localhost:9200/books/_search?q="+checkthis+"&pretty=true"
#url2 = "http://localhost:9200/ratings/_search?q="+uid+"&fields='uid'"
url3 = "http://localhost:9200/_cat/indices?v"

res = requests.get(url)
#res2 = requests.get(url2)
res3 = requests.get(url3)

jsonres = json.loads(res.text)
jsonres=jsonres['hits']['hits']

searchdf = pd.DataFrame.from_dict(jsonres)

print(searchdf['_source'])


ratingsdfuser = ratingsdf.where(ratingsdf['uid']==uid)

#print(ratingsdfuser) #Prints all ratings of said user

#print(ratingonbook)

testdf = searchdf["_source"]

json_string= str(testdf[0])
#json_cols = json.loads(json_string)

medlist = [] #list init
userratinglist=[]

for x in range(10):
    stringx = str(testdf[x]['isbn'])
    ratingsdfbooks = ratingsdf.where(ratingsdf['isbn']==stringx).dropna()
    #print(stringx)#prints isbn of said book in string format

    #print(ratingsdfbooks)#prints all ratings on said book
    med=0 #init

    for index, row in ratingsdfbooks.iterrows():
        med = med+int(row['rating'])
    

    medrating = med/len(ratingsdfbooks)
    medlist.append(medrating)
    print(medrating)#prints average rating

    ratingsdf1=ratingsdf[(ratingsdf['isbn']==stringx) & (ratingsdf['uid']==uid)]
    userratinglist.append(ratingsdf1['rating'])
    
    if  not ratingsdf[(ratingsdf['isbn']==stringx) & (ratingsdf['uid']==uid)].empty:
        print(ratingsdf[(ratingsdf['isbn']==stringx) & (ratingsdf['uid']==uid)])

print(json_string)
print(testdf)

searchdf ['mesosoros']=medlist


searchdf['userrating']=userratinglist
#searchdf= searchdf['_source'].apply(json.loads(json_cols))

#print(testdf)
#df = pd.read_json(response)

print(searchdf['mesosoros'].to_string())
print(searchdf['userrating'].to_string())


lastmetric = []
for x in range(10):

    try :
        a = int(searchdf["userrating"][x])
        lastmetric.append(searchdf["mesosoros"][x]+searchdf["userrating"][x]+searchdf["_score"][x])
    except: 
        lastmetric.append(searchdf["mesosoros"][x]+searchdf["_score"][x])


print(lastmetric)

searchdf["endmetric"] = [lastmetric[0],lastmetric[1],lastmetric[2],lastmetric[3],lastmetric[4],lastmetric[5],lastmetric[6],lastmetric[7],lastmetric[8],lastmetric[9]]

# searchdf.sort_values(by=['endmetric'], inplace=True, ascending=False) #sort
searchdf.sort_values(by=['mesosoros'], inplace=True, ascending=False) #sort

print(searchdf)