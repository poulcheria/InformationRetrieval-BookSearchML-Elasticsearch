from elasticsearch import Elasticsearch, helpers
import csv

es=Elasticsearch(host = "localhost", port = 9200)

# es.indices.delete(index='books', ignore=[400, 404]) #deletes index

with open('Data/BX-books.csv',encoding="utf8") as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, doc_type='books')