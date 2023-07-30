from elasticsearch import Elasticsearch, helpers
import csv

es=Elasticsearch(host = "localhost", port = 9200)

with open('Data/BX-Users.csv',encoding="utf8") as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, doc_type='users')