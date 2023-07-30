from elasticsearch import Elasticsearch, helpers
import csv

es = Elasticsearch(host = "localhost", port = 9200)


with open("Data/BX-Book-Ratings.csv", encoding="utf8") as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='ratings')