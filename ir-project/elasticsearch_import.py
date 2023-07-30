import requests

checkthis=input("Enter search:")

url = "http://localhost:9200/books/_search?q="+checkthis+"&pretty=true"
res = requests.get(url)

print(res.text)