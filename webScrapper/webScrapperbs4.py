from bs4 import BeautifulSoup
import requests

url = raw_input("Enter the web url to scrap")

r = requests.get("http://" + url)

data = r.text

html_soup = BeautifulSoup('html.parser', "lxml")
uls = html_soup('ul', id = "synopsis-py3881753")
lis = []
for ul in uls:
       for li in ul.findAll('li'):
           if li.find('ul'):
               break
           lis.append(li)

for li in lis:
       print li.text.encode("utf-8")
#textFull = html_soup.findAll('ul', id = "synopsis-py3881753")
#print(textFull)


#https://www.imdb.com/title/tt4154756/plotsummary
