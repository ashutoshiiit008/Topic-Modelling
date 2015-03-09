import urllib2
import json
import requests
from bs4 import BeautifulSoup
from urllib2 import urlopen
from nltk.corpus import stopwords

#Fetching URL 
page = urllib2.urlopen( 'http://wogma.com/movies/alphabetic/basic/' )
html_doc = page.read()
soup = BeautifulSoup(html_doc)

#Using Soup to find the required part in the web page
review = soup.find_all('div',{'class':"button related_pages review "} )

array = []
#fetching Links in the web page
for each in review:
	links = each.find_all(href = True)
	array.append(str(links[0]).split('"')[1])

print (array)

#Iterating over all the web page
for i , each in enumerate(array):
	fw = open (str(i)+'.txt' , 'w')

	page = urllib2.urlopen( 'http://wogma.com' + each)
	html_doc = page.read()
	soup = BeautifulSoup(html_doc)
	#Finding Review
	review = soup.find('div' ,{'class': 'review large-first-letter'}).text.strip()
	review = ' '.join ([word for word in review.split() if word not in (stopwords.words('english'))])
	r1 = review.encode('utf-8')

	#Storing the review in separate text file
	fw.write(r1)



