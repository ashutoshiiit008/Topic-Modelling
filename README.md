# Topic-Modeling- Proof of Concept
Assigning topics to movie review

Problem Statement : To assign topics ( 15 in this case) to the movie reviews from a site ( Wogmma.com) Extract the top 5 important words from every topics assigned , which represents the topic significantly.

Approach:

STEP 1: Crawled the website (Wogmma.com ) by coding a Python Script . Used BeautifulSoup to extract the text part of the web page and saved it in independent text files.

STEP 2 : Build the document term matrix which  contains 1092 rows and 23743 coloumns , which represents 1092 documents( movie review) and in total 23743 unique words all over.

STEP 3 : Calculated the tf-idf of the document term matrix and set the threshold 0.8 for better accuracy.

STEP 4 : Applied K-Means clustering to the document term matrix where K= 15 , and found the centroid of every cluster
