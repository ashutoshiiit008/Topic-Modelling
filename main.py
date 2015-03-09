import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import json


from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from pprint import pprint
from sklearn import metrics
# import sys
from sklearn.cluster import KMeans, MiniBatchKMeans

reload(sys)
sys.setdefaultencoding("utf-8")

op = OptionParser()

op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

data_train = load_files('train')
import os
all_data = []
for fil in os.listdir('train/'):
  fill = open('train/'+fil).read().strip()
  all_data.append(fill)

print all_data

true_k = 15 #Assigning 15 topics to the dataset

if opts.use_hashing:
	vectorizer = HashingVectorizer(stop_words='english', non_negative=True,n_features=opts.n_features)
	X_train = vectorizer.transform(all_data)
else:
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8,stop_words='english') #Tfidf calculation
	X_train = vectorizer.fit_transform(all_data)

print (X_train).shape

X = vectorizer.fit_transform(all_data)

print X 

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)


#Printing the clusters formed after K-Means
if not (opts.n_components or opts.use_hashing):
    print("Top terms per cluster:")
    order_centroids = X.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
      print("Cluster" , i)
      for ind in order_centroids[i, :10]:
        print( terms[ind])
