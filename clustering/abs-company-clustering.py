# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer

# df = pd.read_csv(r'corpus/abs-company.csv')
# companies = np.array(df['company'])
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(companies)


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import pandas as pd


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

df = pd.read_csv(r'corpus/abs-company.csv')
companies = np.array(df['company'])
# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

true_k = 5

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(companies)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()


# #############################################################################
# Do the actual clustering
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=True)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
df['cat'] = km.labels_.tolist()
df.to_csv('a.csv')

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))

# print()


# if not opts.use_hashing:
#     print("Top terms per cluster:")

#     if opts.n_components:
#         original_space_centroids = svd.inverse_transform(km.cluster_centers_)
#         order_centroids = original_space_centroids.argsort()[:, ::-1]
#     else:
#         order_centroids = km.cluster_centers_.argsort()[:, ::-1]

#     terms = vectorizer.get_feature_names()
#     for i in range(true_k):
#         print("Cluster %d:" % i, end='')
#         for ind in order_centroids[i, :10]:
#             print(' %s' % terms[ind], end='')
#         print()