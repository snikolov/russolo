import cPickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from util import write_segments


print 'Loading data...'
with open('X_500.pkl') as f:
    X = cPickle.load(f)

print 'Loading encoded data...'
with open('Z_1000.pkl') as f:
    Z = np.transpose(cPickle.load(f))


'Clustering with TSNE...'
tsne = TSNE(n_components=3, random_state=0)
T = tsne.fit_transform(Z)

print np.shape(T)
plt.ion()
plt.figure()
plt.subplot('131')
plt.scatter(T[:,0], T[:,1], alpha = 0.1)
plt.subplot('132')
plt.scatter(T[:,1], T[:,2], alpha = 0.1)
plt.subplot('133')
plt.scatter(T[:,2], T[:,0], alpha = 0.1)
#raw_input()

# HACK
rate = 44100

n_clusters = 10
kmeans = KMeans(n_clusters = n_clusters)
C = kmeans.fit_predict(Z)

name = 'X_500'
for i in xrange(n_clusters):
  cluster_segments = []
  for j, c in enumerate(C):
    print j
    if c == i:
      cluster_segments.append(X[j])
  write_segments(cluster_segments, '%s_%d' % (name, i), rate)

