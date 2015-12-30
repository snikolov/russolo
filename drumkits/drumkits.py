# Cluster drumkit samples

from math import sqrt
from features import mfcc, logfbank, ssc
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
import numpy as np
import os

plt.ion()

base_dir = '/Users/snikolov/Dropbox/projects/russolo/test_data/household/'
sample_dir = os.path.join(base_dir, 'segments')
#sample_dir = '/Users/snikolov/Dropbox/projects/beat_science/segments/sutro_min1'
IGNORE = "ignore_"

def trim_or_pad(samples, length):
  if len(samples) >= length:
    return samples[:length]
  else:
    return np.hstack([samples, np.zeros(length - len(samples))])

def write_sounds(sounds, name):
  target_dir = os.path.join(base_dir, 'clusters', name)
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)
  samples = []
  last = 0
  for i, (fs, sound) in enumerate(sounds):
    norm = sqrt(np.dot(sound, sound))
    normalized = [s / norm for s in sound]
    wavfile.write(open(os.path.join(target_dir, '%d.wav' % i), 'w'), fs, np.array(normalized, dtype='float32'))
    samples.extend(normalized)
  samples = np.array(samples)
  wavfile.write(open(os.path.join(target_dir, 'cluster_%s.wav' % name), 'w'), fs, samples)

max_len_seconds = 0.5
dirs = [ x for x in os.walk(sample_dir) ]
sounds = []
for d in dirs:
  # Get .wav files
  dir_name = d[0]
  file_names = d[2]
  if IGNORE in dir_name:
    continue 
  for f in file_names:
    if IGNORE in f:
      continue
    print os.path.join(dir_name, f)
    with open(os.path.join(dir_name, f)) as fd:
      if f.endswith('.wav') or f.endswith('.WAV'):
        try:
          fs_and_samples = wavfile.read(fd)
          samples = fs_and_samples[1]
          # Convert to mono      
          fs = fs_and_samples[0]
          if fs == 44100:
            samples = trim_or_pad(samples, max_len_seconds * fs)
            if len(np.shape(samples)) == 2:
              samples = samples[:, 0]
            norm = sqrt(np.dot(samples, samples))
            print 'appending', f
            sounds.append((fs, np.array(samples) / norm))
        except (ValueError, TypeError):
          "Couldn't read wav file"

features = []
for fs, s in sounds:
  mfcc_feat = mfcc(s, fs)
  mfcc_feat = np.reshape(mfcc_feat, (1, np.shape(mfcc_feat)[0] * np.shape(mfcc_feat)[1]))
  ssc_feat = ssc(s, fs)
  ssc_feat = np.reshape(ssc_feat, (1, np.shape(ssc_feat)[0] * np.shape(ssc_feat)[1]))
  lfbank_feat = logfbank(s, fs)
  lfbank_feat = np.reshape(lfbank_feat, (1, np.shape(lfbank_feat)[0] * np.shape(lfbank_feat)[1]))

  #import pdb; pdb.set_trace()

  #ceps, mspec, spec = mfcc(s, fs = fs)
  #ceps = np.reshape(ceps, (1, np.shape(ceps)[0] * np.shape(ceps)[1]))
  
  features.append(np.hstack([mfcc_feat, ssc_feat, lfbank_feat]))
  #features.append(np.hstack([ssc_feat]))

M = np.vstack(features)
print np.shape(M)

pca = PCA(n_components=500)
V = pca.fit_transform(M)
print V
print np.shape(V)
plt.subplot('131')
plt.scatter(V[:,0], V[:,1], alpha = 0.25)
plt.subplot('132')
plt.scatter(V[:,1], V[:,2], alpha = 0.25)
plt.subplot('133')
plt.scatter(V[:,2], V[:,0], alpha = 0.25)
raw_input()

"""
tsne = TSNE(n_components=3, random_state=0)
T = tsne.fit_transform(M)
print np.shape(T)
plt.figure()
plt.subplot('131')
plt.scatter(T[:,0], T[:,1], alpha = 0.1)
plt.subplot('132')
plt.scatter(T[:,1], T[:,2], alpha = 0.1)
plt.subplot('133')
plt.scatter(T[:,2], T[:,0], alpha = 0.1)
raw_input()
"""
n_clusters = 10

agglom = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'cosine', linkage = 'complete')
C = agglom.fit_predict(V)
#kmeans = KMeans(n_clusters = n_clusters)
#C = kmeans.fit_predict(V)
print np.shape(C)
print C

for i in xrange(n_clusters):
  cluster_sounds = []
  for j, c in enumerate(C):
    print j
    if c == i:
      cluster_sounds.append(sounds[j])
  write_sounds(cluster_sounds, str(i))
