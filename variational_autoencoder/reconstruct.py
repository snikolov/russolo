import cPickle
import matplotlib.pyplot as plt
import numpy as np

from util import write_segments

import sys
sys.path.append('/Users/stan/Dropbox/projects/Variational-Autoencoder/Theano')


print 'Loading data...'
with open('X_small.pkl') as f:
    X = cPickle.load(f)

print 'Loading encoder...'
with open('encoder_1.pkl', 'r') as f:
    encoder = cPickle.load(f)

print 'Loading encoded data...'
with open('Z_1.pkl') as f:
    Z = cPickle.load(f)

print 'Z', Z.shape
original_segments = X[:50]
reconstructed_segments = np.transpose(encoder.decode(Z[:50]))
print 'reconstructed', reconstructed_segments.shape
write_segments(original_segments, 'original')
write_segments(reconstructed_segments, 'reconstructed')
