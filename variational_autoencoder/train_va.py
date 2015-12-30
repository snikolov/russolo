import signal
import sys
import time

import cPickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

#sys.path.append('/Users/snikolov/Dropbox/projects/Variational-Autoencoder/Theano')
sys.path.append('/Users/stan/Dropbox/projects/Variational-Autoencoder/Theano')
import VariationalAutoencoder


plt.ion()


def serialize():
    print 'serializing encoder and encodings...'
    Z = encoder.encode(np.transpose(X))
    with open('Z_%d.pkl' % n_iter, 'w') as f:
        cPickle.dump(Z, f)
    with open('encoder_%d.pkl' % n_iter, 'w') as f:
        cPickle.dump(encoder, f)

def quit_handler(signal, frame):
    serialize()
    sys.exit(0)

signal.signal(signal.SIGINT, quit_handler)


print 'Loading data...'
with open('X_small.pkl') as f:
    X = cPickle.load(f)
    (N, dimX) = X.shape

HU_decoder = 200
HU_encoder = 200
dimZ = 5
L = 1
learning_rate = 0.005
batch_size = 20


print 'Initializing encoder...'
encoder = VariationalAutoencoder.VA(
    HU_decoder,
    HU_encoder,
    dimX,
    dimZ,
    batch_size,
    L,
    learning_rate)

encoder.continuous = True

print "Creating Theano functions"
encoder.createGradientFunctions()

print "Initializing weights and biases"
encoder.initParams()

lowerbound = np.array([])
testlowerbound = np.array([])

n_iter = 1
begin = time.time()
plt.figure()
for j in xrange(n_iter):
    encoder.lowerbound = 0
    print 'Iteration:', j
    encoder.iterate(X)
    end = time.time()
    print("Iteration %d, lower bound = %.2f,"
          " time = %.2fs"
          % (j, encoder.lowerbound/N, end - begin))
    begin = end

    '''
    if j % 15 == 0:
        Z = np.transpose(encoder.encode(np.transpose(X)))
        tsne = TSNE(n_components=3, random_state=0)
        T = tsne.fit_transform(Z)
        
        plt.hold(False)
        plt.subplot('131')
        plt.scatter(T[:,0], T[:,1], alpha = 0.1)
        plt.subplot('132')
        plt.scatter(T[:,1], T[:,2], alpha = 0.1)
        plt.subplot('133')
        plt.scatter(T[:,2], T[:,0], alpha = 0.1)
        plt.show()
        raw_input()
    '''
    #if j % 5 == 0:
    #    print "Calculating test lowerbound"
    #    testlowerbound = np.append(testlowerbound,encoder.getLowerBound(x_test))

serialize()
