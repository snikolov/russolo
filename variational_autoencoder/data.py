import cPickle
import matplotlib.pyplot as plt
import numpy as np
import os

from aubio import source, onset, pitch
from scipy.io import wavfile

base_dir = '/Users/snikolov/Dropbox/projects/sounds'
sample_dir = os.path.join(base_dir, 'segments')
IGNORE = "ignore_"

FFT_WIN_SIZE = 512
HOP_SIZE = FFT_WIN_SIZE / 2

def trim_or_pad(samples, length):
  if len(samples) >= length:
    return samples[:length]
  else:
    return np.hstack([samples, np.zeros(length - len(samples))])

def generate_data(
    path_in,
    path_out,
    n_samples=22050,
    n_examples_from_each=5,
    n_examples=1000
):
    X = None
    for d in os.walk(path_in):
        dir_name = d[0]
        file_names = d[2]
        for f in file_names:
            file_path = os.path.join(dir_name, f)
            print file_path
            if f.lower().endswith('.wav'):
                Xf = generate_data_for_file(file_path, n_samples)
                np.random.shuffle(Xf)
                Xf = Xf[:n_examples_from_each]
                if X is None:
                    X = Xf
                else:
                    X = np.vstack([X, Xf])
                print X.shape
                if len(X) > n_examples:
                    with open(path_out, 'w') as out:
                        cPickle.dump(X[:n_examples], out)
                        return

def generate_data_for_file(path, n_samples):
    sample_rate = 0
    aubio_source = source(path, sample_rate, HOP_SIZE)
    sample_rate = aubio_source.samplerate
    aubio_onset = onset('default', FFT_WIN_SIZE, HOP_SIZE, sample_rate)

    onsets = []
    while True:
        chunk, read = aubio_source()
        if aubio_onset(chunk):
            onsets.append(aubio_onset.get_last())
            
        if read < HOP_SIZE: break

    samples = wavfile.read(path)[1]
    slices = np.split(samples, onsets)
    snippets = []
    for slice in slices:
        if len(slice) == 0:
            continue
        mono = 0.5 * (slice[:,0] + slice[:,1])
        mono = mono / np.sqrt(np.dot(mono, mono))
        fixed_len = trim_or_pad(mono, n_samples)
        snippets.append(fixed_len)
    return np.vstack(snippets)


if __name__ == '__main__':
    plt.ion()
    path_in = '/Users/snikolov/Dropbox/sound/'
    path_out = 'X_500.pkl'
    generate_data(path_in, path_out, n_samples=22050, n_examples=500)
    import ipdb; ipdb.set_trace()
