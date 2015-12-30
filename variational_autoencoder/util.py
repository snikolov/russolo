import numpy as np
from scipy.io import wavfile

def write_segments(segments, name, rate):
  samples = []
  last = 0
  for segment in segments:
    samples.extend(segment)
  samples = np.array(samples)
  samples = samples / float(np.mean(samples))
  wavfile.write(open('segments_%s.wav' % name, 'w'), rate, samples)
