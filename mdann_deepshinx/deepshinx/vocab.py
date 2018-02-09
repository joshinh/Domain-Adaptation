'''Vocabulary for the dataset'''
import numpy as np

VOCAB = np.asarray(
    ['<eps>', '<s>', '</s>'] + list(' \'.-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + ['<backoff>'])
VOCAB_TO_INT = {}

for ch in VOCAB:
    VOCAB_TO_INT[ch] = len(VOCAB_TO_INT)

# backoff is not in vocabulary
VOCAB_SIZE = len(VOCAB) - 1
