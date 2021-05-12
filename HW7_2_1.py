import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import MultinomialHMM

sequence = np.concatenate(np.load('sequences.npy')) - 1
model = MultinomialHMM(n_components=2, tol=0.001)
model.fit(np.atleast_2d(sequence).T, [30 for i in range(200)])
game = np.array([[3, 2, 1, 3, 4, 5, 6, 5, 1, 4, 2, 6, 6, 2, 6]]) - 1
print(model.emissionprob_)
print(model.startprob_)
print(model.transmat_)
print(model.decode(game.T))
