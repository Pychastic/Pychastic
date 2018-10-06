import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

import BayesianNeuralNetwork as BNN
import pymc3 as pm

N = 5000
X = np.reshape(np.random.normal(0.0, 1.0, N), [-1, 1])
Y = (X ** 2.0) + 0.9

nn = BNN.BayesianNeuralNetwork([2, 5], output='normal', inference_method='advi')
nn.fit(X, Y, samples=250, advi_n=15000, advi_obj_optimizer=pm.adam(learning_rate=.01))
y_preds = nn.predict(X)

rmsd = nn.RMSD(X, Y)

print "Root Mean Square deviation: %s" % rmsd

plt.scatter(X, Y)
plt.scatter(X, y_preds, alpha=0.1)

plt.show()
