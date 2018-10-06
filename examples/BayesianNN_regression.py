import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

import BayesianNeuralNetwork as BNN
import pymc3 as pm

N = 5000
X = np.reshape(np.random.normal(0.0, 1.0, N), [-1, 1])
Y = (X ** 2.0) + 0.9

nn = BNN.BayesianNeuralNetwork([5, 5], output='normal', inference_method='advi')
nn.fit(X, Y, samples=250, advi_n=15000, advi_obj_optimizer=pm.adam(learning_rate=.01))
y_preds = nn.predict(X)

rmsd = nn.RMSD(X, Y)

print "Root Mean Square deviation: %s" % rmsd

# get the 10th and 90th percentiles of certainty
y_generated = nn.generate(X, samples=250)
pct10 = np.percentile(y_generated, q=10, axis=0)
pct90 = np.percentile(y_generated, q=90, axis=0)

plt.scatter(X, Y)
plt.scatter(X, y_preds, alpha=0.1)

plt.scatter(X, pct10, s=1, alpha=0.1, color='red')
plt.scatter(X, pct90, s=1, alpha=0.1, color='red')
plt.show()

