import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an
import seaborn as sns
sns.set_style('white')
from sklearn.preprocessing import scale
from sklearn.datasets import make_moons
import sys
sys.path.append("..")

import StochasticNeuralNetwork as SNN

X, Y = make_moons(noise=0.2, random_state=0, n_samples=4798)
X = scale(X)

colors = Y.astype(str)
colors[Y == 0] = 'r'
colors[Y == 1] = 'b'

interval = 20
subsample = X.shape[0] // interval
chunk = np.arange(0, X.shape[0]+1, subsample)
degs = np.linspace(0, 360, len(chunk))

sep_lines = []

for ii, (i, j, deg) in enumerate(list(zip(np.roll(chunk, 1), chunk, degs))[1:]):
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s], [s, c]])

    X[i:j, :] = X[i:j, :].dot(R)

print "Animating generated data..."
fig, ax = plt.subplots()
ims = []
for i in np.arange(0, len(X), 10):
    ims.append((ax.scatter(X[:i, 0], X[:i, 1], color=colors[:i]),))

ax.set(xlabel='X1', ylabel='X2')
anim = an.ArtistAnimation(fig, ims,
                          interval=10,
                          blit=True)

plt.show()

np.random.seed(123)

nn = SNN.StochasticNeuralNetwork([2, 2], 20, output='bernoulli')
nn.fit(X, Y, samples=500, advi_n=10000)
y_preds = nn.predict(X)

print "X shape = ", X.shape
print "Y shape = ", Y.shape
print "y_preds shape = ", y_preds.shape

# show the time-varying effect
fig, axarr = plt.subplots(1, 5, sharey=True, sharex=True)

for i in range(len(axarr)):
    from_idx = 1000 * i
    to_idx = 1000 * (i + 1)
    if to_idx > y_preds.shape[0]:
        to_idx = y_preds.shape[0]

    blue_points_y = []
    red_points_y = []
    blue_points_pred = []
    red_points_pred = []
    for j in range(from_idx, to_idx):
        if Y[j] > 0.5:
            red_points_y.append(X[j])
        else:
            blue_points_y.append(X[j])

        if y_preds[j] > 0.5:
            red_points_pred.append(X[j])
        else:
            blue_points_pred.append(X[j])

    blue_points_y = np.array(blue_points_y)
    red_points_y = np.array(red_points_y)
    blue_points_pred = np.array(blue_points_pred)
    red_points_pred = np.array(red_points_pred)

    axarr[i].scatter(blue_points_y[:, 0], blue_points_y[:, 1], color='blue')
    axarr[i].scatter(red_points_y[:, 0], red_points_y[:, 1], color='red')
    axarr[i].scatter(blue_points_pred[:, 0], blue_points_pred[:, 1], color='purple', alpha=0.1)
    axarr[i].scatter(red_points_pred[:, 0], red_points_pred[:, 1], color='yellow', alpha=0.1)

    axarr[i].set_title("t range %s to %s" % (from_idx, to_idx))

plt.show()