import pymc3 as pm
import theano.tensor as tt
import numpy as np
import theano

class RandomWalkModel:

    def __init__(self, num_layers, num_nodes, interval, learn_trends=True):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.interval = interval
        self.learn_trends = learn_trends # determines whether we generate the data using the
                                         # learned random walk weights, or if we assume mu=0.

        #cutoff_idx = 1000
        #y_obs = np.ma.MaskedArray(Y, np.arange(N) > cutoff_idx)

        X = np.reshape([0.0], [1000, 1]) # TODO: does the user need to predefine the shape of X and Y?
        Y = np.reshape([0.0], [1000, 1])

        ann_input = theano.shared(X)
        ann_output = theano.shared(Y)

        #n_hidden = [2, 5]
        #interval = 20

        init_1 = np.random.randn(X.shape[1], self.num_nodes).astype(theano.config.floatX)
        # init_2 = np.random.randn(n_hidden[0], n_hidden[1]).astype(theano.config.floatX)
        # init_out = np.random.randn(n_hidden[1]).astype(theano.config.floatX)

        with pm.Model() as self.model:
            self.step_sizes = []
            self.weights_reps = []

            for layer in self.num_layers:

                step_size = pm.HalfNormal('step_size_%s' % layer, sd=np.ones(self.num_nodes),
                                          shape=self.num_nodes)

                weights = pm.GaussianRandomWalk('w%s' % layer, sd=step_size,
                                                 shape=(interval, X.shape[1], self.num_nodes),
                                                 testval=np.tile(init_1, (interval, 1, 1))
                                                 )

                weights_rep = tt.repeat(weights,
                                        ann_input.shape[0] // interval, axis=0)

                self.step_sizes.append(step_size)
                self.weights_reps.append(weights_rep)

            # TODO: mathematical question: does the compounding of random walk variables
            # result in quickly un-inferrable models? (exponential noisiness making it
            # impossible to infer the values themselves)
            weights_1_2 = pm.Normal('w2', mu=0, sd=1.,
                                    shape=(1, n_hidden[0], n_hidden[1]),
                                    testval=init_2)

            weights_1_2_rep = tt.repeat(weights_1_2,
                                        ann_input.shape[0], axis=0)

            weights_2_out = pm.Normal('w3', mu=0, sd=1.,
                                      shape=(1, n_hidden[1]),
                                      testval=init_out)

            weights_2_out_rep = tt.repeat(weights_2_out,
                                          ann_input.shape[0], axis=0)

            # Build neural-network using tanh activation function
            act_1 = tt.tanh(tt.batched_dot(ann_input,
                                           weights_in_1_rep))
            act_2 = tt.tanh(tt.batched_dot(act_1,
                                           weights_1_2_rep))
            act_out = tt.nnet.sigmoid(tt.batched_dot(act_2,
                                                     weights_2_out_rep))

            # Binary classification -> Bernoulli likelihood
            self.out = pm.Bernoulli('out',
                                    act_out,
                                    observed=ann_output)

    def fit(self, X, Y):

        with self.model:
            self.trace = pm.sample(1000, tune=200)

    def generate(self, X, time_steps=1, samples=500):

    def evaluate(self, X, Y):

