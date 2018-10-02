import pymc3 as pm
import theano.tensor as tt
import numpy as np
import theano

class StochasticNeuralNetwork:

    def __init__(self, nodes_per_layer, interval, output='normal', inference_method='advi', learn_trends=False):
        self.nodes_per_layer = nodes_per_layer
        self.interval = interval
        self.output = output
        self.inference_method = inference_method
        self.learn_trends = learn_trends # determines whether we generate the data using the
                                         # learned random walk weights, or if we assume mu=0.
        self.weight_sd = 10.                # TODO: parameterize this somehow
        self.bias_sd = 10.                  # TODO: parameterize this somehow
        self.activation_fn = pm.math.tanh   # TODO: parameterize this somehow

    def _build_model(self, X, Y):
        #cutoff_idx = 1000
        #y_obs = np.ma.MaskedArray(Y, np.arange(N) > cutoff_idx)

        self.ann_input = theano.shared(X)
        self.ann_output = theano.shared(Y)

        layer_inits = []
        for layer in range(len(self.nodes_per_layer)):
            if layer == 0:
                n_in = X.shape[1]
            else:
                n_in = self.nodes_per_layer[layer - 1]

            layer_inits.append(np.random.randn(n_in, self.nodes_per_layer[layer]).astype(theano.config.floatX))

        init_out = np.random.randn(self.nodes_per_layer[-1]).astype(theano.config.floatX)

        with pm.Model() as self.model:
            self.weights = []

            step_size = pm.HalfNormal('step_size',
                                      sd=np.ones(self.nodes_per_layer[0]) * self.weight_sd,
                                      shape=self.nodes_per_layer[0])

            for layer in range(len(self.nodes_per_layer)):
                # TODO: need to add biases?
                if layer == 0:  # only the first layer will be GaussianRandomWalks
                    weights_intervals = pm.GaussianRandomWalk('w%s' % layer,
                                                    sd=step_size,
                                                    shape=(self.interval, X.shape[1], self.nodes_per_layer[layer]),
                                                    testval=np.tile(layer_inits[layer], (self.interval, 1, 1))
                                                    )

                    weights = tt.repeat(weights_intervals, self.ann_input.shape[0] // self.interval, axis=0)
                else:
                    weights_intervals = pm.Normal('w%s' % layer,
                                          mu=0,
                                          sd=self.weight_sd,
                                          shape=(1, self.nodes_per_layer[layer-1], self.nodes_per_layer[layer]),
                                          testval=layer_inits[layer])

                    weights = tt.repeat(weights_intervals, self.ann_input.shape[0], axis=0)

                self.weights.append(weights)

            # TODO: support multidimensional Y output
            weights_out = pm.Normal('w_out', mu=0, sd=self.weight_sd,
                                    shape=(1, self.nodes_per_layer[-1]),
                                    testval=init_out)

            weights_out_rep = tt.repeat(weights_out,
                                        self.ann_input.shape[0], axis=0)

            # Now assemble the neural network
            self.layers = []
            for layer in range(len(self.nodes_per_layer)):
                input = self.ann_input
                if layer > 0:
                    input = self.layers[layer - 1]

                batched_dot_product = tt.batched_dot(input, self.weights[layer])
                self.layers.append(self.activation_fn(batched_dot_product))# + self.biases[layer]))

            if self.output == 'normal':
                layer_out = tt.batched_dot(self.layers[-1], weights_out_rep)
                bias_out = pm.Normal('bias_out', mu=0.0, sd=self.bias_sd)

                # Regression -> Gaussian likelihood
                pm.Normal('y', mu=layer_out + bias_out, sd=0.1, observed=self.ann_output)
            elif self.output == 'bernoulli':
                layer_out = tt.nnet.sigmoid(tt.batched_dot(self.layers[-1], weights_out_rep))

                # Binary classification -> Bernoulli likelihood
                pm.Bernoulli('y', layer_out, observed=self.ann_output)
            else:
                raise Exception("Unknown output parameter value: %s. Choose among 'normal', 'bernoulli'." % self.output)

    def fit(self, X, Y, samples=500, advi_n=50000, advi_n_mc=1, advi_obj_optimizer=pm.adam(learning_rate=.1)):

        self.num_samples = samples

        self._build_model(X, Y)

        with self.model:
            if self.inference_method == 'advi':
                mean_field = pm.fit(n=advi_n,
                                    method='advi',
                                    obj_n_mc=advi_n_mc,
                                    obj_optimizer=advi_obj_optimizer)       # TODO: how to determine hyperparameters?

                self.trace = mean_field.sample(draws=samples)
            elif self.inference_method == 'mcmc':
                self.trace = pm.sample(samples, tune=samples)
            else:
                raise Exception("Unknown output parameter value: %s. Choose among 'normal', 'bernoulli'." % self.output)

    def predict(self, X):

        # for each of the num_samples parameter values sampled above, sample 500 times the expected y value.
        S = 100
        samples = pm.sample_ppc(self.trace, model=self.model, size=S)
        y_preds = samples['y']

        # get the average, since we're interested in plotting the expectation.
        y_preds = np.mean(y_preds, axis=1)
        y_preds = np.mean(y_preds, axis=0)

        return y_preds

    #def generate(self, X, time_steps=1, samples=500):

    def RMSD(self, X, Y):
        y_preds = self.predict(X)

        deviations = y_preds - Y

        return np.sqrt(np.mean(deviations ** 2.0))
