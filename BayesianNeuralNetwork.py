import pymc3 as pm
import numpy as np
import theano
import matplotlib.pyplot as plt

class BayesianNeuralNetwork:

    def __init__(self, nodes_per_layer, output='normal', inference_method='advi'):
        self.nodes_per_layer = nodes_per_layer
        self.output = output
        self.inference_method = inference_method
        self.weight_sd = 10.                # TODO: parameterize this somehow
        self.bias_sd = 10.                  # TODO: parameterize this somehow
        self.activation_fn = pm.math.tanh   # TODO: parameterize this somehow

    def _build_model(self, X, Y):
        self.ann_input = theano.shared(X)
        self.ann_output = theano.shared(Y)

        layer_inits = []
        for layer in range(len(self.nodes_per_layer)):
            if layer == 0:
                n_in = X.shape[1]
            else:
                n_in = self.nodes_per_layer[layer - 1]

            layer_inits.append(np.random.randn(n_in, self.nodes_per_layer[layer]).astype(theano.config.floatX))

        init_out = np.random.randn(self.nodes_per_layer[-1], Y.shape[1]).astype(theano.config.floatX)

        with pm.Model() as self.model:
            self.weights = []
            self.biases = []

            for layer in range(len(self.nodes_per_layer)):
                if layer == 0:
                    first_dim = X.shape[1]
                else:
                    first_dim = self.nodes_per_layer[layer-1]

                self.biases.append(pm.Normal('bias%s' % layer, mu=0.0, sd=self.bias_sd))
                self.weights.append(pm.Normal('layer%s' % layer,
                                              mu=0,
                                              sd=self.weight_sd,
                                              shape=(first_dim, self.nodes_per_layer[layer]),
                                              testval=layer_inits[layer]))

            # output layer
            weights_out = pm.Normal('out',
                                    mu=0,
                                    sd=self.weight_sd,
                                    shape=(self.nodes_per_layer[-1], Y.shape[1]),
                                    testval=init_out)

            self.num_outputs = Y.shape[1]

            # Build neural-network using tanh activation function
            self.layers = []
            for layer in range(len(self.nodes_per_layer)):
                input = self.ann_input
                if layer > 0:
                    input = self.layers[layer - 1]

                dot_product = pm.math.dot(input, self.weights[layer])
                self.layers.append(self.activation_fn(dot_product + self.biases[layer]))

            if self.output == 'normal':
                layer_out = pm.math.dot(self.layers[-1], weights_out)
                bias_out = pm.Normal('bias_out', mu=0.0, sd=self.bias_sd)

                # Regression -> Gaussian likelihood
                pm.Normal('y', mu=layer_out + bias_out, observed=self.ann_output)
            elif self.output == 'bernoulli':
                layer_out = pm.math.sigmoid(pm.math.dot(self.layers[-1], weights_out))
                bias_out = pm.Normal('bias_out', mu=0.0, sd=self.bias_sd)

                # Binary classification -> Bernoulli likelihood
                pm.Bernoulli('y', layer_out + bias_out, observed=self.ann_output)
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
        self.ann_input.set_value(X)
        self.ann_output.set_value(X) # TODO: for some reason I need to set this with something of the same length as Y (and possibly same dimensionality??)

        S = 100
        preds = pm.sample_ppc(self.trace, model=self.model, size=S)

        y_preds = np.reshape(preds['y'], [self.num_samples, S, -1])

        # get the average, since we're interested in plotting the expectation.
        y_preds = np.mean(y_preds, axis=1)
        y_preds = np.mean(y_preds, axis=0)

        return np.reshape(y_preds, [-1, self.num_outputs])

    # TODO: implement this: similar to predict, but don't take the expectation of the output, allow the full uncertainty/variation
    #def generate(self, samples=500):

    def RMSD(self, X, Y):
        y_preds = self.predict(X)

        deviations = y_preds - Y

        return np.sqrt(np.mean(deviations ** 2.0))
