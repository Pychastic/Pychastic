import pymc3 as pm
import numpy as np
import theano

class BayesianNeuralNetwork:

    def __init__(self, num_layers, num_nodes, num_inputs=1, num_outputs=1, output='normal'):
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        X = np.zeros([1000, num_inputs])   # TODO: does the user need to predefine the shape of X and Y?
        Y = np.zeros([1000, num_outputs])

        print "X shape = ", X.shape

        self.ann_input = theano.shared(X)
        self.ann_output = theano.shared(Y)

        layer_inits = []
        for layer in range(self.num_layers):
            n_in = self.num_nodes
            if layer == 0:
                n_in = X.shape[1]

                print "n_in[0]", n_in

            layer_inits.append(np.random.randn(n_in, self.num_nodes).astype(theano.config.floatX))
        init_out = np.random.randn(self.num_nodes, num_outputs).astype(theano.config.floatX)

        with pm.Model() as self.model:
            self.weights = []
            self.biases = []

            for layer in range(self.num_layers):
                first_dim = self.num_nodes
                if layer == 0:
                    first_dim = X.shape[1]

                self.biases.append(pm.Normal('bias%s' % layer, mu=0.0, sd=1.))
                self.weights.append(pm.Normal('layer%s' % layer,
                                              mu=0,
                                              sd=1., # TODO: ???
                                              shape=(first_dim, self.num_nodes),
                                              testval=layer_inits[layer]))

            # output layer
            weights_out = pm.Normal('out',
                                    mu=0,
                                    sd=1.,    # TODO: ???
                                    shape=(self.num_nodes, num_outputs),
                                    testval=init_out)

            # Build neural-network using tanh activation function
            self.layers = []
            for layer in range(self.num_layers):
                input = self.ann_input
                if layer > 0:
                    input = self.layers[layer-1]

                # TODO: make it possible to customize the activation function
                # TODO: BUG: ValueError: Shape mismatch: batch sizes unequal. x.shape is (1000, 1, 1), y.shape is (1, 5, 1).
                print "layer = ", layer
                dot_product = pm.math.dot(input, self.weights[layer])
                self.layers.append(pm.math.tanh(dot_product + self.biases[layer]))

            if output == 'normal':
                layer_out = pm.math.dot(self.layers[-1], weights_out)
                bias_out = pm.Normal('bias_out', mu=0.0, sd=1.)

                # Regression -> Gaussian likelihood
                pm.Normal('y', mu=layer_out + bias_out, observed=self.ann_output)
            elif output == 'bernoulli':
                layer_out = pm.math.sigmoid(pm.math.dot(self.layers[-1], weights_out))
                bias_out = pm.Normal('bias_out', mu=0.0, sd=1.)

                # Binary classification -> Bernoulli likelihood
                pm.Bernoulli('y', layer_out + bias_out, observed=self.ann_output)
            else:
                raise Exception("Unknown output parameter value: %s. Choose among 'normal', 'bernoulli'." % output)

    def fit(self, X, Y, samples=500):
        self.num_samples = samples
        self.ann_input.set_value(X)
        self.ann_output.set_value(Y)
        with self.model:
            self.trace = pm.sample(samples, tune=samples)

    def predict(self, X):
        samples = pm.sample_ppc(self.trace, model=self.model, size=10)
        y_preds = np.reshape(samples['y'], [self.num_samples, 10, X.shape[0]])

        # get the average, since we're interested in plotting the expectation.
        y_preds = np.mean(y_preds, axis=1)
        y_preds = np.mean(y_preds, axis=0)

        return y_preds

    #def generate(self, samples=500):

    def RMSD(self, X, Y):
        y_preds = self.predict(X)

        return np.sqrt(np.mean((y_preds - Y) ** 2.0))
