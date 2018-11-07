import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pymc3.distributions.timeseries as ts
import theano.tensor as tt

class MDHAssetModel:

    def __init__(self, num_lags=0, w=[0.7, 0.3], intercept=-5.35, theta=0.8, sigma=0.01, rhos=[]):
        self.num_lags = num_lags
        self.estimated_intercept = intercept
        self.estimated_rhos = rhos
        self.estimated_w = w
        self.estimated_theta = theta
        self.estimated_sigma = sigma
        self.model = None
        self.trace = None

    def loo(self):

        return pm.loo(self.trace, self.model)

    def display_params(self):

        print "component weights = ", self.estimated_w
        print "volatility long-term equilibrium = ", self.estimated_intercept
        print "volatility speed of mean reversion = ", self.estimated_theta
        print "volatility standard deviation of innovations = ", self.estimated_sigma

        if self.num_lags > 0:
            print "auto-correlation coefficients of the first %s lags = %s" % (self.num_lags, self.estimated_rhos)

    def fit(self, prices):

        logreturns = np.log(prices[1:]) - np.log(prices[:-1])
        self.fit_logreturns(logreturns)

    def _lags(self, data):

        lagged_data = []
        for l in range(self.num_lags + 1):
            if l == 0:
                lagged_data.append(np.array(data[self.num_lags:]))
            else:
                lagged_data.append(np.array(data[self.num_lags - l:-l]))

        return np.array(lagged_data)

    # TODO: 1) try a log-OU model instead of just OU for the stochastic volatility
    # TODO: 2) try a flexible inner standard deviation, rather than fixed at 1e-100
    def fit_logreturns(self, data):

        def likelihood(x):

            def _normal(x, sigma):  # assumes a mu of 0
                return pm.Normal.dist(mu=0., sd=sigma).logp(x)

            nu_t = pm.math.dot(rhos, x[1:])
            err = tt.reshape(x[0] - nu_t, [-1])

            logps = (w[0] * pm.math.exp(_normal(err, pm.math.exp(s)))) + (w[1] * pm.math.exp(_normal(x[0], float(1e-100))))

            return pm.math.log(logps)

        with pm.Model() as self.model:
            W = np.array([1., 1.])

            w = pm.Dirichlet('w', W)

            intercept = pm.Normal('intercept', mu=-5, sd=5., testval=-5.)
            theta = pm.Uniform('theta', lower=0.001, upper=1.)
            sigma = pm.Uniform('sigma', lower=0.001, upper=10.)

            rhos = pm.Uniform('rhos', lower=-1., upper=1., shape=self.num_lags)

            sde = lambda x, theta, mu: (theta * (mu-x), sigma)
            s = ts.EulerMaruyama('path',
                                 1.0,
                                 sde,
                                 [theta, intercept],
                                 shape=len(data) - self.num_lags,
                                 testval=np.ones_like(data[self.num_lags:]))

            lagged_data = self._lags(data)

            pm.DensityDist('obs', likelihood, observed=lagged_data)

            self.trace = pm.sample(3000, tune=3000, nuts_kwargs=dict(target_accept=0.95))
            pm.traceplot(self.trace, varnames=[w, intercept, rhos, theta, sigma])

        self.estimated_rhos = np.mean(self.trace['rhos'], axis=0)
        self.estimated_w = np.mean(self.trace['w'], axis=0)
        self.estimated_intercept = np.mean(self.trace['intercept'], axis=0)
        self.estimated_theta = np.mean(self.trace['theta'], axis=0)
        self.estimated_sigma = np.mean(self.trace['sigma'], axis=0)

        self.data = data

    def plot_comparisons(self):

        fig, axarr = plt.subplots(2, 2)

        axarr[0][0].plot(self.generated)
        axarr[0][0].set_title('Generated')

        axarr[0][1].plot(self.data)
        axarr[0][1].set_title('Actual')

        axarr[1][0].hist(self.generated)
        axarr[1][0].set_title('Generated')

        axarr[1][1].hist(self.data)
        axarr[1][1].set_title('Actual')

        plt.show()

    def plot_price_monte_carlos(self, prices=None, compare_with_actual=False, N=10000, M=100):

        if compare_with_actual:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            fig, ax1 = plt.subplots(1, 1)

        for _ in range(M):
            simulation, _ = self.generate(N, prices[0])
            ax1.plot(simulation)
            ax1.set_title("Generated price series")

        if compare_with_actual:
            ax2.plot(prices)
            ax2.set_title("Actual prices")

        plt.show()

    def generate_returns(self, N):

        generated = [0.0] * self.num_lags
        current_volatility = self.estimated_intercept
        for t in range(self.num_lags, N):
            current_volatility = current_volatility \
                                + self.estimated_theta * (self.estimated_intercept - current_volatility) \
                                + np.random.normal(0., self.estimated_sigma)

            component = np.random.choice(2, p=self.estimated_w)

            if component == 0:
                y = np.random.standard_normal() * np.exp(current_volatility)
                for l in range(self.num_lags):
                    y += self.estimated_rhos[l] * generated[-(l+1)]

            else:
                y = np.random.normal(0., float(1e-100))


            generated.append(y)

        self.generated = np.array(generated)
        return self.generated

    def generate(self, N, initial_price):

        self.generate_returns(N-1)

        # transform log-returns into prices
        log_initial_price = np.log(initial_price)
        generated_prices = [log_initial_price]

        for r in self.generated:
            y = generated_prices[-1] + r
            generated_prices.append(y)

        return np.exp(np.array(generated_prices)), self.generated
