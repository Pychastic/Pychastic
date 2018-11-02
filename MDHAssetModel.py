import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import pymc3.distributions.timeseries as ts

class MDHAssetModel:

    def __init__(self, num_lags=0, w=[0.62, 0.38], nu=10.853, sigma=0.51, intercept=-5.668, theta=0.943, rhos=[]):
        self.num_lags = num_lags
        self.estimated_w = w
        self.estimated_nu = nu
        self.estimated_sigma = sigma
        self.estimated_intercept = intercept
        self.estimated_theta = theta
        self.estimated_rhos = rhos
        self.model = None
        self.trace = None

    def loo(self):

        return pm.loo(self.trace, self.model)

    def display_params(self):

        print "component weights = ", self.estimated_w
        print "returns distribution Student nu = ", self.estimated_nu
        print "stochastic volatility standard deviation = ", self.estimated_sigma
        print "stochastic volatility mean reversion speed = ", self.estimated_theta
        print "stochastic volatility long-term mean = ", self.estimated_intercept

        if self.num_lags > 0:
            print "auto-correlation coefficients of the first %s lags = %s" % (self.num_lags, self.estimated_rhos)

    def fit(self, prices):

        logreturns = np.log(prices[1:]) - np.log(prices[:-1])

        self.fit_logreturns(logreturns)

    def fit_logreturns(self, data):

        if self.num_lags > 0:
            print "Estimating the %s auto-correlation lags first." % (self.num_lags)

            with pm.Model() as ar_model:

                rhos = pm.Uniform('rhos', lower=-1., upper=1., shape=self.num_lags)

                # First estimate the lags
                pm.AR('auto-correlations', rho=rhos, sd=0.0001, observed=data)

                trace = pm.sample(2000, tune=2000)

            self.estimated_rhos = np.mean(trace['rhos'], axis=0)

            print "Now estimating the stochastic volatility and the return distribution components."

        with pm.Model() as self.model:
            W = np.array([1., 1.])

            w = pm.Dirichlet('w', W)

            intercept = pm.Normal('intercept', mu=-5, sd=5., testval=-5.)
            theta = pm.Uniform('theta', lower=0.01, upper=1.)
            sigma = pm.Uniform('sigma', lower=0.01, upper=1.)

            sde = lambda x, theta, mu: (theta * (mu-x), sigma)
            s = ts.EulerMaruyama('path', 1.0, sde, [theta, intercept], shape=len(data), testval=np.ones_like(data))

            nu = pm.Exponential('nu', .1)

            comp1 = pm.StudentT.dist(mu=0., nu=nu, sd=pm.math.exp(s))
            comp2 = pm.Normal.dist(mu=0., sd=float(1e-100))

            pm.Mixture('obs', w=w, comp_dists=[comp1, comp2], observed=data)

            mean_field = pm.fit(20000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
            self.trace = mean_field.sample(10000)

        self.estimated_w = np.mean(self.trace['w'], axis=0)
        self.estimated_nu = np.mean(self.trace['nu'], axis=0)
        self.estimated_intercept = np.mean(self.trace['intercept'], axis=0)
        self.estimated_sigma = np.mean(self.trace['sigma'])
        self.estimated_theta = np.mean(self.trace['theta'])

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
                y = np.random.standard_t(self.estimated_nu) * np.exp(current_volatility)
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
