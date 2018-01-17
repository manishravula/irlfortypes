from scipy.stats import norm

class Gaussian_fitter:
    def __init__(self, sigma):
        self.mean = 0.0
        self.sigma = sigma

    def fit(self, states, actions):
        lh = 1.0
        for action in actions:
            lh = lh * norm.pdf(action, self.mean, self.sigma)

        return (lh, self.sigma)
    
