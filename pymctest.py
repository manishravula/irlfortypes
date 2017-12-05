import pymc as mc
import numpy as np

data = np.random.normal(-200,15,size=1000)

mean = mc.Uniform('mean', lower=min(data), upper=max(data))
std_dev = mc.Uniform('std_dev', lower=0, upper=50)

@mc.stochastic(observed=True)
def custom_stochastic(value=data, mean=mean, std_dev=std_dev):
    return np.sum(-np.log(std_dev) - 0.5*np.log(2) -
                  0.5*np.log(np.pi) -
                  (value-mean)**2 / (2*(std_dev**2)))


model = mc.MCMC([mean,std_dev,custom_stochastic])
model.sample(iter=5000)
