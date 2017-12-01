import numpy as np
from scipy.stats import norm
from Queue import Queue
import math
from random import randint
from numpy.random import normal

class Config:
    def __init__(self, model_fitters, length_mean = 20.0, length_sigma = 10.0, length_min = 5):
        self.model_fitters = model_fitters
        self.seg_length_mean = length_mean
        self.seg_length_sigma = length_sigma
        self.seg_length_min = length_min

class Champ:
    def __init__(self, config):
        self.model_fitters = config.model_fitters
        self.seg_length_mean = config.seg_length_mean
        self.seg_length_sigma = config.seg_length_sigma
        self.seg_length_min = config.seg_length_min
        #self.max_particles = config.max_particles
        #self.resamp_particles = config.resamp_particles

        self.states = []
        self.actions = []
        self.particles = []
        self.prev_particle = []
        self.prev_param = []
        self.prev_MAP = Queue()
        self.step = 0
        self.CDF = 1 - norm.cdf(self.seg_length_min, self.seg_length_mean, self.seg_length_sigma)

        self.pi = math.log(1.0/len(self.model_fitters))
        for i in range(len(self.model_fitters)):
            p = Particle(0, self.pi , i)
            self.particles.append(p)

    def observe(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        if(self.step > 2 * self.seg_length_min - 2):
            self.create_particles()
        if(self.step >= self.seg_length_min - 1):
            self.compute_MAP()
        else:
            self.prev_particle.append(None)
            self.prev_param.append(None)
        self.step = self.step + 1

    def create_particles(self):
        prev_MAP = self.prev_MAP.get()
        for i in range(len(self.model_fitters)):
            p = Particle(self.step - self.seg_length_min + 1, prev_MAP, i)
            self.particles.append(p)

    def compute_MAP(self):
        max_MAP = -float('inf')
        max_particle = None
        max_param = None
        for p in self.particles:
            fitter = self.model_fitters[p.model_index]
            lh, theta = fitter.fit(self.states[p.pos:self.step], self.actions[p.pos:self.step])
            g = self.compute_g(self.step - p.pos + 1)
            if(lh == 0.0 or g == 0.0):
                MAP = -float('inf')
            else:
                MAP = math.log(lh) + p.prev_MAP + self.pi + math.log(g)
            if MAP > max_MAP:
                max_MAP = MAP
                max_particle = p
                max_param = theta
        self.prev_MAP.put(max_MAP)
        self.prev_param.append(theta)
        self.prev_particle.append(max_particle)
            
    def compute_g(self, length):
        return norm.pdf(length, self.seg_length_mean, self.seg_length_sigma)/self.CDF

    def backtrack(self, index):
        if(index >= self.step):
            return None

        model_indices = []
        params = []
        while(index >= 0 and self.prev_particle[index] != None):
            prev_particle = self.prev_particle[index]
            print("prev changepoint at: " + str(prev_particle.pos))
            model_indices.append(prev_particle.model_index)
            params.append(self.prev_param[index])
            index = prev_particle.pos - 1

        return (model_indices, params)

class Particle:
    def __init__(self, pos, prev_MAP, model_index):
        self.pos = pos
        self.prev_MAP = prev_MAP
        self.model_index = model_index
        
if __name__ == "__main__":
    from fitter import Gaussian_fitter
    sigmas = [0.1, 1.0, 10.0]
    
    fitters = []
    for sigma in sigmas:
        fitters.append(Gaussian_fitter(sigma))
    config = Config(fitters, length_min = 5, length_mean = 20.0, length_sigma = 10.0)

    champ = Champ(config)

    
    for _ in range(3):
        sigma = sigmas[randint(0, len(sigmas) - 1)]
        length = randint(config.seg_length_min, 50.0)
        print("segment length: "+ str(length) +"   sigma: " + str(sigma))
        for j in range(length):
            champ.observe(None, normal(scale = sigma))

    champ.backtrack(champ.step - 1)
            
    
        

    
