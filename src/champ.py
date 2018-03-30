from scipy.stats import norm
from Queue import Queue
from numpy.random import normal
from sampling import *
import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, model_fitters, length_mean = 20.0, length_sigma = 10.0, length_min = 5, max_particles = -1, resamp_particles = -1):

        self.model_fitters = model_fitters
        self.seg_length_mean = length_mean
        self.seg_length_sigma = length_sigma
        self.seg_length_min = length_min

        #when num of particles exceed max_particles, do resampling
        self.max_particles = max_particles

        #after resampling, num of particles is at least resamp_particles
        self.resamp_particles = resamp_particles

class Champ:
    def __init__(self, config):
        self.model_fitters = config.model_fitters
        self.seg_length_mean = config.seg_length_mean
        self.seg_length_sigma = config.seg_length_sigma
        self.seg_length_min = config.seg_length_min
        self.max_particles = config.max_particles
        self.resamp_particles = config.resamp_particles

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
        logger.info('Initialized CHAMP with the following config' )
        logger.info('seg_length_mean: {}'.format(config.seg_length_mean))
        logger.info('seg_length_sigma: {}'.format(config.seg_length_sigma))
        logger.info('seg_length_min: {}'.format(config.seg_length_min))
        logger.info('max_particles: {}'.format(config.max_particles))
        logger.info('resamp_particles: {}'.format(config.resamp_particles))

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
        if(self.max_particles > 0):
            self.particles = resample_particles(self.particles, self.max_particles, self.resamp_particles)
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
            lh, theta = fitter(self.states[p.pos], self.states[self.step])
            g = self.compute_g(self.step - p.pos + 1)
            if(math.isnan(lh) or g == 0.0):
                MAP = -float('inf')
            else:
                MAP = lh + p.prev_MAP + self.pi + math.log(g)
            p.update_MAP(MAP)
            if MAP > max_MAP:
                max_MAP = MAP
                max_particle = p
                max_param = theta
                max_particle.theta = theta
                max_particle.MAP = MAP
        self.prev_MAP.put(max_MAP)
        self.prev_param.append(theta)
        self.prev_particle.append(max_particle)
            
    def compute_g(self, length):
        return norm.pdf(length, self.seg_length_mean, self.seg_length_sigma)/self.CDF

    def backtrack(self, index):
        logger.info('Backtracking in CHAMP')
        if(index >= self.step):
            return None

        model_indices = []
        params = []
        thetas = []
        changepoints = []
        MAPs = []
        NMAPs = []
        while(index >= 0 and self.prev_particle[index] != None):
            prev_particle = self.prev_particle[index]
            logger.info("prev changepoint at: " + str(prev_particle.pos) + " and type is: " + str(prev_particle.model_index))
            changepoints.append(prev_particle.pos)
            model_indices.append(prev_particle.model_index)
            thetas.append(prev_particle.theta)
            params.append(self.prev_param[index])
            MAPs.append(prev_particle.MAP)
            index = prev_particle.pos - 1

        return {'models':model_indices, 'paramestims':params, 'theta':thetas, 'cpindices':changepoints,'map':MAPs}

class Particle:
    def __init__(self, pos, prev_MAP, model_index):
        self.pos = pos
        self.prev_MAP = prev_MAP
        self.model_index = model_index

    def update_MAP(self,MAP):
        self.MAP = MAP

    def update_NMAP(self,NMAP):
        self.NMAP = NMAP
        
if __name__ == "__main__":
    from champ.fitter import Gaussian_fitter
    sigmas = [0.1, 1.0, 10.0]
    
    fitters = []
    for sigma in sigmas:
        fitters.append(Gaussian_fitter(sigma))

    config = Config(fitters, length_min = 10, length_mean = 40.0, length_sigma = 100.0, max_particles = 1000, resamp_particles = 1000)

    champ = Champ(config)

    
    for _ in range(3):
        sigma = sigmas[randint(0, len(sigmas) - 1)]
        length = randint(config.seg_length_min, 50.0)
        logger.info("segment length: "+ str(length) +"   sigma: " + str(sigma))
        for j in range(length):
            champ.observe(None, normal(scale = sigma))

    champ.backtrack(champ.step - 1)
            
    
        

    
