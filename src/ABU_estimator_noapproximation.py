import numpy as np
import time
from .agent_param import Agent_lh
from . import rejection_sampler as rs
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from tests import tests_helper as Tests
from scipy.interpolate import interp1d
import seaborn as sns
sns.set()
import logging
from .ABU_estimator import ABU as abu

logger = logging.getLogger(__name__)



epsilon = np.power(10.0,-10)

class ABU(abu):
    #class to facilitate Approximate Bayesian Updates.
    def __init__(self,mimicking_agent,arena_obj,kwargs):
        logger.info('Calling parent ABU via inheritance. This is ABU_noapproximation')
        abu.__init__(self,mimicking_agent,arena_obj,kwargs)
        self.inital_prior_points = np.ones(self.resolution,'float')/(len(self.x_points))
        self.likelihood_dense_typesList = []
        self.prior_dense_typesList = []
        self.posterior_dense_typesList = []
        self.posteriorEstimates_maximum_withoutApproximation = []
        self.posteriorEstimates_sample_withoutApproximation = []

    def get_likelihoodArray(self,tp):
        agents_set = self.lh_agents[tp]
        y_val = [agent.likelihood_curr for agent in agents_set]
        y = np.array(y_val)
        ll_vals = y/np.sum(y)
        return ll_vals

    def get_likelihoodValues_allTypes(self):
        ll_array = []
        for tp in self.types:
            ll_values = self.get_likelihoodArray(tp)
            ll_array.append(ll_values)
        ll_array = np.array(ll_array)
        self.likelihood_dense_typesList.append(ll_array)




    def estimate_parameter_withoutApproximation(self,likelihood_densePoints,Prior_desnsePoints,tp):
        logger.debug("No approx estimator called with type {}".format(tp))
        posterior_points = np.multiply(likelihood_densePoints,Prior_desnsePoints)
        posterior_points_normalized = posterior_points/np.sum(posterior_points)
        pdf_func = interp1d(self.x_points,posterior_points_normalized)

        pdfmax = np.max(posterior_points_normalized)

        num_samples = 100
        posterior_samples = rs.rejection_sample(pdf_func,self.xrange[0],self.xrange[1],pdfmax+.2,num_samples)

        while(len(posterior_samples)<20):
            num_samples = num_samples*10
            posterior_samples = rs.rejection_sample(pdf_func,self.xrange[0],self.xrange[1],pdfmax+.2,num_samples)

        posterior_estimate_sample = np.mean(posterior_samples)

        if self.visualize:
            plt.plot(self.x_points,posterior_points_normalized,'-r*',label='Normalized posterior')
            plt.plot(self.x_points,likelihood_densePoints,'-gx',label='Likelhood')
            plt.legend()
            plt.title('Type {}'.format(tp))
            plt.show()

        posterior_estimate_maximum = self.x_points[np.argmax(posterior_points_normalized)]
        return posterior_points_normalized,posterior_estimate_sample,posterior_estimate_maximum

    def estimate_allTypes_withoutApproximation(self,i):
        estimates_list = []
        posterior_list = []
        for tp in self.types:
            likelihood_points = self.likelihood_dense_typesList[i][tp]
            if i==0:
                prior_points = self.inital_prior_points
            else:
                prior_points = self.posterior_dense_typesList[i-1][tp]
            update_posteriorPoints, estim_sample,estim_maximum = self.estimate_parameter_withoutApproximation(likelihood_points,prior_points,tp)
            posterior_list.append(update_posteriorPoints)
            estimates_list.append([estim_sample,estim_maximum])
        self.posterior_dense_typesList.append(posterior_list)
        self.posteriorEstimates_maximum_withoutApproximation.append([estimate[1] for estimate in estimates_list])
        self.posteriorEstimates_sample_withoutApproximation.append([estimate[0] for estimate in estimates_list])
        return estimates_list, posterior_list



    def estimate_singleType_segment_forChamp_withoutApprox(self,i,j,tp):
        logger.debug("No approxmiation based CHAMP single type estimator called for type {}".format(tp))
        if j<i:
            logger.critical('Exception in CHAMP request')
            raise Exception('Reverse time requested')
        if j>self.total_simSteps-1:
            logger.critical('Exception in CHAMP request')
            raise Exception("Simulation not reached until there")
        if i<0:
            raise Exception('Negative time asked')


        if tp>3:
            raise Exception('Wrong type requested')

        likelihood_vals = self.likelihood_dense_typesList[i][tp]

        estimate_list = []
        mev_list = []
        prior_vals = self.inital_prior_points[tp]

        for t in range(i+1, j+1):
            likelihood_vals = self.likelihood_dense_typesList[i][tp]

            posterior_vals, pestim_sample, pestim_max = self.estimate_parameter_withoutApproximation(likelihood_vals, prior_vals, tp)
            prior_vals = posterior_vals

            mevidence = np.sum(likelihood_vals[:-1])*(self.xdiff) + (.5)*(likelihood_vals[-1]-likelihood_vals[0]) #integral under linear interpolation
            mev_list.append(mevidence)
            estimate_list.append(pestim_sample)


        final_estimate = estimate_list[-1]
        likelihood_total = np.sum(np.log(mev_list))
        return likelihood_total, final_estimate



    def estimate_segmentForChamp_type0_withoutApprox(self,i,j):
        return self.estimate_singleType_segment_forChamp_withoutApprox(i,j,0)

    def estimate_segmentForChamp_type1_withoutApprox(self, i, j):
        return self.estimate_singleType_segment_forChamp_withoutApprox(i,j,1)

    def estimate_segmentForChamp_type2_withoutApprox(self, i, j):
        return self.estimate_singleType_segment_forChamp_withoutApprox(i,j,2)

    def estimate_segmentForChamp_type3_withoutApprox(self, i, j):
        return self.estimate_singleType_segment_forChamp_withoutApprox(i,j,3)
