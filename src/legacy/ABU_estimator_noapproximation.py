import numpy as np
import time
from src.agent_param import Agent_lh
import src.rejection_sampler as rs
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from tests import tests_helper as Tests
from scipy.interpolate import interp1d
import seaborn as sns
sns.set()
import logging

logger = logging.getLogger(__name__)

def polynomial_normalize(polycoeffs,xrange):
    '''

    :param polycoeffs: polynomial co-efficients to normalize. (according to numpy.polynomial.polynomial.polynomial convention)
    :param xrange: The [beginning, ending] point to evaluate around on the normalization on x-axis
    :return: normailzed polynomial and the normalization scale.
    '''
    integral = poly.polyint(polycoeffs)
    sum = np.diff(poly.polyval(xrange,integral))*1.0
    normalized_polynomial = polycoeffs/sum
    # print(sum)
    # polynomial_normalize(normalized_polynomial,xrange)

    Tests.test_for_normalization(normalized_polynomial,xrange)

    # try:
    #     #See if it is normalized yet.
    #     Tests.test_for_normalization(normalized_polynomial,xrange)
    # except AssertionError:
    #     #IF it isn't, call itself again.
    #     polynomial_normalize(normalized_polynomial,xrange)
    return normalized_polynomial, sum

epsilon = np.power(10.0,-10)

class ABU():
    #class to facilitate Approximate Bayesian Updates.
    def __init__(self,mimicking_agent,arena_obj,kwargs):
        self.arena_obj = arena_obj #the arena we are basing everything on.
        self.mim_agent = mimicking_agent #the agent whose parameter-variations we are going to work on.
        self.target_pos = self.mim_agent.curr_position
        self.radius_range = kwargs.get('radius_range',[.1,1])
        self.viewangle_range = kwargs.get('angle_range',[.1,1])

        self.capacity_range_max = np.max(self.arena_obj.grid_matrix)
        self.capacity_range = [.1,self.capacity_range_max]
        self.types = [0,1,2,3]
        self.parameter_range = [self.capacity_range,self.radius_range,self.viewangle_range] #standard order


        self.resolution = kwargs.get('resolution',100)
        self.refit_density = kwargs.get('refit_density',20) #No of division in each dimension p to sample to calculate posterior.

        self.lh_agents = [] #list of type-set-agents with different paremeter settings used to calculate likelihood
        # self.create_lh_objects(0)

        self.degree_likelihoodPolynomial = kwargs.get('likelihood_polyDegree',5)
        self.degree_posteriorPolynomial = kwargs.get('posterior_polyDegree',4)
        self.degree_priorPolynomial = kwargs.get('prior_polyDegree',4)


        self.likelihood_polyCoeff_typesList = []
        self.posterior_polyCoeff_typesList = []
        self.prior_polyCoeff_typesList=[]


        # self.point_density = 100
        self.likelihood_dense_typesList = []
        self.prior_dense_typesList = []
        self.posterior_dense_typesList = []


        self.param_curr = 0 #should be one or two - view radius or view angle
        self.create_lh_objects(self.param_curr)

        self.fit_initialPrior()

        self.posteriorEstimates_sample=[]
        self.posteriorEstimates_maximum=[]

        self.posteriorEstimates_maximum_withoutApproximation = []
        self.posteriorEstimates_sample_withoutApproximation = []

        self.visualize = kwargs.get('visualize',False)
        self.saveplots = kwargs.get('saveplots',False)

        self.total_simSteps = 0
        self.model_evidence = []

        self.fitstats_ll= []
        self.fitstats_po = []

        if self.param_curr == 0:
            self.estimating_parameter = 'view_radius'
        else:
            self.estimating_parameter = 'view_angle'



        logger.info('ABU Estimator class initialized with {} agents and working on estimating parameter {} with a resolution of {}'.format(
                len(self.lh_agents[0]), self.estimating_parameter, self.resolution))
        logger.info('ABU polynomial info: resolution: {}, llpoly_degree: {}, priorpoly_degree: {}, postpoly_degree: {}, refitdensity: {}'.format(
                self.resolution, self.degree_likelihoodPolynomial, self.degree_priorPolynomial,
                self.degree_posteriorPolynomial, self.refit_density))

    def fit_initialPrior(self):
        x_val = self.x_points
        n_parameterset = len(self.x_points)
        y_val =  np.ones(n_parameterset)*1.0/n_parameterset

        priorPoly_coeffs = poly.polyfit(x_val,y_val,deg=self.degree_priorPolynomial)

        self.inital_prior = [priorPoly_coeffs for tp in self.types]
        self.inital_prior_points = np.ones(self.resolution,'float')/n_parameterset
        # self.prior_polyCoeff_typesList.append(self.currPrior_polyCoeff_list)




    def create_lh_objects(self,param_index):
        #only vary one parameter.
        #As capacity leads little information about the actions, so never consider that as a parameter.

        self.capacity_points = self.mim_agent.capacity*np.ones(self.resolution)
        self.capacity_pointsDense =  self.mim_agent.capacity*np.ones(self.refit_density)




        if param_index == 0:
            #choose view radius to estimate
            self.radius_points = np.linspace(self.radius_range[0],self.radius_range[1],self.resolution)
            self.x_points = self.radius_points #Xaxis points for fitting polynomial

            self.radius_pointsDense = np.linspace(self.radius_range[0],self.radius_range[1],self.refit_density)
            self.x_pointsDense = self.radius_pointsDense

            self.xrange = self.radius_range

            self.angle_points = self.mim_agent.viewAngle_param*np.ones(self.resolution)
            self.angle_pointsDense = self.mim_agent.viewAngle_param*np.ones(self.refit_density)
        else:
            self.radius_points = self.mim_agent.viewRadius_param*np.ones(self.resolution)
            self.radius_pointsDense = self.mim_agent.viewRadius_param*np.ones(self.refit_density)

            self.angle_points = np.linspace(self.viewangle_range[0],self.viewangle_range[1],self.resolution)
            self.x_points = self.angle_points

            self.xrange = self.viewangle_range

            self.angle_pointsDense =np.linspace(self.viewangle_range[0],self.viewangle_range[1],self.refit_density)
            self.x_pointsDense = self.angle_pointsDense

        self.types_points =np.linspace(0,3,4).astype('int')


        parameter_set = []
        self.parameter_set = np.vstack((self.capacity_points,self.radius_points,self.angle_points)).T #generates a list of coords with

        #a denser parameter set for using during refit-sampling
        self.paramConfig_denseUniform = np.vstack((self.capacity_pointsDense,self.radius_pointsDense,self.angle_pointsDense)).T

        self.n_parameter_configs = np.shape(self.parameter_set)[0]

        #the last axis varying linearly and the repetition continues across to left.
        self.types_parameterSet_array = []

        for i in self.types_points:
            self.types_parameterSet_array.append(np.hstack((self.parameter_set,i*np.ones((self.n_parameter_configs,1)))))


        for type_param_set,tp in zip(self.types_parameterSet_array,self.types_points):
            type_lh_agent_list = []
            for param_config in type_param_set:
                lh_agent = Agent_lh(param_config,tp,self.target_pos,self.arena_obj)
                lh_agent.curr_orientation = self.mim_agent.curr_orientation
                type_lh_agent_list.append(lh_agent)
            self.lh_agents.append(type_lh_agent_list)

    def poly_findMaximum(self, polyCoeffs):
        derivative = poly.polyder(polyCoeffs, 1)
        inflexion_points_complex = poly.polyroots(derivative)

        #Only need real roots.
        inflexion_points = inflexion_points_complex[np.abs(inflexion_points_complex)==inflexion_points_complex.real]
        inflexion_points = inflexion_points.real

        inflexion_points_inrange = inflexion_points[
            np.logical_and(inflexion_points < self.xrange[1], inflexion_points > self.xrange[0])]
        flex_ypoints = poly.polyval(inflexion_points_inrange,polyCoeffs)

        # if there are no inflexion points in range, then this is a monotonic function. The extremum is on of the edges.
        if len(inflexion_points_inrange) == 0:
            edge_1 = poly.polyval(self.xrange[0],polyCoeffs)
            edge_2 = poly.polyval(self.xrange[1],polyCoeffs)
            if edge_1 < edge_2:
                return self.xrange[1], edge_2
            else:
                return self.xrange[0], edge_1

        maximum_xpoint = inflexion_points_inrange[np.argmax(flex_ypoints)]
        maximum_ypoint = flex_ypoints[int(maximum_xpoint)]
        return maximum_xpoint, maximum_ypoint

    def all_agents_behave(self):
        "Batch behave for all parameter configurations"
        for type_list in self.lh_agents:
            for agent_parameterconf in type_list:
                agent_parameterconf.behave_dummy()
                if agent_parameterconf.type==0:
                    ag = agent_parameterconf
                    vilocs = [item.position for item in ag.visible_items]
                    valocs = [agent.curr_position for agent in ag.visible_agents]
                    # print('In position {}, with visible items {}, agents {} and destination {}'.format(ag.curr_position,vilocs,valocs,ag.curr_destination))
            # if agent_parameterconf.type==0:
            #     print('----------')
            # pass

    def all_agents_imitate(self,action_and_consequence):
        "Batch imitate actions for all parameter configurations"
        action = action_and_consequence
        for type_list in self.lh_agents:
            for agent_parameterconf in type_list:
                agent_parameterconf.execute_action_dummy(action_and_consequence)

    def all_agents_calc_likelihood(self,action_and_consequence):
        "Batch calculate likelihood of all parameter configurations"
        for type_list in self.lh_agents:
            for agent_parameterconf in type_list:
                agent_parameterconf.calc_likelihood(action_and_consequence)
                ag = agent_parameterconf

    def calculate_differenceInProbability(self):
        mse_list = []
        for agent in self.lh_agents[self.mim_agent.type]:
            mse_list.append(np.linalg.norm(self.mim_agent.action_probability-agent.action_probability))
        return np.array(mse_list)

    def calculate_modelEvidence(self,i):
         mevd=[]

         for tp in self.types:
             likelihood_poly = self.likelihood_polyCoeff_typesList[i][tp]
             polyintegral = poly.polyint(likelihood_poly)
             integral = np.diff([poly.polyval(self.xrange,polyintegral)])[0]
             mevd.append(integral)
         self.model_evidence.append(mevd)

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



    def fit_polynomialForLikelihood(self,action_and_consequence,tp):
        """
        We need a method to fit a polynomial to P(a_i/H,theta,p) at every time step i.

        This function should return the polynomial that is fitted at every time step t.

        #fit f_hat as in the paper.This should be a function that can give us the likelihood of taking
        #an action (action_and_consequence) given this (tp) and a parameter setting.
        :param action_and_consequence: action to fit to
        :param tp: tp to fit in
        :return: Nothing really.

        """
        agents_set = self.lh_agents[tp]

        x_val = self.x_points
        y_val = [agent.likelihood_curr for agent in agents_set]
        y = np.array(y_val)



        polyFit_coeffs,stats = poly.polyfit(x_val,y_val,deg=self.degree_likelihoodPolynomial,full=True)
        self.fitstats_ll.append(stats)
        # self.polynomialTransform_list.append(self.polynomial_fit)
        if self.visualize:
            plt.plot(x_val,y,'-g1')
            plt.plot(x_val,poly.polyval(x_val,polyFit_coeffs),'-rs')
            plt.title("Likelihood fit vs real values, action taken is {} and destination is {}".format(action_and_consequence[0],self.mim_agent.curr_destination))
            if self.saveplots:
                plt.savefig('../results/ll-{}-{}.png'.format(tp,time.asctime()))
                plt.close()
            else:
                plt.show()
                plt.close()

        return polyFit_coeffs

    def fit_likelihoodPolynomial_allTypes(self,action_and_consequence):
        likelihood_coeffsArray = []
        likelihood_valuesApprox = []
        for tp in self.types:
            poly_coeffs =  self.fit_polynomialForLikelihood(action_and_consequence,tp)
            likelihood_coeffsArray.append(poly_coeffs)
        self.likelihood_polyCoeff_typesList.append(likelihood_coeffsArray)
        return likelihood_coeffsArray



    def estimate_parameter(self,likelihoodPoly_coeffs,priorPoly_coeffs,tp):
        #prior(as a polynomial)*likelihood(as a polynomial) = posterioir(double degree polynomial)
        #posterior --> downdegree to get back to normal polynomial
        #prior == posterior

        # priorPoly_coeffs = self.currPrior_polyCoeff
        # likelihoodPoly_coeffs = self.likelihood_polyCoeff_list[-1] #latest

        #OPTIMIZE
        prior_values = np.abs(poly.polyval(self.x_pointsDense,priorPoly_coeffs))


        likelihood_values = np.abs(poly.polyval(self.x_pointsDense,likelihoodPoly_coeffs))
        if self.visualize:
            plt.plot(self.x_pointsDense,likelihood_values,'-r1',label='likelihood poly for dense values')
            plt.plot(self.x_points,poly.polyval(self.x_points,likelihoodPoly_coeffs),'-go',label='likelihood poly for orig vals')
            plt.plot(self.x_pointsDense,prior_values,'-kx',label='Prior dense vals')
            plt.title('Likelihood fit for sparse vs dense values')

        #find the posterior probability as a product of prior and likelihood
        posteriorProb_polyCoeffs = poly.polymul(priorPoly_coeffs,likelihoodPoly_coeffs)

        #Densely sample from the polynomial to do a refit to lower degree, actual posterior polynomial
        # posteriorVals = np.abs(poly.polyval(self.x_pointsDense,posteriorProb_polyCoeffs))#ABs because it can become negative in the multiplication

        #testing a different stability mechanism
        posteriorVals = (poly.polyval(self.x_pointsDense,posteriorProb_polyCoeffs)) #ABs because it can become negative in the multiplication
        posteriorVals[posteriorVals<0] = 0

        if self.visualize:
            plt.plot(self.x_pointsDense,posteriorVals,'-bs',label='multiplied posterior poly gen values')

        # plt.plot(posteriorVals)
        # plt.show()
        posteriorProb_polyCoeffs_refit,stats= poly.polyfit(self.x_pointsDense,posteriorVals,self.degree_posteriorPolynomial,full=True)
        self.fitstats_po.append(stats)

        if self.visualize:
            plt.plot(self.x_pointsDense,poly.polyval(self.x_pointsDense,posteriorProb_polyCoeffs_refit),'-cH',label='multiplied rft post pgen vals')

        #integrate the posterior to get normalization
        posteriorProb_polyCoeffs_normalized, posterior_normalization =polynomial_normalize(posteriorProb_polyCoeffs_refit,self.xrange)


        # posteriorProb_polyCoeffs_normalized[np.abs(posteriorProb_polyCoeffs_normalized)<epsilon]=0 #stabilize
        if self.visualize:
            # plt.plot(self.x_pointsDense,poly.polyval(posteriorProb_polyCoeffs_normalized,self.x_pointsDense), label='normalized posterior poly gen values')
            plt.legend()
            if self.saveplots:
                plt.savefig('../results/poll-compa-{}-{}.png'.format(tp,time.asctime()))
                plt.close()
            else:
                plt.show()
                plt.close()

        def pdf_func(p):
            return poly.polyval(p,posteriorProb_polyCoeffs_normalized)

        pdfmax = np.max(posteriorVals)/posterior_normalization
        posterior_samples = rs.rejection_sample(pdf_func,self.xrange[0],self.xrange[1],pdfmax+2,100)

        posterior_estimate_sample = np.mean(posterior_samples)
        posterior_estimate_maximum, maxprob = self.poly_findMaximum(posteriorProb_polyCoeffs_normalized)


        Tests.test_for_normalization(posteriorProb_polyCoeffs_normalized,self.xrange)
        # self.currPrior_polyCoeff = posteriorProb_polyCoeffs_normalized
        # self.posteriorEstimate_sample.append(posterior_estimate_sample)
        # self.posteriorEstimate_maximum.append(posterior_estimate_maximum)
        # print("The posterior estimates are"+str(posterior_estimate_maximum)+' '+str(posterior_estimate_sample))

        return posteriorProb_polyCoeffs_normalized,posterior_estimate_sample,posterior_estimate_maximum

    def estimate_parameter_withoutApproximation(self,likelihood_densePoints,Prior_desnsePoints,tp):
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
            print('Requesting to estimate type {} without approximation'.format(tp))
            update_posteriorPoints, estim_sample,estim_maximum = self.estimate_parameter_withoutApproximation(likelihood_points,prior_points,tp)
            posterior_list.append(update_posteriorPoints)
            estimates_list.append([estim_sample,estim_maximum])
        self.posterior_dense_typesList.append(posterior_list)
        self.posteriorEstimates_maximum_withoutApproximation.append([estimate[1] for estimate in estimates_list])
        self.posteriorEstimates_sample_withoutApproximation.append([estimate[0] for estimate in estimates_list])
        return estimates_list, posterior_list


    def estimate_allTypes(self,i):
        if i>len(self.posterior_polyCoeff_typesList):
            raise Exception("asking for estimate from time samples that don't exist yet")

        estimates_list = []
        posterioir_list = []
        for tp in self.types:
            likelihood_polyCoeffs = self.likelihood_polyCoeff_typesList[i][tp]
            if i==0:
                prior_polyCoeffs = self.inital_prior[tp]
            else:
                prior_polyCoeffs = self.posterior_polyCoeff_typesList[i-1][tp]
            print("Requesting to estimate type {}".format(tp))
            updated_posterioirPoly,pestim_sample,pestim_max = self.estimate_parameter(likelihood_polyCoeffs,prior_polyCoeffs,tp)
            posterioir_list.append(updated_posterioirPoly)
            estimates_list.append([pestim_sample,pestim_max])
        self.posterior_polyCoeff_typesList.append(posterioir_list)
        self.posteriorEstimates_maximum.append([estimate[1] for estimate in estimates_list])
        self.posteriorEstimates_sample.append([estimate[0] for estimate in estimates_list])
        return estimates_list,posterioir_list

    def estimate_singleType_forChamp(self,i,tp):
        if tp>4:
            raise Exception('Wrong type requested')

        likelihood_polyCoeffs = self.likelihood_polyCoeff_typesList[i][tp]
        prior_polyCoeffs = self.inital_prior[tp]
        _, pestim_sample, pestim_max = self.estimate_parameter(likelihood_polyCoeffs, prior_polyCoeffs)
        return pestim_sample, pestim_max


    def estimate_segmentForChamp_type0(self,i,j):
        if j<i:
            raise Exception('Reverse time requested')
        if j>self.total_simSteps-1:
            raise Exception("Simulation not reached until there")
        if i<0:
            raise Exception('Negative time asked')

        tp = 0

        estimate_list = []
        for t in range(i,j+1):
            estimates = self.estimate_singleType_forChamp(t,tp)
            estimate_list.append(estimates)

        final_estimate = estimate_list[-1][0]
        likelihood_total = np.sum(np.log([poly.polyval([final_estimate],self.likelihood_polyCoeff_typesList[i][tp])[0]]))
        return likelihood_total, final_estimate

    def estimate_segmentForChamp_type1(self, i, j):
        if j < i:
            raise Exception('Reverse time requested')
        if j > self.total_simSteps - 1:
            raise Exception("Simulation not reached until there")
        if i < 0:
            raise Exception('Negative time asked')

        tp = 1

        estimate_list = []
        for t in range(i, j + 1):
            estimates = self.estimate_singleType_forChamp(t, tp)
            estimate_list.append(estimates)

        final_estimate = estimate_list[-1][0]
        likelihood_total = np.sum(np.log([poly.polyval([final_estimate],self.likelihood_polyCoeff_typesList[i][tp])[0]]))
        return likelihood_total, final_estimate


    def estimate_segmentForChamp_type2(self, i, j):
        if j < i:
            raise Exception('Reverse time requested')
        if j > self.total_simSteps - 1:
            raise Exception("Simulation not reached until there")
        if i < 0:
            raise Exception('Negative time asked')

        tp = 2

        estimate_list = []
        for t in range(i, j + 1):
            estimates = self.estimate_singleType_forChamp(t, tp)
            estimate_list.append(estimates)

        final_estimate = estimate_list[-1][0]
        likelihood_total = np.sum(np.log([poly.polyval([final_estimate],self.likelihood_polyCoeff_typesList[i][tp])[0]]))
        return likelihood_total, final_estimate


    def estimate_segmentForChamp_type3(self, i, j):
        if j < i:
            raise Exception('Reverse time requested')
        if j > self.total_simSteps - 1:
            raise Exception("Simulation not reached until there")
        if i < 0:
            raise Exception('Negative time asked')

        tp = 3

        estimate_list = []
        for t in range(i, j + 1):
            estimates = self.estimate_singleType_forChamp(t, tp)
            estimate_list.append(estimates)

        final_estimate = estimate_list[-1][0]
        likelihood_total = np.sum(np.log([poly.polyval([final_estimate],self.likelihood_polyCoeff_typesList[i][tp])[0]]))
        return likelihood_total, final_estimate

