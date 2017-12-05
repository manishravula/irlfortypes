import numpy as np
from arena import arena
from agent import Agent
import time
import pdb
import copy
from MCTS import mcts_unique as mu
from agent_param import Agent_lh
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import polynomial_integrals as pintegral
from scipy import stats
from scipy import optimize as sopt


class ABU():
    def __init__(self,mimicking_agent,arena_obj):
        self.arena_obj = arena_obj
        self.mim_agent = mimicking_agent
        self.target_pos = self.mim_agent.curr_position
        self.radius_range = [.1,1]
        self.viewangle_range = [.1,1]
        self.capacity_range_max = np.max(self.arena_obj.grid_matrix)
        self.capacity_range = [.1,self.capacity_range_max]
        self.types = [0,1,2,3]
        self.parameter_range = [self.capacity_range,self.radius_range,self.viewangle_range] #standard order


        self.resolution = 9
        self.refit_density = 20 #No of division in each dimension p to sample to calculate posterior.

        self.lh_agents = [] #list of type-set-agents with different paremeter settings used to calculate likelihood
        self.create_lh_objects()

        self.degree_likelihoodPolynomial = 20
        self.degree_posteriorPolynomial = 6
        self.degree_priorPolynomial = 6

        self.likelihood_polyTransform = PolynomialFeatures(self.degree_likelihoodPolynomial)
        self.likelihood_polyTransform_paramConfigXval=  self.likelihood_polyTransform.fit_transform(self.parameter_set)
        self.likelihood_polyTransform_paramConfig_denseUniformXval = self.likelihood_polyTransform.fit_transform(self.paramConfig_denseUniform)

        self.likelihood_polyCoeffRegressor_list = [] #list to hold LinearRegression objects for polynomial coefficients
        self.likelihood_polyCoeffRegressor_weightsList= [] #list to hold polynomial co-efficients


        self.prior_polyTransform=PolynomialFeatures(self.degree_priorPolynomial)
        self.prior_polyTransfrom_paramConfig_denseUniformXval = self.prior_polyTransform.fit_transform(self.paramConfig_denseUniform)
        self.prior_polyCoeffRegressorList= [] #list to hold the priors over p at each step

        self.posterior_polyTransfrom = PolynomialFeatures(self.degree_posteriorPolynomial)
        self.posterior_polyTransfrom_paramConfig_denseUniformXval = self.posterior_polyTransfrom.fit_transform(self.paramConfig_denseUniform)
        self.posterior_polyCoeffRegressorList= [] #list to hold the posteriors over p at each step.


        self.currPrior_polyCoeffRegressor = self.get_initialPrior()
        self.prior_polyCoeffRegressorList.append(self.currPrior_polyCoeffRegressor)

        self.parameter_estimates = []





    def get_initialPrior(self):
        x_val = self.parameter_set
        n_parameterset = len(self.parameter_set)*1.0
        y_val =  np.ones(n_parameterset)/n_parameterset

        xval_poly = self.prior_polyTransform.fit_transform(x_val)
        l = LinearRegression()
        l.fit(xval_poly,y_val)
        return l



    def create_lh_objects(self):
        self.radius_points = np.linspace(self.radius_range[0],self.radius_range[1],self.resolution)
        self.angle_points = np.linspace(self.viewangle_range[0],self.viewangle_range[1],self.resolution)
        self.capacity_points = np.linspace(self.capacity_range[0],self.capacity_range[1],self.resolution)
        self.types_points =np.linspace(0,3,4)

        rp_dense = np.linspace(self.radius_range[0],self.radius_range[1],self.refit_density)
        ap_dense = np.linspace(self.viewangle_range[0],self.viewangle_range[1],self.refit_density)
        cp_dense = np.linspace(self.capacity_range[0],self.capacity_range[1],self.refit_density)

        parameter_set = []
        self.parameter_set = np.array(np.meshgrid(self.capacity_points,self.radius_points,self.angle_points)).T.reshape(-1,3) #generates a list of coords with

        #a denser parameter set for using during refit-sampling
        self.paramConfig_denseUniform = np.array(np.meshgrid(cp_dense,rp_dense,ap_dense)).T.reshape(-1,3)

        self.n_parameter_configs = np.shape(self.parameter_set)[0]

        #the last axis varying linearly and the repetition continues across to left.
        self.types_parameterSet_array = []

        for i in self.types_points:
            self.types_parameterSet_array.append(np.hstack((self.parameter_set,i*np.ones((self.n_parameter_configs,1)))))


        for type_param_set,tp in zip(self.types_parameterSet_array,self.types_points):
            type_lh_agent_list = []
            for param_config in type_param_set:
                lh_agent = Agent_lh(param_config,tp,self.target_pos,self.arena_obj)
                type_lh_agent_list.append(lh_agent)
            self.lh_agents.append(type_lh_agent_list)


    def all_agents_behave(self):
        "Batch behave for all parameter configurations"
        for type_list in self.lh_agents:
            for agent_parameterconf in type_list:
                agent_parameterconf.behave_dummy()

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

        x_val = self.likelihood_polyTransform_paramConfigXval
        y_val = [agent.likelihood_curr for agent in agents_set]
        y = np.array(y_val)

        if np.all(y==np.sort(y)):
            print("FUCK")

        self.polynomial_fit = PolynomialFeatures(self.degree_likelihoodPolynomial)
        # self.polynomialTransform_list.append(self.polynomial_fit)


        l = LinearRegression()
        l.fit(self.likelihood_polyTransform_paramConfigXval,y_val)

        self.likelihood_polyCoeffRegressor_list.append(l)
        self.likelihood_polyCoeffRegressor_weightsList.append(l.coef_)


    def estimate_parameter(self,tp):
        #prior(as a polynomial)*likelihood(as a polynomial) = posterioir(double degree polynomial)
        #posterior --> downdegree to get back to normal polynomial
        #prior == posterior

        prior_func = self.currPrior_polyCoeffRegressor
        likelihood_func = self.likelihood_polyCoeffRegressor_list[-1] #latest

        #OPTIMIZE

        prior_values = prior_func.predict(self.prior_polyTransfrom_paramConfig_denseUniformXval)
        likelihood_values = likelihood_func.predict(self.likelihood_polyTransform_paramConfig_denseUniformXval)

        posteriorProbFunc_yvals = np.multiply(prior_values,likelihood_values)

        #now fit polynomial to posterior.
        l = LinearRegression(fit_intercept=False)
        l.fit(self.posterior_polyTransfrom_paramConfig_denseUniformXval,posteriorProbFunc_yvals)

        #integrate the posterior to get normalization
        integral_coeffs = pintegral.calculate_integral_multipliers(self.parameter_range,self.degree_posteriorPolynomial)

        poly_coeffs = l.coef_
        poly_coeffs = np.array(poly_coeffs).reshape(-1,1)

        polyIntegral_coeffs = np.hstack((poly_coeffs,integral_coeffs))
        polyIntegral = np.sum(np.prod(polyIntegral_coeffs,axis=1))

        #normalizing the posterior probability by dividing it with the integral
        l.coef_/=polyIntegral
        l.intercept_/=polyIntegral


        #resample from posterior
        self.get_MAPestimate(l,self.posterior_polyTransfrom)
        # class posterior_dist(stats.rv_continuous):
        #     def _pdf(self, param):
        #         l.predict(param)
        #
        # distribution = posterior_dist()
        #
        # n_samples = 10
        # resampled_parameters =[]
        # for i in range(n_samples):
        #     resampled_parameters.append(distribution.rvs())
        # parameter_estimate = np.mean(resampled_parameters,axis=1)
        # self.parameter_estimates.append(parameter_estimate)

        self.currPrior_polyCoeffRegressor = l

    def get_MAPestimate(self,distribution,polyTransform):

        def f(scaled_param):
            new_param = np.multiply(scaled_param,range_array)+min_array
            print new_param.shape
            val = distribution.predict(polyTransform.fit_transform([new_param]))
            pass
            return 0-val[0]

        #capacity, radius, view_angle

        #preopt - scaling.
        range_1 = self.capacity_range[1]-self.capacity_range[0]
        range_2 = self.radius_range[1]-self.radius_range[0]
        range_3 = self.viewangle_range[1]-self.viewangle_range[0]

        range_array=np.array([range_1,range_2,range_3])
        min_array = np.array([self.capacity_range[0],self.radius_range[0],self.viewangle_range[0]])

        initial_guess = np.array([.5,.5,.5]).reshape(1,-1)
        bound = np.array([[0,1],[0,1],[0,1]]).reshape((3,2))
        # opt = sopt.fmin_l_bfgs_b(f,x0=initial_guess,bounds=bound)
        start=time.time()
        opt = sopt.differential_evolution(f,[[0,1],[0,1],[0,1]])
        print(time.time()-start)
        print("the parameter estimate is "+str(np.multiply(opt['x'],range_array)+min_array))

    def get_likelihoodForParam(self,time_step,param_vector):
        """
        Calculate the likelihood for action at timestep if the agent had param_vector params
        :param timestep: index of timestep, this automagically fixes action
        :param param_vector: param_vector in the universal param format
        :return:
        """
        #optimize
        return self.regressor_list[time_step].predict(self.polynomial_fit.fit_transform([param_vector]))











    def get_agentFromParamConfig(self,param_config,tp):
        """
        The method returns the simluating-agent object holding param_config settings
        :param param_config: The parameter config for while
        :return:Agent object
        """
        type_agentsSet = self.lh_agents[tp]

        #can be optimize
        agent_index = np.argwhere(np.all(self.parameter_set==param_config,axis=1))[0][0]
        return type_agentsSet[agent_index]



















grid_matrix = np.load('grid.npy')
grid_matrix/=2.0
g2 = copy.deepcopy(grid_matrix)


are = arena(grid_matrix,True)
a1 = Agent(0.42,.46,.25,0,np.array([3,4]),are)
a1.load = False

# a2 = Agent(0.2,3,4,3,np.array([5,5]),2,are)
# a2.load = True

a3 = Agent(.49,.2,.2,2,np.array([6,7]),are)
a3.load = False

a2 = Agent(.3,.25,.3,0,np.array([7,7]),are)
a2.load = False

# ad = Agent_lh(.1,4,.6,0,np.array([7,7]),are)
# ad.load = False
#
# ad2 = Agent_lh(.1,4,.6,0,np.array([7,7]),are)
# ad2.load = False

# are.add_agents([a4,a2,a3,a1])
are.add_agents([a1,a2,a3])
abu = ABU(a1,are)


g1= are.grid_matrix

gm=[]
time_array = []
i=0
j=0
prob_lh = []
prob_lh2 = []
prob_ori = []
while i<10:
    print("iter "+str(i))
    print("fail "+str(j))
    i+=1

    start = time.time()

    #first estimate the probability
    abu.all_agents_behave()
    all_dest = []
    all_pos = []
    for tp in range(4):
        all_dest+=[agent.curr_destination for agent in abu.lh_agents[tp]]
        all_pos+=[agent.curr_position for agent in abu.lh_agents[tp]]



    print("--------------------------------")
    n_fuck =np.sum(all_dest==[None for l in range(9*9*9*4)])
    print(n_fuck)

    if n_fuck==9*9*9*4:
        j+=1
    print("--------------------------------")

    # a1 = abu.lh_agents[0][35]
    # a2 = abu.lh_agents[1][450]
    # print(abu.lh_agents[0][35].action_probability)
    # print(abu.lh_agents[0][35].param_vector)
    # print(abu.lh_agents[1][450].action_probability)
    # print(abu.lh_agents[1][450].param_vector)


    print(a1.curr_destination)
    print(a2.curr_destination)
    #then let the true-agents perform their actions
    agent_actions_list,action_probs = are.update()

    #then let fake-act on by imitating this true-action
    abu.all_agents_imitate(agent_actions_list[0]) #zero because we are following the first agent.

    abu.all_agents_calc_likelihood(agent_actions_list[0])
    abu.fit_polynomialForLikelihood(agent_actions_list[0][0],2)
    abu.estimate_parameter(0)

    are.check_for_termination()

    delta = time.time()-start
    time_array.append(delta)
    print(delta)


print time_array
# prob_lh = np.array(prob_lh).astype('float32')
# prob_ori = np.array(prob_ori)
#
#
# print np.where(prob_lh==0)
# print(np.product(prob_lh))
# print(np.sum(np.log(prob_lh)))
# print(np.sum(np.log(prob_lh2)))
# if np.all(prob_lh):
#     print('All set')
# else:
#     print("This is not right")
#
# if np.all(prob_ori):
#     print('All set here too')
# else:
#     print("this is not right either.")
