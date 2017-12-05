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
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt


epsilon = np.power(10,-10)

class ABU():
    #class to facilitate Approximate Bayesian Updates.
    def __init__(self,mimicking_agent,arena_obj):
        self.arena_obj = arena_obj #the arena we are basing everything on.
        self.mim_agent = mimicking_agent #the agent whose parameter-variations we are going to work on.
        self.target_pos = self.mim_agent.curr_position
        self.radius_range = [.1,1]
        self.viewangle_range = [.1,.49]

        self.capacity_range_max = np.max(self.arena_obj.grid_matrix)
        self.capacity_range = [.1,self.capacity_range_max]
        self.types = [0,1,2,3]
        self.parameter_range = [self.capacity_range,self.radius_range,self.viewangle_range] #standard order


        self.resolution = 9
        self.refit_density = 200 #No of division in each dimension p to sample to calculate posterior.

        self.lh_agents = [] #list of type-set-agents with different paremeter settings used to calculate likelihood

        self.likelihood_polyCoeff_list = []
        self.posterior_polyCoeff_list = []

        self.param_curr = 1 #should be one or two - view radius or view angle
        self.create_lh_objects(self.param_curr)

        self.degree_likelihoodPolynomial = 20

        #prior and posterior need to be same.
        self.degree_posteriorPolynomial = 6
        self.degree_priorPolynomial = 6

        self.fit_initialPrior()

        self.posteriorEstimate_sample=[]
        self.posteriorEstimate_maximum=[]



    def fit_initialPrior(self):
        x_val = self.x_points
        n_parameterset = len(self.x_points)*1.0
        y_val =  np.ones(n_parameterset)/n_parameterset

        priorPoly_coeffs = np.polyfit(x_val,y_val,deg=self.degree_priorPolynomial)

        self.currPrior_polyCoeff = priorPoly_coeffs




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

            self.angle_points = self.mim_agent.view_angle*np.ones(self.resolution)
            self.angle_pointsDense = self.mim_agent.view_angle*np.ones(self.refit_density)
        else:
            self.radius_points = self.mim_agent.view_radius*np.ones(self.resolution)
            self.radius_pointsDense = self.mim_agent.view_radius*np.ones(self.refit_density)

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


    def all_agents_behave(self):
        "Batch behave for all parameter configurations"
        for type_list in self.lh_agents:
            for agent_parameterconf in type_list:
                if agent_parameterconf.view_angle>np.pi:
                    pass
                agent_parameterconf.behave_dummy()
                print agent_parameterconf.curr_orientation
            pass

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
            pass
        pass

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

        if np.all(y==np.sort(y)):
            print("FUCK")

        polyFit_coeffs = np.polyfit(x_val,y_val,deg=self.degree_likelihoodPolynomial)
        # self.polynomialTransform_list.append(self.polynomial_fit)

        self.likelihood_polyCoeff_list.append(polyFit_coeffs)


    def estimate_parameter(self,tp):
        #prior(as a polynomial)*likelihood(as a polynomial) = posterioir(double degree polynomial)
        #posterior --> downdegree to get back to normal polynomial
        #prior == posterior

        priorPoly_coeffs = self.currPrior_polyCoeff
        likelihoodPoly_coeffs = self.likelihood_polyCoeff_list[-1] #latest

        #OPTIMIZE
        prior_values = np.polyval(priorPoly_coeffs,self.x_pointsDense)
        likelihood_values = np.polyval(likelihoodPoly_coeffs,self.x_pointsDense)

        #find the posterior probability as a product of prior and likelihood
        posteriorProb_polyCoeffs = np.polymul(priorPoly_coeffs,likelihoodPoly_coeffs)

        #Densely sample from the polynomial to do a refit to lower degree, actual posterior polynomial
        posteriorVals = np.abs(np.polyval(posteriorProb_polyCoeffs,self.x_pointsDense))#ABs because it can become negative in the multiplication
        # plt.plot(posteriorVals)
        # plt.show()
        posteriorProb_polyCoeffs_refit = np.polyfit(self.x_pointsDense,posteriorVals,self.degree_posteriorPolynomial)

        #integrate the posterior to get normalization
        posterior_integral = np.polyint(posteriorProb_polyCoeffs_refit)
        posterior_normalization = np.diff(np.polyval(posterior_integral,self.xrange))[0]

        posteriorPoly_coeffs = posteriorProb_polyCoeffs_refit/posterior_normalization

        posteriorPoly_coeffs[posteriorPoly_coeffs<epsilon]=0 #stabilize

        #sample from posterior
        class posteriorGen(stats.rv_continuous):
            #sampler for posterior
            def _pdf(self, x, *args):
                return np.polyval(posteriorPoly_coeffs,x)

        posterior_samples = posteriorGen(a=self.xrange[0],b=self.xrange[1]).rvs(size=10)

        posterior_estimate_sample = np.mean(posterior_samples)
        posterior_estimate_maximum, maxprob = self.poly_findMaximum(posteriorPoly_coeffs)

        self.currPrior_polyCoeff = posteriorPoly_coeffs
        self.posteriorEstimate_sample.append(posterior_estimate_sample)
        self.posteriorEstimate_maximum.append(posterior_estimate_maximum)
        print("The posterior estimates are"+str(posterior_estimate_maximum)+' '+str(posterior_estimate_sample))




    def poly_findMaximum(self,polyCoeffs):
        derivative = np.polyder(polyCoeffs,1)
        inflexion_points = poly.polyroots(derivative)

        inflexion_points_inrange = inflexion_points[np.logical_and(inflexion_points<self.xrange[1],inflexion_points>self.xrange[0])]
        flex_ypoints = np.polyval(polyCoeffs,inflexion_points_inrange)

        #if there are no inflexion points in range, then this is a monotonic function. The extremum is on of the edges.

        if len(inflexion_points_inrange)==0:
            edge_1 = np.polyval(polyCoeffs,self.xrange[0])
            edge_2 = np.polyval(polyCoeffs, self.xrange[1])
            if edge_1<edge_2:
                return self.xrange[1],edge_2
            else:
                return self.xrange[0],edge_1

        maximum_xpoint = inflexion_points_inrange[np.argmax(flex_ypoints)]
        maximum_ypoint = flex_ypoints[maximum_xpoint]

        return maximum_xpoint,maximum_ypoint

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
while not are.isterminal:
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
    abu.fit_polynomialForLikelihood(agent_actions_list[0][0],0)
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