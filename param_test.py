import numpy as np
from arena import arena
from agent import Agent
import time
import pdb
import copy
from MCTS import mcts_unique as mu
from agent_param import Agent_lh
from sklearn.preprocessing import PolynomialFeatures

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


        self.resolution = 5
        self.lh_agents = [] #list of type-set-agents with different paremeter settings used to calculate likelihood
        self.create_lh_objects()


        self.polynomial_degree = 4


    def create_lh_objects(self):
        self.radius_points = np.linspace(self.radius_range[0],self.radius_range[1],self.resolution)
        self.angle_points = np.linspace(self.viewangle_range[0],self.viewangle_range[1],self.resolution)
        self.capacity_points = np.linspace(self.capacity_range[0],self.capacity_range[1],self.resolution)
        self.types_points =np.linspace(0,3,4)


        parameter_set = []
        self.parameter_set = np.array(np.meshgrid(self.capacity_points,self.radius_points,self.angle_points)).T.reshape(-1,3) #generates a list of coords with
        self.n_parameter_configs = np.shape(self.parameter_set)[0]

        #the last axis varying linearly and the repetition continues across to left.
        self.types_parameterSet_array = []

        for i in self.types_points:
            self.types_parameterSet_array.append(np.hstack((self.parameter_set,i*np.ones((self.n_parameter_configs,1)))))


        for type_param_set in self.types_parameterSet_array:
            type_lh_agent_list = []
            for param_config in type_param_set:
                lh_agent = Agent_lh(param_config,self.target_pos,self.arena_obj)
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


    def fit_polynomial(self,action_and_consequence,type):
        """
        We need a method to fit a polynomial to P(a_i/H,theta,p) at every time step i.

        This function should return the polynomial that is fitted at every time step t.

        #fit f_hat as in the paper.This should be a function that can give us the likelihood of taking
        #an action (action_and_consequence) given this (type) and a parameter setting.
        :param action_and_consequence: action to fit to
        :param type: type to fit in
        :return: Nothing really.

        """
        agents_set = self.lh_agents[type]

        x_val = self.parameter_set #This is the set of variables that form the domain of the polynomial
        y_val = [agent.likelihood_curr for agent in agents_set]

        p = PolynomialFeatures(self.polynomial_degree)
        polyTransform_xval = p.fit_transform(x_val)
        4







    def get_agentFromParamConfig(self,param_config,type):
        """
        The method returns the simluating-agent object holding param_config settings
        :param param_config: The parameter config for while
        :return:Agent object
        """
        type_agentsSet = self.lh_agents[type]

        #can be optimize
        agent_index = np.argwhere(np.all(self.parameter_set==param_config,axis=1))[0][0]
        return type_agentsSet[agent_index]



















grid_matrix = np.load('grid.npy')
grid_matrix/=2.0
g2 = copy.deepcopy(grid_matrix)


are = arena(grid_matrix,False)
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
prob_lh = []
prob_lh2 = []
prob_ori = []
while not are.isterminal:
    print("iter "+str(i))
    i+=1

    start = time.time()

    #first estimate the probability
    abu.all_agents_behave()

    #then let the true-agents perform their actions
    agent_actions_list,action_probs = are.update()

    #then let fake-act on by imitating this true-action
    abu.all_agents_imitate(agent_actions_list[0]) #zero because we are following the first agent.

    abu.all_agents_calc_likelihood(agent_actions_list[0])

    are.check_for_termination()

    time_array.append(time.time()-start)


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
