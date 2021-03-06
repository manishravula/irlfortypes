import numpy as np
import types
# import time
import threading
import matplotlib.pyplot as plt
from copy import copy
from copy import deepcopy
import pdb
import levelbasedforaging_visualizer as lvlvis
import itertools
import config_experiment as config

"""
ITEM should also be an object. with position and capacity.

"""
"""
WORLD:

1) Methods:
    a) Instantiate - start with locations of food, agents and parameters of agents, objects of other agents.
    b) ExternalStep - Recieve a step-next position from the external agent object. 
    c) InternalStep - Recieve a step-next position from the internal agent object.
    d) Check the world for consequences - And enforce them.
    e) If there is a reward mechanism, save it. 

2) MCTS Wrapper:
    a) How does MCTS connect to everything else?
    b) MCTS sub-simulation of the entire world. So each MCTS run utilizes a new WORLD instance to run forward.

"""

"""
REFDOC for ARENA Class:

1) Connections.
        
                                                           ->/   Agent1     \->
                                                          ->/    Agent2      \->   
                                                           /                  \ 
     Variables: Grid_matrix,item_list,agent_list          /                    \    Variables: Pos, parameters, orientation. (The passed objects include this)
     Methods: simulation_nextstep()-Make food disappear   \                    /    Methods: Run next_step. -> Should give the next position. 
                                                           \                  /      
                                                          ->\    Agent3      /->
                                                           ->\   MCTSAgent1 /->
                                                        
2) Each simluation step should also update the info in the grid_matrix to make the pygame animation display run well.
3) arena.update()
    This function should be updating the arena after every object has moved.
    It needs to 
      a) Check agents reaching to pick up any object. 
          if they do, make that object disappear by deleting it off the grid_matrix
          else
          let everything be.
      b) Update visualization - through pygame
4) This function

##THE SIM CLASS HAS TO BE SEPERATE - BECAUSE THE MCTS agent has to play often!

Methods:
    1) init()
    2) MCTS - helpers. 
        current_board()
        players_


"""



class item():
    def __init__(self,position,weight):
        self.position = position
        self.weight = weight
    def copy(self):
        return item(copy.deepcopy(self.position),copy.deepcopy(self.weight))


class arena():
    def __init__(self,grid_matrix,visualize):
        self.grid_matrix = grid_matrix
        self.agents = []
        self.items = []
        self.visualize = visualize
        self.create_objectitems()
        self.consumed_items = []

        self.dict_actionIndex = {'u':0,'d':1,'l':2,'r':3,'a':4}
        self.action_hashes =  [action for action in self.dict_actionIndex.iterkeys()]
        self.actions = np.arange(5)
        self.isterminal = False
        self.astar_computed_dest = []
        self.astar_computed_path = []

    def get_item_posarray(self):
        posarray = []
        for item in self.items:
            posarray.append(item.position)
        self.item_pos_array = np.array(posarray)

    def get_agent_posarray(self):
        posarray = []
        for agent in self.agents:
            posarray.append(agent.curr_position)
        self.agent_pos_array = np.array(posarray)

    def update_mod_gridmatrx(self):
        #Regular grid matrix with ones in the place of agents to make MCTS work.
        self.mod_gridmatrix = np.copy(self.grid_matrix)
        self.mod_gridmatrix[self.agent_pos_array.T[0],self.agent_pos_array.T[1]]+=1



    def add_agents(self,agents_list):

        #Add agent objects once they are created.
        #The last agent added is a dummy agent, used for MCTS
        self.agents = agents_list
        self.no_agents = len(self.agents)
        self.mcts_agent = self.agents[-1]
        for agent in self.agents:
            self.grid_matrix[agent.curr_position[0],agent.curr_position[1]]=1
        if self.visualize:
            agent_parameters = [agent.params for agent in self.agents]
            self.visualizer = lvlvis.LVDvisualizer(self.grid_matrix,agent_parameters)
            self.visualize_thread = threading.Thread(target=self.visualizer.wait_on_event)
            self.visualize_thread.start()


    def create_objectitems(self):
        items_loc = np.argwhere(np.logical_and(self.grid_matrix>0,self.grid_matrix<1)) #agents' positions are identified by ones
        for loc in items_loc:
            item_obj = item(loc,self.grid_matrix[loc[0],loc[1]])
            self.items.append(item_obj)
        self.no_items = len(self.items)
        self.get_item_posarray()

    def update(self):
        agent_actions = []
        agent_probs = []

        #retrieve what the agent wants to do
        for agent in self.agents:
            #Check what the agent wants to do
            action_probs = agent.behave(False)

            #retrieve the action.
            agent_action = agent.behave_act(action_probs)
            if np.any(agent_action[2]>9):
                pass



            agent_actions.append(agent_action)
            agent_probs.append(action_probs)

            #Approve the agent's action. This way, if agent moves further and is in
            #collision path with another agent, then this is not going to be aproble
            #as the other agent will plan accordingly.
            agent.execute_action(agent_action)


        #See if there is any load operation.
        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()

        if self.visualize:
            self.update_vis()

        return agent_actions,agent_probs

    def experiment(self):

        for i in range(1):
            self.update()
            print(self.no_items)
            if i==5:
                pdb.set_trace()
            # time.sleep(.4)
        print()
        return




    def update_foodconsumption(self):
        agents_around = []

        #Check how far each agent is from each item by a numpy array manipulation.
        agents_relative_positions  = np.array([self.item_pos_array-agent.curr_position for agent in self.agents]) #Array holding agents' relative positions with respect to each of the objects.
        agents_relative_distances = np.linalg.norm(agents_relative_positions,axis=2)

        #no_of agents surrounding each item
        is_agent_adjacent = agents_relative_distances<=1
        no_surrounding_agents = np.sum(is_agent_adjacent,axis=0)
        is_consumable = no_surrounding_agents>0

        #Fixme

        potentially_consumable_items = [[self.items[i],i] for consumable,i in zip(is_consumable,range(self.no_items)) if consumable]#List of items and their indexes that

        #are consumable
        potentially_consumable_items_indices = [item[1] for item in potentially_consumable_items]
        potentially_consuming_agents = [[agent for (agent,is_adjacent) in zip(self.agents,is_agent_adjacent[:,i]) if is_adjacent]for i in potentially_consumable_items_indices]
        #list of agents that could consume potential items ordered according to the previous list of items that could be consumed.

        #Criteria for load operation being on
        # consumable_criteria_1 = np.array([True if np.all(np.array([agent.load for agent in probable_consumers])) else False for probable_consumers in potentially_consuming_agents])

        #Criteria for sum of capacities of agents in 'load' mode being greater than weight of the item
        consumable_criteria = np.array([True if sum([agent.capacity for agent in potentially_consuming_agents[i] if agent.load])>probably_consumed_item[0].weight else False for i,probably_consumed_item in enumerate(potentially_consumable_items)])


        # consumable_allpass = np.all((consumable_criteria_1,consumable_criteria_2),axis=0)
        items_to_consume = [item for consumable,item in zip(consumable_criteria,potentially_consumable_items) if consumable]

        #now that we have all items to be consumed, eliminate them off the grid.
        for item,i in items_to_consume:
            item_loc = item.position
            self.grid_matrix[item_loc[0],item_loc[1]] = 0
            self.consumed_items.append([item,item_loc])
            self.items.remove(item)
            self.no_items-=1
            self.get_item_posarray()
            print("item_consumed")
        return

    def update_vis(self):
        self.get_agent_posarray()
        orientations = np.array([agent.curr_orientation for agent in self.agents])
        self.update_event = lvlvis.pygame.event.Event(self.visualizer.update_event_type,{'food_matrix': self.grid_matrix,'agents_positions':self.agent_pos_array,'agents_orientations':orientations})
        lvlvis.pygame.event.post(self.update_event)
        # self.visualizer.snapshot(str(time.time()))

    def copy(self):
        """
        Copy the existing arena and return a new object with exactly the same state. (It should be replaceable)
        :return: A new arena object."""

        # all_attrs = self.__dict__.keys()

        # def is_variable(item):
        #     t = type(item)
        #     if t is np.ndarray
        #     is types.IntType or is types.BooleanType or is types.LongType or is types.FloatType:
        #         return True
        #     else:
        #         return False

        cpd = deepcopy
        gm_c = cpd(self.grid_matrix)
        # self.update_mod_gridmatrx()
        # modgm_c = cpd(self.mod_gridmatrix)
        iposarray_c = cpd(self.item_pos_array)
        # aposarray_c = cpd(self.agent_pos_array)
        nagents_c = cpd(self.no_agents)
        nitems_c = cpd(self.no_items)


        #Removing methods:
        # all_variables = [attr for attr in all_attrs if is_variable(self.__dict__[attr])]
        # all_variables.remove('visualizer')
        # all_variables.remove('agents')
        # all_variables.remove('items')



        # new_dict_variables = {variable_name:copy.deepcopy(self.__dict__[variable_name]) for variable_name in all_variables}
        # ndv = new_dict_variables
        new_arena = arena(gm_c,False)
        new_arena.no_items = nitems_c
        new_arena.no_agents = nagents_c
        # new_arena.mod_gridmatrix = modgm_c
        # new_arena.agent_pos_array = aposarray_c
        new_arena.item_pos_array = iposarray_c

        agents_new_objects = [agent.copy(new_arena) for agent in self.agents]
        new_arena.add_agents(agents_new_objects)


        #1) Agents' state is preserved
        #2) Grid_matrix is preservered
        #3) Visualization need not be carried
        #4)
        # new_arena.__dict__.update(new_dict_variables)

        return new_arena

    def hash_action(self,action):
        """
        :action: n_agents length array of actions.

        possible actions for each agent are:
        1) move up - u
        2) move down - d
        3) move right -r
        4) move left - l
        5) load - a

        :return: no_agents sized string.
        """
        hash_action= ''
        for act in action:
            hash_action+=self.action_hashes[act]
        return hash_actio

    def unhash_action(self,hashed_action):
        """

        :param hashed_action: str of hashed actions
        :return: vector of actions
        """

        action_vector = []
        for char in hashed_action:
            action_vector.append(self.dict_actionIndex[char])
        return np.array(action_vector)

    def react(self,action):
        real_action = self.unhash_action(action)
        real_action = real_action[-1] #actionprob of the last MCTS agent.

        action_probs = np.zeros(5)
        action_probs[real_action]=1

        action_and_consequence = self.mcts_agent.behave_act(action_probs)
        self.mcts_agent.execute_action(action_and_consequence)

        pre_length = len(self.items)

        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()
        reward = len(self.items)-pre_length
        self.check_for_termination()

        if self.visualize:
            self.update_vis()

        #we return action itself because we assume a unique action-state relationship.
        #and it seems pointless to hash a state and unhash it
        return reward, action

    def act(self):
        agent_actions = []

        # retrieve what the agent wants to do
        for agent in self.agents:
            # Check what the agent wants to do
            action_probs = agent.behave()

            # retrieve the action.
            agent_action = agent.behave_act(action_probs)

            agent_actions.append(agent_action)

            # Approve the agent's action. This way, if agent moves further and is in
            # collision path with another agent, then this is not going to be aproble
            # as the other agent will plan accordingly.
            agent.execute_action(agent_action)

        pre_length = len(self.items)
        # See if there is any load operation.
        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()

        self.check_for_termination()

        reward = len(self.items) - pre_length
        if self.visualize:
            self.update_vis()

        #state just needs to be something unique.
        return reward, str(reward)


    def act_external(self,action):

        n_agents = len(self.agents)-1
        real_action = self.unhash_action(action)
        agent_actions = []
        for i in range(n_agents):
            # Check what the agent wants to d
            agent = self.agents[i]
            action_probs = np.ones(5)
            action_probs[real_action[i]]=1

            # retrieve the action.
            agent_action = agent.behave_act(action_probs)

            agent_actions.append(agent_action)

            # Approve the agent's action. This way, if agent moves further and is in
            # collision path with another agent, then this is not going to be aproble
            # as the other agent will plan accordingly.
            agent.execute_action(agent_action)

        pre_length = len(self.items)
        # See if there is any load operation.
        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()

        self.check_for_termination()

        reward = len(self.items) - pre_length
        if self.visualize:
            self.update_vis()

        # state just needs to be something unique.
        return reward, str(reward)

    def check_for_termination(self):
        if len(self.items)==0:
            self.isterminal = True
        else:
            self.isterminal = False
        return

    def get_legalActions(self,turn):

        if turn is False: #universe's turn
            validactionString_list = []
            for agent in self.agents[0:-1]: #exclude MCTS agent
                validactionProb = agent.get_legalActionProbs()
                valid_actions = np.argwhere(validactionProb!=0).reshape(-1)
                validAction_string = ''
                for action in valid_actions:
                    validAction_string+=self.action_hashes[action]
                validactionList.append(validAction_string)
            validactionString_list.append('n')
            all_actionStringVectors = itertools.product(*validactionString_list)
            for i in range(len(validactionString_list)):
                validactionString_list[i]=''.join(validactionString_list[i])
            return validactionString_list
        else:
            validactionString_list=[]
            default_string = ''.join(['n' for i in range(self.no_items - 1)])
            validactionProb = self.mcts_agent.get_legalActionProbs()
            valid_actions = np.argwhere(validactionProb!=0).reshape(-1)
            for action in valid_actions:
                validactionString_list.append(default_string+self.action_hashes[action])

            return validactionString_list

        ##Yet to finish

    def get_legalActions_N(self,turn):
        if turn in False: #universe's turn
            tot_actions = 1
            for agent in self.agents[0:-1]:
                validactionProb = agent.get_legalActionProbs()
                n_actions = np.sum(validactionProb!=0)
                tot_actions*=n_actions
            return tot_actions

        else:
            agent = self.agents[-1]
            return np.sum(agent.get_legalActionProbs()!=0)


    def get_legalAction_random(self,turn):
        action_string =''
        if turn is False: #universe's turn
            for agent in self.agents[0:-1]:
                validactionProb = agent.get_legalActionProbs()
                randomaction = np.random.choice(np.where(validactionProb!=0)[0])
                action_string+=self.action_hashes[randomaction]
            action_string+='n'
        else:
            action_string+=sum(['n' for i in range(self.no_agents-1)])
            validactionProb=self.mcts_agent.get_legalActionProbs()
            randomaction = np.random.choice(np.where(validactionProb!=0)[0])
            action_string+=self.action_hashes[randomaction]
        return action_string

