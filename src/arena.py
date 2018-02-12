import numpy as np
# import time
import threading
from copy import copy
from copy import deepcopy
import pdb
from src import levelbasedforaging_visualizer as lvlvis
import itertools
import seaborn as sns
import logging
logger = logging.getLogger(__name__)


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

ACTION2INDEX = {'u':0,'d':1,'l':2,'r':3,'a':4}
ACTIONHASHES =  [action for action in ACTION2INDEX.iterkeys()]


class item():
    def __init__(self,position,weight):
        self.position = position
        self.weight = weight
    def copy(self):
        return item(deepcopy(self.position),deepcopy(self.weight))
    def __eq__(self, other):
        try:
            if np.all(self.position == other.position) and self.weight == other.weight:
                return True
            else:
                return False
        except KeyError:
            #Means some wrong object was passed.
            logger.debug("Requested an equality comparision of {} with an item-instance".format(other))
            return False


class arena():
    def __init__(self,grid_matrix,visualize):
        self.grid_matrix = grid_matrix
        self.agents = []
        self.items = []
        self.visualize = visualize

        self.actions = np.arange(5)
        self.isterminal = False
        self.astar_computed_dest = []
        self.astar_computed_path = []


    def init_add_agents(self,agents_list):

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

        self.init_build_itemObjects()

        # for agent in self.agents:
        #     _ = agent.get_visibleAgentsAndItems() #Setting state through init.

    def init_build_itemObjects(self):
        self.items=[]
        items_loc = np.argwhere(
            np.logical_and(self.grid_matrix > 0, self.grid_matrix < 1))  # agents' positions are identified by ones
        for loc in items_loc:
            item_obj = item(loc, self.grid_matrix[loc[0], loc[1]])
            self.items.append(item_obj)
        self.no_items = len(self.items)
        self.build_itemPositionArray()


    def build_itemPositionArray(self):
        posarray = []
        for item in self.items:
            posarray.append(item.position)
        return np.array(posarray)

    def build_agentPositionArray(self):
        posarray = []
        for agent in self.agents:
            posarray.append(agent.curr_position)
        return np.array(posarray)


    def update(self):
        agent_actions = []
        agent_probs = []

        #retrieve what the agent wants to do
        for agent in self.agents:
            #Check what the agent wants to do
            action_probs = agent.behave(False)

            #retrieve the action.
            agent_action = agent.behave_act(action_probs)
            assert np.all(action_probs == agent.action_probability); "Behave_act shouldn't change the action probability"
            agent_actions.append(agent_action)
            agent_probs.append(action_probs)

            #Approve the agent's action. This way, if agent moves further and is in
            #collision path with another agent, then this is not going to be aproble
            #as the other agent will plan accordingly.
            agent.execute_action(agent_action)
            assert np.all(action_probs==agent.action_probability); "Execute action shouldn't change the action probability"


        #See if there is any load operation.
        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()

        if self.visualize:
            self.update_vis()

        return agent_actions,agent_probs


    def update_foodconsumption(self):
        agents_around = []

        #Check how far each agent is from each item by a numpy array manipulation.
        item_pos_array = self.build_itemPositionArray()
        agents_relative_positions  = np.array([item_pos_array-agent.curr_position for agent in self.agents]) #Array holding agents' relative positions with respect to each of the objects.
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
            item_index = self.items.index(item)

            self.grid_matrix[item_loc[0],item_loc[1]] = 0
            self.items.remove(item)
            self.no_items-=1
            print("item_consumed at {}".format(item.position))
        self.check_for_termination()
        return

    def update_vis(self):
        agent_pos_array = self.build_agentPositionArray()
        orientations = np.array([agent.curr_orientation for agent in self.agents])
        self.update_event = lvlvis.pygame.event.Event(self.visualizer.update_event_type,{'food_matrix': self.grid_matrix,'agents_positions':agent_pos_array,'agents_orientations':orientations})
        lvlvis.pygame.event.post(self.update_event)
        # self.visualizer.snapshot(str(time.time()))


    def __getstate__(self):
        cp = deepcopy
        dict_state = {}
        dict_state['grid_matrix'] = cp(self.grid_matrix)
        dict_state['no_agents'] = cp(self.no_agents)
        dict_state['no_items'] = cp(self.no_items)
        return cp(dict_state)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.init_build_itemObjects() #Required when agent uses arena.items to check its surroundings.



    def check_for_termination(self):
        if len(self.items)==0:
            self.isterminal = True
        else:
            self.isterminal = False
        return



    def __eq__(self, other):
        #Compare if two arenas are equal
        #Get all variables first.
        allVars = self.__dict__
        logger.debug("Comparision request between {} and {} \n".format(self,other))

        compResult = []
        excludeKeys = ['agents','mcts_agent']
        for key in allVars.keys():
            if 'visua' not in key and key not in excludeKeys:
                currEle = self.__dict__[key]
                if isinstance(currEle,list):
                    try:
                        if currEle[0] is not None:
                            #Now we are comparing lists of items. This could be problematic when they are not in-order.
                            #We should compare ordered representation.
                            itemList1 = currEle
                            itemList2 = other.__dict__[key]

                            #O(N2) search. Just compare everything with everything.
                            compresult = True
                            for ele in itemList1:
                                found = False
                                for ele2 in itemList2:
                                    if ele==ele2:
                                        found = True
                                        break
                                if found is not True: #Even if one of the element is not found, then break.
                                    compresult = False
                                break
                        else:
                            compresult = self.__dict__[key] == other.__dict__[key]
                    except IndexError:
                        #Empty list
                        if len(other.__dict__[key])>0:
                            #If the other object's list isn't empty
                            compresult = False
                        else:
                            compresult = True

                else:
                    compresult = self.__dict__[key]==other.__dict__[key]

                logger.debug("Comparision of {} is {} \n".format(key,compresult))
                compResult.append(compresult)

        overallResult = [np.all(np.array(ele)) for ele in compResult]
        logger.debug("Overall comaprision result is {}".format(np.all(overallResult)))
        return np.all(overallResult)






