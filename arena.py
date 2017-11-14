import numpy as np
import threading
import matplotlib.pyplot as plt
from copy import copy
import levelbasedforaging_visualizer as lvlvis


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


class arena():
    def __init__(self,grid_matrix,visualize):
        self.grid_matrix = grid_matrix
        self.agents = []
        self.items = []
        self.visualize = visualize
        self.create_objectitems()




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
        self.agents = agents_list
        self.no_agents = len(self.agents)
        if self.visualize:
            agent_parameters = [agent.params for agent in self.agents]
            self.visualizer = lvlvis.LVDvisualizer(self.grid_matrix,agent_parameters)
            self.visualize_thread = threading.Thread(target=self.visualizer.wait_on_event)
            self.visualize_thread.start()


    def create_objectitems(self):
        items_loc = np.argwhere(self.grid_matrix>0)
        for loc in items_loc:
            item_obj = item(loc,self.grid_matrix[loc[0],loc[1]])
            self.items.append(item_obj)
        self.no_items = len(self.items)
        self.get_item_posarray()


    def update(self):
        agents_around = []

        #Check how far each agent is from each item by a numpy array manipulation.
        agents_relative_positions  = np.array([self.item_pos_array-agent.curr_position for agent in self.agents]) #Array holding agents' relative positions with respect to each of the objects.
        agents_relative_distances = np.linalg.norm(agents_relative_positions,axis=2)

        #no_of agents surrounding each item
        is_agent_adjacent = agents_relative_distances<=np.sqrt(2)
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
            self.items.remove(item)
        return

    def update_vis(self):
        self.get_agent_posarray()
        orientations = np.array([agent.curr_orientation for agent in self.agents])
        self.update_event = lvlvis.pygame.event.Event(self.visualizer.update_event_type,{'food_matrix': self.grid_matrix,'agents_positions':self.agent_pos_array,'agents_orientations':orientations})
        lvlvis.pygame.event.post(self.update_event)



