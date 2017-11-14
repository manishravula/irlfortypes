import numpy as np
import matplotlib.pyplot as plt
#from Algorithms.AStar import astar as astar_nav
from Algorithms.AStar import astar as ast
from copy import copy


#Ref doc:
"""
Input to the agent class:
    1) Agent's parameters:  
        capacity [0,1] scaled.
        viewcone_radius [.1,1] scaled - p2*[grid_size]
        viewcone_angle [.1,1] scaled - p3*2pi
    2) Type specification - 0/1/2/3
    3) Arena object - Holds details about the 
        a) Current food in the foraging domain.
        b) Positions of other agents in the domain.
        NOTE: The inital arena object doesn't really have any other agent related data.
        Only after the agent objects are instantiated, will they be attached to a variable
        inside the arena object.

Output data:
    1) get_position()
    2) get_params()
SimFuncs:
    1) self.next_step()
        Needs to move one step forward in simulation time basing on its type and 
        the current state of the arena.
InternalFuncs:
    1) self.visibleAgentsAndItems(curr_position)
    2) self.choosetarget(visible agents and items)
    3) self.get_path(curr_position, curr_dest) - get path from the Astar algorithm.
    
TODO:

    1) Create an arena class
    2) Create a driver between arena and plot visualization
    3) Connect the MCTS player to the arena
    4) Create a sim class. 
    
DEBUG TEST:
    1) Arena and the agents are interconnected. So once arena is done, embed the plots and let them play
        a) Let one agent play. See if the item-related behavior is all right.
        b) Let one agent play with another static agent without items. See if the sole-agent related behavior works.
        c) Add static agent and static item. See if they work together.
        d) Dynamic type-agent and static item. This should work.
    2) MCTS player creation can be slightly hard.
        a) For the arena, you need to add functions that the MCTS solver uses. 
        b) Test using the known-type distribution with one-agent system. Just give the type and see how it plays.
        c) Testing MCTS is the toughest part. Something that can help.
              i) Create a function for MCTS visualization. See what the tree state at each point is. 
              ii) See if it works well enough at each state.
"""


class agent():
    def __init__(self,capacity,view_radius,view_angle,type,curr_pos,curr_heading,foraging_arena):
        """

        :param capacity:  The capacity of the agent
        :param view_radius: View radius of the agent in pixels
        :param view_angle: View angle of the agent in pixels
        :param foraging_arena: The foraging arena object, to know where the food and other agents are.
        """
        self.capacity = capacity
        self.view_radius = view_radius
        self.view_angle = view_angle
        self.params = [self.capacity,self.view_radius,self.view_angle]
        self.load = False
        self.type = type
        self.arena = foraging_arena
        self.grid_matrix = self.arena.grid_matrix
        self.curr_destination = None
        self.curr_position = curr_pos
        self.curr_position_realaxis = (-1,1)*self.curr_position
        self.memory = self.curr_position
        self.curr_orientation = curr_heading

        self.visible_agents = []
        self.visible_items = []
        self.action_probability = .25*np.ones(4)

        #need to init curr_position by reading from the foraging_arena's grid
        #action probabilities are [up,down,left,right]

    def get_outerandinnerAngles(self):
        self.outerangles = (self.curr_orientation+(self.view_angle/2))%(2*np.pi)
        self.innerangles = (self.curr_orientation-(self.view_angle/2))%(2*np.pi)

    def behave(self):
        #sets the probabilities of actions after every loop of 'behaving'
        #Is an implementation of the behavior part of the type
        loc = self.curr_position
        self.curr_destination = None

        if self.memory and self.curr_position!=self.memory:
            self.curr_destination = self.memory
        else:
            visible_entities = self.get_visibleAgentsAndItems()
            target = self.choosetarget(visible_entities)
            if target:
                self.curr_destination = target
        self.memory = self.curr_destination

        #assign action probabilities
        if not self.curr_destination:
            return 0.25*np.ones(4)
        else:
            if ((self.grid_matrix[self.curr_destination[0],self.curr_destination[1]]) and (np.linalg.norm(self.curr_position-self.curr_destination) <= np.sqrt(2))):
                self.load = True
                self.action_probability = np.zeros(4)
            else:
                mod_grid_matrix = self.arena.mod_gridmatrix
                mod_grid_matrix[self.curr_position[0],self.curr_position[1]]=0
                self.astar = ast.astar(mod_grid_matrix,self.curr_position,self.curr_destination,False)
                path = self.astar.find_minimumpath()
                self.curr_destination = path[0].loc

    def get_position(self):
        return copy(self.curr_position)
    def get_params(self):
        return copy(self.params)

    def get_visibleAgentsAndItems(self):
        items_list = self.arena.items
        agents_list = self.arena.agents

        items_locarray = np.array([item.position for item in items_list])
        items_is_visible = self.is_visible(items_locarray)
        items_visible = [item for (item,is_in) in zip(items_list,items_is_visible) if is_in]

        agents_locarray = np.array([agent.curr_position for agent in agents_list])
        agents_is_visible = self.is_visible(agents_locarray)
        agents_visible = [agent for (agent,is_in) in zip(agents_list,agents_is_visible) if is_in]

        return [agents_visible, items_visible]


    def is_visible(self,loc_array):
        distance_list = np.linalg.norm(loc_array-self.curr_position, axis=1)
        direction_vectors = loc_array-self.curr_position
        angle_vectors = np.arctan2(0-direction_vectors[:,0],direction_vectors[:,1])%(2*np.pi) #Compensate for numpy and real axis diff

        constraint1 = distance_list<self.view_radius
        self.get_outerandinnerAngles()
        constraint2 = np.all((angle_vectors<self.outerangles,angle_vectors>self.innerangles),axis=0)
        return np.all((constraint1,constraint2),axis=0)


    def choosetarget(self,visible_entities):
        """
        :params: uses self.visible_agents and self.visible_items
        :return:
        """
        [self.visible_agents,self.visible_items] = visible_entities
        self.allagents = self.arena.agents
        self.grid_matrix = self.arena.grid_matrix


        if self.type==0:
            if self.visible_items:
                return (self.get_furthestItem(self.visible_items)).position
            else:
                return None
        elif self.type==1:
            if self.visible_items:
                return (self.get_highestItemBelowSelf(self.visible_items)).position
            else:
                return None
        elif self.type==2:
            if self.visible_agents:
                furthest_agent = self.get_furthestAgent(self.visible_agents)
                if self.visible_items:
                    #Saving
                    curr_position = copy(self.curr_position)
                    curr_type = copy(self.type)

                    #transitioning
                    self.curr_position = furthest_agent.curr_position
                    self.type = 0
                    dest = self.choosetarget(visible_entities)

                    #restoring
                    self.curr_position = copy(curr_position)
                    self.type = curr_type
                    return dest.curr_position
                else:
                    return furthest_agent.curr_position
            else:
                return None
        elif self.type==3:
            if self.visible_agents:
                highest_agent = self.get_highestAgentBelowSelf(self.visible_agents)
                if highest_agent:
                    dest = highest_agent
                else:
                    dest = self.get_furthestAgent(self.visible_agents)

                if self.visible_items:
                    #saving
                    curr_position = copy(self.curr_position)
                    curr_type = copy(self.type)

                    #transitioning
                    self.curr_position = copy(dest.curr_position)
                    dest = self.choosetarget(visible_entities)

                    #Restoring
                    self.curr_position = copy(curr_position)
                    self.type = curr_type

                    return dest.curr_position

                return dest.curr_position
            else:
                return None

    def get_furthestItem(self,visible_items):
        #According to type 1's furthest item
        distances = np.array([np.linalg.norm(item.position-self.curr_position) for item in visible_items])
        farthest_item = visible_items[np.argmax(distances)]
        return farthest_item

    def get_highestItemBelowSelf(self,visible_items):
        #According to type 2's highest item def
        item_capacities = np.array([item.weight for item in visible_items])
        if np.any(item_capacities<self.capacity):
            return visible_items[np.argmax(item_capacities[item_capacities<self.capacity])]
        else:
            return visible_items[np.argmax(item_capacities)]

    def get_furthestAgent(self,visible_agents):
        distances = np.array([np.linalg.norm(agent.curr_position-self.curr_position)])
        farthest_agent = visible_agents[np.argmax(distances)]
        return farthest_agent

    def get_highestAgentBelowSelf(self,visible_agents):
        agent_capacities = np.array([agent.capacity for agent in visible_agents])
        if np.any(agent_capacities>self.capacity):
            return visible_agents[np.argmax(agent_capacities[agent_capacities>self.capacity])]
        else:
            return None
















