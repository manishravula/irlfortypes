import numpy as np
import types
import matplotlib.pyplot as plt
#from Algorithms.AStar import astar as astar_nav
from AStar import astar as ast
from copy import copy
from copy import deepcopy
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
        # type: (object, object, object, object, object, object, object) -> object
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
        # self.grid_matrix = self.arena.grid_matrix
        self.curr_destination = None
        self.curr_position = np.array(curr_pos)
        self.curr_position_realaxis = (-1,1)*self.curr_position
        self.memory = self.curr_position
        self.curr_orientation = curr_heading

        self.visible_agents = []
        self.visible_items = []
        self.action_probability = .25*np.ones(4)
        #up,down,right,left


        self.dict_moves_actionsProbs = {'-10':[1,0,0,0],'10':[0,1,0,0],'01':[0,0,1,0],'0-1':[0,0,0,1]} #If the key is the difference between dest and curr,
        #The list returns the action probs.
        self.actions = np.arange(5)
        self.dict_actiontoIndices = {'-10':0,'10':1,'01':2,'0-1':3} #Get the action index given the desired movement.

        self.action_to_movements = np.array([[-1,0],[1,0],[0,1],[0,-1],[0,0]]) #Given an action index, this array gives us the vector
        # to add to current states to get the result
        self.action_to_orientations = np.array([np.pi/2,1.5*np.pi,0,np.pi]) #Given an action index, this array gives us the
        # what the orientation should be.

        #need to init curr_position by reading from the foraging_arena's grid
        #action probabilities are [up,down,left,right]


    def get_outerandinnerAngles(self):
        self.outerangles = (self.curr_orientation+(self.view_angle/2))%(2*np.pi)
        self.innerangles = (self.curr_orientation-(self.view_angle/2))%(2*np.pi)


    def behave(self):
        """

        :return: actionprobabilites to sample the action from.
        """
        #sets the probabilities of actions after every loop of 'behaving'
        #Is an implementation of the behavior part of the type
        self.curr_position = np.array(self.curr_position)
        loc = self.curr_position
        self.curr_destination = None

        if self.memory!=None and np.any(self.curr_position!=self.memory):
            self.curr_destination = self.memory
        else:
            visible_entities = self.get_visibleAgentsAndItems()
            target = self.choosetarget(visible_entities)
            if target is not None:
                self.curr_destination = target
        self.memory = self.curr_destination

        #assign action probabilities
        if self.curr_destination is None:
            self.load=False
            self.action_probability = self.valid_randActionProb()
        else:
            if ((self.arena.grid_matrix[self.curr_destination[0],self.curr_destination[1]]) and (np.linalg.norm(self.curr_position-self.curr_destination) <= 1)):
                self.action_probability = np.hstack((np.zeros(4),1))
            else:
                #we need to make a copy grid matrix to pass to astar

                mod_grid_matrix = deepcopy(self.arena.grid_matrix)
                for a in self.arena.agents:
                    #This is for the agents to be treated as hard-obstacles
                    mod_grid_matrix[a.curr_position[0],a.curr_position[1]]=1
                mod_grid_matrix[self.curr_destination[0],self.curr_destination[1]]=0
                mod_grid_matrix[self.curr_position[0],self.curr_position[1]]=0
                self.astar = ast.astar(mod_grid_matrix,self.curr_position,self.curr_destination,False)
                del(mod_grid_matrix)
                path = self.astar.find_minimumpath()
                if len(path) is 0:
                    #Meaning the astar algorithm didn't find the path. Then just move about randomly.
                    # action_probabilites=np.array([1,1,1,1,0])/4.0
                    action_probabilites = self.valid_randActionProb()
                else:
                    to_move = path[0]-self.curr_position
                    action_probs = self.dict_moves_actionsProbs[str(to_move[0])+str(to_move[1])]
                    self.action_probability = np.hstack((action_probs,0))
        print('For agent with '+str(self.capacity)+' the destination is '+str(self.curr_destination))
        return action_probs

    def behave_act(self,action_probs):
        """
        The method picks up an action from the action probabilites and executes the action.
        :param action_probs:  action probabilities to behave according to
        :return action_and_consequence: the selected action and consequence of such an action.
        """
        final_action = np.random.choice(self.actions,1,p=action_probs)[0]
        final_movement = self.action_to_movements[final_action]
        final_nextposition = self.curr_position+final_movement
        action_and_consequence = [final_action,final_movement,final_nextposition]
        return(action_and_consequence)

    def valid_randActionProb(self):
        #Can optimize by replacing this with single array operations
        mod_grid_matrix = np.copy(self.arena.grid_matrix)
        mod_grid_matrix = np.lib.pad(mod_grid_matrix,1,'constant',constant_values=1)
        valid_actionProb = []
        currloc = self.curr_position
        for diff in self.action_to_movements:
            new_loc = currloc+diff+np.array([1,1])
            #Good thing about this is that the load action will always get zero prob
            #which is what we want when we are selecting random movements.
            if mod_grid_matrix[new_loc[0],new_loc[1]]:
                valid_actionProb.append(0)
            else:
                valid_actionProb.append(1.0)
        valid_actionProb = np.array(valid_actionProb)/np.sum(valid_actionProb)
        return valid_actionProb


    def execute_action(self,action_and_consequence):
        #The action is approved by the arena, and is being executed by the agent.
        #We can only move if it is not a load action.
        [final_action,final_movement,final_nextposition]=action_and_consequence
        if final_action!=4:
            self.curr_orientation = self.action_to_orientations[final_action]
            self.arena.grid_matrix[self.curr_position[0],self.curr_position[1]]=0
            self.curr_position = final_nextposition
            self.arena.grid_matrix[self.curr_position[0],self.curr_position[1]]=1
        else:
            #if this is a load action, this is probably already taken care of, by the arena.
            #Turn towards the item
            self.load = True
            to_move = self.curr_destination-self.curr_position
            action_index = self.dict_actiontoIndices[str(to_move[0])+str(to_move[1])]
            orientation = self.action_to_orientations[action_index]
            self.curr_orientation = orientation
            # self.load = False
        return

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
        # self.arena.agents = self.arena.agents
        # self.arena.grid_matrix = self.arena.grid_matrix


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
                    return dest
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
                    self.type = 1
                    dest = self.choosetarget(visible_entities)

                    #Restoring
                    self.curr_position = copy(curr_position)
                    self.type = curr_type

                    return dest

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
            lighter_item_index = np.where(item_capacities==np.max(item_capacities[item_capacities<self.capacity]))
            return visible_items[lighter_item_index[0]]
        else:
            return visible_items[np.argmax(item_capacities)]


    def get_furthestAgent(self,visible_agents):
        positions_list = np.array([agent.curr_position for agent in visible_agents])
        distances = np.linalg.norm(positions_list-self.curr_position,axis=1)
        farthest_agent = visible_agents[np.argmax(distances)]
        return farthest_agent


    def get_highestAgentBelowSelf(self,visible_agents):
        agent_capacities = np.array([agent.capacity for agent in visible_agents])
        if np.any(agent_capacities>self.capacity):
            desired_index = np.where(agent_capacities==np.max(agent_capacities[agent_capacities>self.capacity]))
            return visible_agents[desired_index[0]]
        else:
            return None

    def copy(self,new_arena):
        """
        new_arena is the new arena object we need to preserve the state.
        Copy the current agent with the exact state.
        :return: A new agent object, without the underlying arena, but still replaceable.
        """

        # all_attrs = dir(self)
        cd = deepcopy
        cp_c = cd(self.capacity)
        vr_c = cd(self.view_radius)
        va_c = cd(self.view_angle)
        load_c = cd(self.load)
        type_c = cd(self.type)
        currdest_c = cd(self.curr_destination)
        currpos_c = cd(self.curr_position)
        currposreal_c = cd(self.curr_position_realaxis)
        mem_c = cd(self.memory)
        currorien_c = cd(self.curr_orientation)
        ap_c = cd(self.action_probability)



        # all_variables = [attr for attr in all_attrs if not type(attr.item)==types.MethodType()]
        # all_variables.remove('arena')

        # new_dict_variables = {variable_name:copy.deepcopy(self.__dict__[variable_name]) for variable_name in all_variables}
        # ndv = new_dict_variables
        # new_object = agent(ndv['capacity'],ndv['view_radius'],ndv['view_angle'],ndv['type'],ndv['curr_pos'],ndv['curr_heading'],
        #                    new_arena)
        new_agent = agent(cp_c,vr_c,va_c,type_c,currpos_c,currorien_c,new_arena)
        new_agent.curr_destination = currdest_c
        new_agent.memory = mem_c
        new_agent.action_probability = ap_c
        new_agent.curr_position_realaxis = currposreal_c
        new_agent.load = load_c

        return new_agent

    def get_legalActions(self):
        validrand_actionprob = self.valid_randActionProb()

        #load action too
        validrand_actionprob[-1]=1

        #normalize probability.
        no_validactinons = np.sum(validrand_actionprob!=0)
        validrand_actionprob[np.where(validrand_actionprob!=0)]=1.0/no_validactinons
        return validrand_actionprob








        #Removing methods:


