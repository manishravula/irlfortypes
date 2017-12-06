import numpy as np
import types
import matplotlib.pyplot as plt
#from Algorithms.AStar import astar as astar_nav
from AStar import astar as ast
from copy import copy
from copy import deepcopy
import math
import config_experiment as config
import time
import pdb
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


class Agent():
    def __init__(self,capacity_param,viewRadius_param,viewAngle_param,type,curr_pos,foraging_arena):
        """

        :param capacity_param:  Paramter indicating capacity of the agent
        :param viewRadius_param: Parameter related to view radius - [0.1 - 1]*grid_size = viewradius
        :param viewAngle_param: Parameter related to view angel - [0.1 -1]*2pi = viewangle
        :param type:
        :param curr_pos:
        :param curr_heading:
        :param foraging_arena:
        """
        self.param_vector = np.array([capacity_param,viewRadius_param,viewAngle_param,type])
        self.capacity_param = capacity_param
        self.capacity = self.capacity_param

        self.viewAngle_param = viewAngle_param
        self.view_angle = self.viewAngle_param*np.pi*2
        self.load = False
        self.type = type
        self.arena = foraging_arena
        self.grid_matrix_size = np.shape(self.arena.grid_matrix)[0] #assuming it is a square

        self.viewRadius_param = viewRadius_param
        self.view_radius = self.viewRadius_param*self.grid_matrix_size
        self.curr_destination = None
        self.curr_position = np.array(curr_pos)
        self.curr_position_realaxis = (-1,1)*self.curr_position
        self.memory = self.curr_position
        self.curr_orientation = np.random.random()*np.pi*2

        self.params = [self.capacity,self.view_radius,self.view_angle]
        self.visible_agents = []
        self.visible_items = []
        self.action_probability = .25*np.ones(4)
        #up,down,right,left


        self.dict_moves_actionsProbs = {'-10':[1,0,0,0,0],'10':[0,1,0,0,0],'01':[0,0,1,0,0],'0-1':[0,0,0,1,0],'00':[0,0,0,0,1]} #If the key is the difference between dest and curr,
        #The list returns the action probs.
        self.actions = np.arange(5)
        self.dict_actiontoIndices = {'-10':0,'10':1,'01':2,'0-1':3,'00':4} #Get the action index given the desired movement.
        self.dict_indicestoActions = {0:'-10',1:'10',2:'01',3:'0-1',4:'00'}

        self.action_to_movements = np.array([[-1,0],[1,0],[0,1],[0,-1],[0,0]]) #Given an action index, this array gives us the vector
        # to add to current states to get the result
        self.action_to_orientations = np.array([np.pi/2,1.5*np.pi,0,np.pi,self.curr_orientation]) #Given an action index, this array gives us the
        # what the orientation should be.

        #need to init curr_position by reading from the foraging_arena's grid
        #action probabilities are [up,down,left,right]
        self.pad_grid_matrix = np.zeros_like(self.arena.grid_matrix)
        self.pad_grid_matrix = np.lib.pad(self.pad_grid_matrix, 1, 'constant', constant_values=1)


    @classmethod
    def create_from_param_vector(cls,param_vector,curr_pos,arena):
        """

        :param param_vector: capacity, radius, angle, type
        :param curr_pos: position
        :param arena: arena
        :return:  agent object
        """
        param_vector = np.array(param_vector)
        return cls(param_vector[0],param_vector[1],param_vector[2],param_vector[3],curr_pos,arena)


    def calc_path(self):
        mod_grid_matrix = deepcopy(self.arena.grid_matrix)
        for a in self.arena.agents:
            #This is for the agents to be treated as hard-obstacles
            mod_grid_matrix[a.curr_position[0],a.curr_position[1]]=1
        mod_grid_matrix[self.curr_destination[0],self.curr_destination[1]]=0
        mod_grid_matrix[self.curr_position[0],self.curr_position[1]]=0
        self.astar = ast.astar(mod_grid_matrix,self.curr_position,self.curr_destination,False)
        del(mod_grid_matrix)
        path = self.astar.find_minimumpath()
        self.arena.astar_computed_dest.append(self.curr_destination)
        self.arena.astar_computed_path.append(path)
        return path

    def behave(self,is_dummy):
        """

        :return: actionprobabilites to sample the action from.
        """
        #sets the probabilities of actions after every loop of 'behaving'
        #Is an implementation of the behavior part of the type
        self.curr_position = np.array(self.curr_position)
        loc = self.curr_position
        self.curr_destination = None

        if self.memory is not None and np.any(self.curr_position!=self.memory) and self.arena.grid_matrix[self.memory[0],self.memory[1]]:
            #we need to have something in memory, it shouldn't be where we are, and it should have something there.
            self.curr_destination = self.memory
        else:
            visible_entities = self.get_visibleAgentsAndItems()
            target = self.choosetarget(visible_entities)
            if target is not None:
                self.curr_destination = target
        self.memory = self.curr_destination

        #assign action probabilities
        legal_actionProbs_withLoad,legal_actionProbs_withoutLoad,valid_movesMask = self.valid_randMoveActionProb()

        if self.curr_destination is None:
            #random actions
            self.action_probability = legal_actionProbs_withLoad
        else:

            # legal_action_prob = self.get_legalActionProbs() #all possible legal actions will have non-zero probabilites in the expression.
            if ((self.arena.grid_matrix[self.curr_destination[0],self.curr_destination[1]]) and (np.linalg.norm(self.curr_position-self.curr_destination) <= 1)):
                #just load - no movement
                self.action_probability = self.dict_moves_actionsProbs['00'] #load action has 00 movement, so.
            else:
                #Chose an action that takes you towards your goal.

                resulting_pos = self.curr_position+self.action_to_movements
                # if resulting_pos>self.grid_matrix_size:
                    # print("UEYES______")
                resulting_distances = np.linalg.norm(self.curr_destination-resulting_pos,axis=1)
                resulting_distances[np.logical_not(valid_movesMask)] = np.inf #Make chosing invalid actions impossible.

                actionProb = self.dict_moves_actionsProbs[self.dict_indicestoActions[np.argmin(resulting_distances)]]
                self.action_probability = actionProb


                #resulting_dist = np.linalg.norm(resulting_pos,0)
                #resulting_dist_possible = np.logical_and(resulting_dist,legal_action_prob) #If a block is stopped,

                #we need to find the next step to take.
                #we need to make a copy grid matrix to pass to astar
                # if is_dummy:
                #     try:
                #         pathIdx = self.arena.astar_computed_dest.index(self.curr_destination) #the path exists in memory
                #         path = self.arena.astar_computed_path[pathIdx]
                #     except ValueError:
                #         # the path doesnt exist in memory
                #         path = self.calc_path()
                # else:
                #     # this is not a dummy agent.
                #     path=self.calc_path()
                #
                # #
                # if len(path) is 0:
                #     #Meaning the astar algorithm didn't find the path. Then just move about randomly.
                #     # action_probabilites=np.array([1,1,1,1,0])/4.0
                #     action_probabilites = self.valid_randMoveActionProb(False)
                # else:
                #     to_move = path[0]-self.curr_position
                #     action_probs = self.dict_moves_actionsProbs[str(to_move[0])+str(to_move[1])]
                #     self.action_probability = np.hstack((action_probs,0.0)).astype('float')

            #no 'valid'(actions that don't push the agent into boundaries) should be left with zero-probability.

            #first find all the valid actions, and check if any of these has zero probability in the action_prob vector we
            #currently have.


        # legal_action_prob=self.get_legalActionProbs()
        #optimize
        for i in range(len(self.action_probability)):
            if legal_actionProbs_withLoad[i] and not self.action_probability[i]:
                #means if the action is valid, but is assigned zero probability in action_probability,
                self.action_probability[i]+=.01 #then assign it a minute non-zero probability

        self.action_probability /= np.sum(self.action_probability)
        # if self.action_probability[4]==0:
        #     pass
        # ap = copy(self.action_probability)
        # if ap[4]==0:
        #     pass

        return copy(self.action_probability)

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

    def valid_randMoveActionProb(self,debug=False):
        #Can optimize by replacing this with single array operations

        # valid_actionProb = []

        # currloc = self.curr_position
        # if debug:
        #     pdb.set_trace()
        #optimize
        final_pos = self.curr_position+self.action_to_movements

        # start_time = time.time()
        mask1_1 = np.all(final_pos>=0,axis=1) #boundary condition check.
        mask1_2 = np.all(final_pos<self.grid_matrix_size,axis=1) #right and bottom boundary check


        mask1 = np.logical_and(mask1_1,mask1_2)

        final_pos[np.logical_not(mask1)]=0 #now checking cordinates that are inside bounds if they have an agent or item in them
        mask2 = self.arena.grid_matrix[final_pos.T[0],final_pos.T[1]]==0 #no agent or item in the target position. The load action also gets nullified.
        valid_actions_mask = np.logical_and(mask1,mask2)


        valid_actions_mask = valid_actions_mask.astype('float')
        # if valid_actions_mask[-1]==0:
        #     pass
        valid_actions_prob_without_load = valid_actions_mask/math.fsum(valid_actions_mask)

        valid_actions_mask[-1] =1 #to consider load action too.
        valid_actions_prob_withLoad = valid_actions_mask/math.fsum(valid_actions_mask)

        valid_actions_mask[-1] = 0 #again to return valid movements only

        # time1 = time.time()

        # valid_actionProb=[]
        # currloc = self.curr_position
        # for diff in self.action_to_movements:
        #     curr_loc_diff = currloc+diff
        #     new_loc = curr_loc_diff+np.array([1,1])
        #
        #     #Good thing about this is that the load action will always get zero prob
        #     #which is what we want when we are selecting random movements.
        #     if self.pad_grid_matrix[new_loc[0],new_loc[1]] or self.arena.grid_matrix[curr_loc_diff[0],curr_loc_diff[1]]:
        #         valid_actionProb.append(0)
        #     else:
        #         valid_actionProb.append(1.0)
        # valid_actionProb = np.array(valid_actionProb)/math.fsum(valid_actionProb)
        #
        # time2 = time.time()
        # if np.all(valid_actions_prob_without_load==valid_actionProb):
        #     pass
        #     t1 = time1-start_time
        #     t2 = time2-time1
        #     # print(t2)
        #     # print(t1)
        #     # print("speed up achieved is "+str(t2/t1))
        # else:
        #     # print(valid_actions_prob_without_load)
        #     # print(valid_actionProb)
        #     print('Method failed')

        return valid_actions_prob_withLoad,valid_actions_prob_without_load, valid_actions_mask


    def execute_action(self,action_and_consequence):
        #The action is approved by the arena, and is being executed by the agent.
        #We can only move if it is not a load action.
        [final_action,final_movement,final_nextposition]=action_and_consequence
        if final_action!=4:
            self.load=False
            self.curr_orientation = self.action_to_orientations[final_action]
            self.arena.grid_matrix[self.curr_position[0],self.curr_position[1]]=0
            self.curr_position = final_nextposition
            try:
                self.arena.grid_matrix[self.curr_position[0],self.curr_position[1]]=1
            except IndexError:
                print("Tried to exceed boundaries")

        else:
            #if this is a load action, this is probably already taken care of, by the arena.
            #Turn towards the item
            self.load = True
            if np.any(self.curr_destination):
                to_move = (self.curr_destination-self.curr_position)
                if np.sum(np.abs(to_move))<2: #Only align orientation if the item is near by.
                    action_index = self.dict_actiontoIndices[str(to_move[0])+str(to_move[1])]
                    orientation = self.action_to_orientations[action_index]
                    self.curr_orientation = orientation
            # self.load = False
        return

    def get_position(self):
        return copy(self.curr_position)

    def get_params(self):
        return copy(self.params)

    def get_visibleAgentsAndItems(self,debug=False):

        items_list = self.arena.items
        agents_list = self.arena.agents

        items_locarray = np.array([item.position for item in items_list])
        # if debug:
        #     print self.curr_position
        #     print (items_locarray)
        #
        # if debug:
        items_is_visible = self.is_visible(items_locarray)
        # else:
        #     items_is_visible = self.is_visible(items_locarray,False)

        self.visible_items = [item for (item,is_in) in zip(items_list,items_is_visible) if is_in]

        agents_locarray = np.array([agent.curr_position for agent in agents_list])

        agents_is_visible = self.is_visible(agents_locarray)
        # if debug:
        #     print agents_is_visible
        self.visible_agents = [agent for (agent,is_in) in zip(agents_list,agents_is_visible) if is_in]

        return [self.visible_agents,self.visible_items]


    def is_visible(self,loc_array):


        distance_list = np.linalg.norm(loc_array-self.curr_position, axis=1)
        distance_list[distance_list==0]=np.inf


        direction_vectors = loc_array-self.curr_position
        angle_vectors = np.arctan2(0-direction_vectors[:,0],direction_vectors[:,1])%(2*np.pi) #Compensate for numpy and real axis diff
        # if debug:
        #     print('In is_visible')
        #     print direction_vectors,angle_vectors

        constraint1 = distance_list<self.view_radius
        self.get_outerandinnerAngles()
        loc_array_real = np.fliplr(loc_array)
        loc_array_real[:,1]*=-1 #y axis is inverted.
        # if debug:
            # print loc_array_real

        constraint2 = np.array([self.is_withinSector(loc) for loc in loc_array_real])
        # if debug:
        #     print constraint2
        #     print ("out of is_visible")
        return np.all((constraint1,constraint2),axis=0)

    def is_withinSector(self,target_loc):
        target_vector = np.array(target_loc-self.curr_position_realaxis)

        #Is the angle subtended between target and left most boundary, clockwise < 180?
        #is the angle subtended between target and right most boundary, anticlockwise < 180?

        left_normal_vector = np.array([0-target_vector[1],target_vector[0]])
        right_normal_vector = -1*left_normal_vector

        # if debug:
        #     print ("in is_withinSector")
        #     print target_vector
        #     print left_normal_vector
        #     print right_normal_vector
        #
        # if debug:
        #     print np.dot(self.left_boundary_vector,left_normal_vector)
        #     print np.dot(self.right_boundary_vector,right_normal_vector)
        #     print np.dot(self.right_boundary_vector, left_normal_vector)
        #     print np.dot(self.left_boundary_vector,right_normal_vector)
        #     print ("out of is_withinSector")

        if self.view_angle<=np.pi:
            if (np.dot(self.left_boundary_vector,left_normal_vector)>=0 and np.dot(self.right_boundary_vector,right_normal_vector)>=0):
                return True
            else:
                return False
        else:
            if (np.dot(self.right_boundary_vector, left_normal_vector) >= 0 and np.dot(self.left_boundary_vector,right_normal_vector) >= 0):
                return False
            else:
                return True



    def get_outerandinnerAngles(self):
        self.outerangle = (self.curr_orientation+(self.view_angle/2))%(2*np.pi)
        self.innerangle = (self.curr_orientation-(self.view_angle/2))%(2*np.pi)

        self.curr_position_realaxis = np.array([self.curr_position[1],-self.curr_position[0]])
        self.right_boundary_vector = np.array([np.cos(self.innerangle),np.sin(self.innerangle)])
        self.left_boundary_vector = np.array([np.cos(self.outerangle),np.sin(self.outerangle)])




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
        else:
            raise Exception("NO TYPE FOUND")

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

    def get_legalActionProbs(self):
        validrand_actionprob = self.valid_randMoveActionProb(True)

        #load action too
        # validrand_actionprob[-1]=1
        #
        # normalize probability.
        # no_validactinons = np.sum(validrand_actionprob!=0)
        # validrand_actionprob[np.where(validrand_actionprob!=0)]=1.0/no_validactinons
        return validrand_actionprob








        #Removing methods:


