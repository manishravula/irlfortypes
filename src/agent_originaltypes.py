import numpy as np
# from Algorithms.AStar import astar as astar_nav
from AStar import astar as ast
from copy import copy
from copy import deepcopy
import math

import logging
# Ref doc:
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

logger = logging.getLogger(__name__)

DICT_MOVES2ACTIONPROBS = {'-10': [1, 0, 0, 0, 0], '10': [0, 1, 0, 0, 0], '01': [0, 0, 1, 0, 0],
                                '0-1': [0, 0, 0, 1, 0],
                                '00': [0, 0, 0, 0, 1]}  # If the key is the difference between dest and curr,
# The list returns the action probs.
ACTIONS = np.arange(5)
DICT_ACTION2INDEX = {'-10': 0, '10': 1, '01': 2, '0-1': 3,
                             '00': 4}  # Get the action index given the desired movement.
DICT_INDEX2ACTION = {0: '-10', 1: '10', 2: '01', 3: '0-1', 4: '00'}

ACTION2MOVEMENTVECTOR = np.array(
    [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])  # Given an action index, this array gives us the vector
# to add to current states to get the result
ACTION2ORIENTATION = np.array(
    [np.pi / 2, 1.5 * np.pi, 0, np.pi])  # Given an action index, this array gives us the


# what the orientation should be.

class Agent():
    def __init__(self, capacity_param, viewRadius_param, viewAngle_param, type, curr_pos, foraging_arena):
        """

        :param capacity_param:  Paramter indicating capacity of the agent
        :param viewRadius_param: Parameter related to view radius - [0.1 - 1]*grid_size = viewradius
        :param viewAngle_param: Parameter related to view angel - [0.1 -1]*2pi = viewangle
        :param type:
        :param curr_pos:
        :param curr_heading:
        :param foraging_arena:
        """
        self.param_vector = np.array([capacity_param, viewRadius_param, viewAngle_param, type])
        self.capacity_param = capacity_param
        self.capacity = self.capacity_param

        self.viewAngle_param = viewAngle_param
        self.view_angle = self.viewAngle_param * np.pi * 2
        self.load = False
        self.type = type
        self.arena = foraging_arena
        self.grid_matrix_size = np.shape(self.arena.grid_matrix)[0]  # assuming it is a square

        self.viewRadius_param = viewRadius_param
        self.view_radius = self.viewRadius_param * self.grid_matrix_size
        self.curr_destination = None
        self.curr_position = np.array(curr_pos)
        self.memory = self.curr_position
        self.curr_orientation = np.random.random() * np.pi * 2

        self.params = [self.capacity, self.view_radius, self.view_angle]
        self.action_probability = .25 * np.ones(4)
        # up,down,right,left


        logger.info('Created an agent with location {}, type {}, capacity {}, view_radius {}, view_angle {} in standard ranges'.format(self.curr_position, self.type, self.capacity_param,self.viewRadius_param,self.viewAngle_param))


    @classmethod
    def create_from_param_vector(cls, param_vector, curr_pos, arena):
        """

        :param param_vector: capacity, radius, angle, type
        :param curr_pos: position
        :param arena: arena
        :return:  agent object
        """
        param_vector = np.array(param_vector)
        return cls(param_vector[0], param_vector[1], param_vector[2], param_vector[3], curr_pos, arena)

    def calc_path(self):
        mod_grid_matrix = deepcopy(self.arena.grid_matrix)
        for a in self.arena.agents:
            # This is for the agents to be treated as hard-obstacles
            mod_grid_matrix[a.curr_position[0], a.curr_position[1]] = 1
        mod_grid_matrix[self.curr_destination[0], self.curr_destination[1]] = 0
        mod_grid_matrix[self.curr_position[0], self.curr_position[1]] = 0
        self.astar = ast.astar(mod_grid_matrix, self.curr_position, self.curr_destination, False)
        del (mod_grid_matrix)
        path = self.astar.find_minimumpath()
        self.arena.astar_computed_dest.append(self.curr_destination)
        self.arena.astar_computed_path.append(path)
        return path

    def behave(self, is_dummy):
        """

        :return: actionprobabilites to sample the action from.
        """
        # sets the probabilities of actions after every loop of 'behaving'
        # Is an implementation of the behavior part of the type
        self.curr_position = np.array(self.curr_position)
        loc = self.curr_position
        self.curr_destination = None

        if self.memory is not None and np.any(self.curr_position != self.memory) and self.arena.grid_matrix[
            self.memory[0], self.memory[1]]:
            # we need to have something in memory, it shouldn't be where we are, and it should have something there.
            self.curr_destination = self.memory
        else:
            visible_entities = self.get_visibleAgentsAndItems()
            target = self.choosetarget(visible_entities)
            if target is not None:
                self.curr_destination = target
        self.memory = self.curr_destination

        # assign action probabilities
        legal_actionProbs_withLoad, legal_actionProbs_withoutLoad, valid_movesMask = self.valid_randMoveActionProb()

        if self.curr_destination is None:
            # random actions
            self.action_probability = legal_actionProbs_withLoad
        else:

            # legal_action_prob = self.get_legalActionProbs() #all possible legal actions will have non-zero probabilites in the expression.
            # if False and self.stagnant_count>self.maximum_stagnation:
            #     self.action_probability = legal_actionProbs_withoutLoad #If the agent has been stagnant for this long,then just give it random values to move around.

            if ((self.arena.grid_matrix[self.curr_destination[0], self.curr_destination[1]]) and (
                    np.linalg.norm(self.curr_position - self.curr_destination) <= 1)):
                # just load - no movement
                self.action_probability = DICT_MOVES2ACTIONPROBS['00']  # load action has 00 movement, so.
            else:
                # Chose an action that takes you towards your goal.

                resulting_pos = self.curr_position + ACTION2MOVEMENTVECTOR
                resulting_distances = np.linalg.norm(self.curr_destination - resulting_pos, axis=1)
                resulting_distances[
                    np.logical_not(valid_movesMask)] = np.inf  # Make chosing invalid actions impossible.

                actionProb = DICT_MOVES2ACTIONPROBS[DICT_INDEX2ACTION[np.argmin(resulting_distances)]]
                self.action_probability = actionProb

            # no 'valid'(actions that don't push the agent into boundaries) should be left with zero-probability.

            # first find all the valid actions, and check if any of these has zero probability in the action_prob vector we
            # currently have.

        # legal_action_prob=self.get_legalActionProbs()
        # optimize
        for i in range(len(self.action_probability)):
            if legal_actionProbs_withLoad[i] and not self.action_probability[i]:
                # means if the action is valid, but is assigned zero probability in action_probability,
                self.action_probability[i] += .01  # then assign it a minute non-zero probability
            if self.action_probability[i] and not legal_actionProbs_withLoad[i]:
                self.action_probability[i] = 0

        self.action_probability /= np.sum(self.action_probability)
        # if self.action_probability[4]==0:
        #     pass
        # ap = copy(self.action_probability)
        # if ap[4]==0:
        #     pass

        return copy(self.action_probability)

    def behave_act(self, action_probs):
        """
        The method picks up an action from the action probabilites and executes the action.
        :param action_probs:  action probabilities to behave according to
        :return action_and_consequence: the selected action and consequence of such an action.
        """
        try:
            final_action = np.random.choice(ACTIONS, 1, p=action_probs)[0]
        except ValueError:
            pass
        final_movement = ACTION2MOVEMENTVECTOR[final_action]
        final_nextposition = self.curr_position + final_movement
        action_and_consequence = [final_action, final_movement, final_nextposition]
        return (action_and_consequence)

    def valid_randMoveActionProb(self, debug=False):
        # Can optimize by replacing this with single array operations

        # valid_actionProb = []

        # currloc = self.curr_position
        # if debug:
        #     pdb.set_trace()
        # optimize
        final_pos = self.curr_position + ACTION2MOVEMENTVECTOR

        # start_time = time.time()
        mask1_1 = np.all(final_pos >= 0, axis=1)  # boundary condition check.
        mask1_2 = np.all(final_pos < self.grid_matrix_size, axis=1)  # right and bottom boundary check

        mask1 = np.logical_and(mask1_1, mask1_2)

        final_pos[np.logical_not(
            mask1)] = 0  # now checking cordinates that are inside bounds if they have an agent or item in them
        mask2 = self.arena.grid_matrix[final_pos.T[0], final_pos.T[
            1]] == 0  # no agent or item in the target position. The load action also gets nullified.
        valid_actions_mask = np.logical_and(mask1, mask2)

        valid_actions_mask = valid_actions_mask.astype('float')
        valid_actions_prob_without_load = valid_actions_mask / math.fsum(valid_actions_mask)

        valid_actions_mask[-1] = 1  # to consider load action too.
        valid_actions_prob_withLoad = valid_actions_mask / math.fsum(valid_actions_mask)

        valid_actions_mask[-1] = 0  # again to return valid movements only

        return valid_actions_prob_withLoad, valid_actions_prob_without_load, valid_actions_mask

    def get_legalActionProbs(self):
        validrand_actionprob = self.valid_randMoveActionProb(True)

        # load action too
        # validrand_actionprob[-1]=1
        #
        # normalize probability.
        # no_validactinons = np.sum(validrand_actionprob!=0)
        # validrand_actionprob[np.where(validrand_actionprob!=0)]=1.0/no_validactinons
        return validrand_actionprob

        # Removing methods:


    def execute_action(self, action_and_consequence):
        # The action is approved by the arena, and is being executed by the agent.
        # We can only move if it is not a load action.
        [final_action, final_movement, final_nextposition] = action_and_consequence
        curr_position = self.curr_position.copy()
        if final_action != 4:
            self.load = False
            self.curr_orientation = ACTION2ORIENTATION[final_action]
            self.arena.grid_matrix[self.curr_position[0], self.curr_position[1]] = 0
            self.curr_position = final_nextposition

            try:
                self.arena.grid_matrix[self.curr_position[0], self.curr_position[1]] = 1
            except IndexError:
                raise ("Tried to exceed boundaries")

        else:
            # if this is a load action, this is probably already taken care of, by the arena.
            # Turn towards the item
            self.load = True
            if np.any(self.curr_destination):
                to_move = (self.curr_destination - self.curr_position)
                if np.sum(np.abs(to_move)) < 2:  # Only align orientation if the item is near by.
                    action_index = DICT_ACTION2INDEX[str(to_move[0]) + str(to_move[1])]
                    orientation = ACTION2ORIENTATION[action_index]
                    self.curr_orientation = orientation
            # self.load = False


        return

    def get_position(self):
        return copy(self.curr_position)

    def get_params(self):
        return copy(self.params)

    def get_visibleAgentsAndItems(self, debug=False):

        items_list = self.arena.items
        agents_list = self.arena.agents

        items_locarray = np.array([item.position for item in items_list])
        items_is_visible = self.is_visible(items_locarray)
        visible_items = [item for (item, is_in) in zip(items_list, items_is_visible) if is_in]

        agents_locarray = np.array([agent.curr_position for agent in agents_list])

        agents_is_visible = self.is_visible(agents_locarray)
        visible_agents = [agent for (agent, is_in) in zip(agents_list, agents_is_visible) if is_in]

        return [visible_agents, visible_items]

    def is_visible(self, loc_array):

        distance_list = np.linalg.norm(loc_array - self.curr_position, axis=1)
        distance_list[distance_list == 0] = np.inf

        direction_vectors = loc_array - self.curr_position
        angle_vectors = np.arctan2(0 - direction_vectors[:, 0], direction_vectors[:, 1]) % (
                2 * np.pi)  # Compensate for numpy and real axis diff

        constraint1 = distance_list < self.view_radius
        loc_array_real = np.fliplr(loc_array)
        loc_array_real[:, 1] *= -1  # y axis is inverted.
        curr_position_realaxis = np.array([self.curr_position[1], -self.curr_position[0]])

        right_boundary_vector, left_boundary_vector= self.get_outerandinnerAngles()

        constraint2 = np.array([self.is_withinSector(loc,right_boundary_vector,left_boundary_vector,curr_position_realaxis) for loc in loc_array_real])
        return np.all((constraint1, constraint2), axis=0)

    def is_withinSector(self, target_loc,right_boundary_vector,left_boundary_vector,curr_position_realaxis):

        target_vector = np.array(target_loc - curr_position_realaxis)

        # Is the angle subtended between target and left most boundary, clockwise < 180?
        # is the angle subtended between target and right most boundary, anticlockwise < 180?

        left_normal_vector = np.array([0 - target_vector[1], target_vector[0]])
        right_normal_vector = -1 * left_normal_vector

        if self.view_angle <= np.pi:
            if (np.dot(left_boundary_vector, left_normal_vector) >= 0 and np.dot(right_boundary_vector,
                                                                                      right_normal_vector) >= 0):
                return True
            else:
                return False
        else:
            if (np.dot(right_boundary_vector, left_normal_vector) >= 0 and np.dot(left_boundary_vector,
                                                                                       right_normal_vector) >= 0):
                return False
            else:
                return True

    def get_outerandinnerAngles(self):
        outerangle = (self.curr_orientation + (self.view_angle / 2)) % (2 * np.pi)
        innerangle = (self.curr_orientation - (self.view_angle / 2)) % (2 * np.pi)


        right_boundary_vector = np.array([np.cos(innerangle), np.sin(innerangle)])
        left_boundary_vector = np.array([np.cos(outerangle), np.sin(outerangle)])
        return right_boundary_vector,left_boundary_vector

    def choosetarget(self, visible_entities):
        """
        :params: uses self.visible_agents and self.visible_items
        :return:
        """
        [visible_agents, visible_items] = visible_entities
        # self.arena.agents = self.arena.agents
        # self.arena.grid_matrix = self.arena.grid_matrix

        if self.type == 0:
            if visible_items:
                return (self.get_furthestItem(visible_items)).position
            else:
                return None
        elif self.type == 1:
            if visible_items:
                return (self.get_highestItemBelowSelf(visible_items)).position
            else:
                return None
        elif self.type == 2:
            if visible_agents:
                furthest_agent = self.get_furthestAgent(visible_agents)
                if visible_items:
                    # Saving
                    curr_position = copy(self.curr_position)
                    curr_type = copy(self.type)

                    # transitioning
                    self.curr_position = furthest_agent.curr_position
                    self.type = 0
                    dest = self.choosetarget(visible_entities)

                    # restoring
                    self.curr_position = copy(curr_position)
                    self.type = curr_type
                    return dest
                else:
                    return furthest_agent.curr_position
            else:
                return None
        elif self.type == 3:
            if visible_agents:
                highest_agent = self.get_highestAgentBelowSelf(visible_agents)
                if highest_agent:
                    dest = highest_agent
                else:
                    dest = self.get_furthestAgent(visible_agents)

                if visible_items:
                    # saving
                    curr_position = copy(self.curr_position)
                    curr_type = copy(self.type)

                    # transitioning
                    self.curr_position = copy(dest.curr_position)
                    self.type = 1
                    dest = self.choosetarget(visible_entities)

                    # Restoring
                    self.curr_position = copy(curr_position)
                    self.type = curr_type

                    return dest

                return dest.curr_position
            else:
                return None
        else:
            raise Exception("NO TYPE FOUND")

    def get_furthestItem(self, visible_items):
        # According to type 1's furthest item
        distances = np.array([np.linalg.norm(item.position - self.curr_position) for item in visible_items])
        farthest_item = visible_items[np.argmax(distances)]
        return farthest_item

    def get_highestItemBelowSelf(self, visible_items):
        # According to type 2's highest item def
        item_capacities = np.array([item.weight for item in visible_items])
        if np.any(item_capacities < self.capacity):
            lighter_item_index = np.where(item_capacities == np.max(item_capacities[item_capacities < self.capacity]))
            try:
                return visible_items[lighter_item_index[0]]
            except TypeError:
                #Happens sometimes. From Numpy version upgrade.
                return visible_items[lighter_item_index[0][0]]

        else:
            return visible_items[np.argmax(item_capacities)]

    def get_furthestAgent(self, visible_agents):
        positions_list = np.array([agent.curr_position for agent in visible_agents])
        distances = np.linalg.norm(positions_list - self.curr_position, axis=1)
        farthest_agent = visible_agents[np.argmax(distances)]
        return farthest_agent

    def get_highestAgentBelowSelf(self, visible_agents):
        agent_capacities = np.array([agent.capacity for agent in visible_agents])
        if np.any(agent_capacities > self.capacity):
            desired_index = np.where(agent_capacities == np.max(agent_capacities[agent_capacities > self.capacity]))
            try:
                return visible_agents[desired_index[0]]
            except TypeError:
                return visible_agents[desired_index[0][0]]
        else:
            return None


    def __getstate__(self):
        cd = deepcopy
        dict_state = {}
        dict_state['type'] = cd(self.type)
        dict_state['capacity_param'] = cd(self.capacity_param)
        dict_state['view_radius'] = cd(self.view_radius)
        dict_state['view_angle'] = cd(self.view_angle)
        dict_state['viewAngle_param'] = cd(self.viewAngle_param)
        dict_state['viewRadius_param'] = cd(self.viewRadius_param)

        dict_state['load']= cd(self.load)
        dict_state['curr_destination'] = cd(self.curr_destination)
        dict_state['memory'] = cd(self.memory)

        dict_state['curr_position'] = cd(self.curr_position)
        dict_state['curr_orientation'] = cd(self.curr_orientation)

        dict_state['action_probability'] = cd(self.action_probability)
        return dict_state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __eq__(self,other):
        #Method called when the equality operator is sought.
        logging.debug("Comparision called between {} and {} ".format(self,other))
        allVars = self.__dict__
        compResult = []
        exclude_keys = ['visible_agents','arena']

        for key in allVars.keys():
            if 'visua' not in key and key not in exclude_keys:
                currEle = self.__dict__[key]
                if isinstance(currEle,list):
                    try:
                        from src.arena import item
                        if isinstance(currEle[0],item):
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
        return np.all([np.all(np.array(ele)) for ele in compResult])









