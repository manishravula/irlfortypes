# Algorithm


# Aim decide next move when you are in state-t

# Algo

# Examine children, if you have best children,

"""
REF DOC
MCTS has to be refactored.
1) State is when you make a move.
2) You have to keep track of all states. - states can be remembered as dictionaries or hashtables

"""

""""""

"""
What should MCTS work on?

1) A 'WORLD' doing this:
    a) You take an action. The world does something too. Now it is your turn to take action again.
    b) Everytime you take some action and (the world does something too) and then you land yourself in a new states,
       the world spits out a reward.
2) Programmatically, what should a 'WORLD' look like.
    a) Given a current state, it should give us all possible actions. S->A func
    b) Given an action on a current state, it should gives us all possible new-states that you pushed the world into. T(S,A)
    c) Given an action on a current state, it should give us rewards(BANDIT context) for that particular transition. R(S,A)


3) Overview of the algorithm in the newfound MCTS design.
    a) Each node in the tree is a world-state coupled with MCTS information of UCT related stuff (Expected reward, N_tries).
    b) Expected reward comes from sum(rewards_allpaths)/N_paths simulated starting at that node.
    c) N_tries comes from all the time this particular node has been tried out.
    d) After we start-off at current_state, we do:


4) Each run of the simulation:
    a) We sample a type. 
    b) We retrieve the best parameter estimate of the type.
    c) We run MCTS for given number of iterations and stop to pick the best action:
        i) Initialize a new gym in the world with the current state of the world in the simulator.
        ii) Create a new node object and add statedef from the world object to this.
        ii) Start building the tree with the starting node as the current state.
        iii) Create a dictionary of state-nodeobj pairs. Add initial state to it.
        iii) While loop until specified iters or time.
            1) Pick a action according to UCT (greatest)
            2) Transition into a new state - retrieve the stateconfig from the gym
            3) Search if the new_state has already been encountered. If it was: 
            2) Collect discounted reward at each state according to time from now. (Absolute discount doesn't matter as we compare)
            3) For each state in the trajectory beginning from the end:
                a) 
"""

# DESIGN: USING GRAPH TOOL NOW BECAUSE
# Design: 1) We need to store stuff
# design 2) We need speed and can't compromise on those intializations and search and loads.


import numpy as np
import time
import random
import matplotlib.pyplot as plt
from operator import attrgetter
import random
from graph_tool.all import *
import pdb
from copy import copy, deepcopy

global C
C = np.sqrt(2)
external_player = 0
internal_agent = 1

UNIVERSE = False
AIAGENT = True

"""
New design of the MCTS

Expectations from the Universe:

1) Provide a method to copy itself, and return a new instance.
2) Provide a convenient access to adding a new action.
3) Provide a list of legal actions when asked.
4) Provide information about resulting state and reward obtained after we give our action
5) When it itself acts, should return the action it took, and the state that the action resulted and the reward obtained from it.
6) A means to name (hash) a state.
7) A way to know if a state is terminal or not.
8) A means to call on the universe to act and react.


The greatest improvement comes in considering the classes as state machines.

Methods:
 1) u.copy()
    Returns: A new instance of the universe with exact state.
    
 2) u.accept_action(valid_action)
    Notes: Accepts action valid_action from an external agent and updates itself.
    Returns: The new-state that this action resulted in, and the consequent reward.
 
 3) u.get_legalActions()
    Returns: Legal actions possible from its current internal state.
    
 4) u.act()
    Returns: action it took, resulting state, and the corresponding reward

MCTS methods:
 0) self.train(universe)
    Performs n rollouts at the current state of the universe.  
    It is assumed that the MCTS is at the same state as the universe.

 1) self.rollout(copy of universe)
    Perform its thing. Rollouts, and backups. Have enough to decide when asked for.
    Returns: Nothing
 
 2) self.act()
    Decide the best action being in the current state.
    Returns: action - pertaining to the universe's actions    
 
 3) self._resetState(state_hash)
    Traverse the tree backwards until you reach the state as described. Used in getting back to the state after going in the simulation phase.
    Returns: Nothing. Resets internal state.
    
 4) self.__terminated__
    Flag to indicate terimnation. If this is True, then the environment has reached a terminal state. Else, we can still go forward.

 5) self.follow(action,state,reward)
    Follow the action that the universe/environment took, and traverse the tree accrodingly.


"""





class mcts():
    def __init__(self, name='Default', visualize=False):


        self.discount = .95
        self.C = 1.95

        # Graph properties
        self.graph = Graph()
        # gname = self.graph.new_graph_property("string")
        # self.graph.properties["config"] = gname
        # self.graph.properties["config"] = name

        # Vertex Properties
        vp_stateKey = self.graph.new_vertex_property("string")  # Key to identify, lookup states
        self.graph.vp.state_key = vp_stateKey

        vp_avgreward = self.graph.new_vertex_property("float")
        self.graph.vp.avg_reward = vp_avgreward

        vp_cumreward = self.graph.new_vertex_property("float")
        self.graph.vp.cum_reward = vp_cumreward

        vp_reward = self.graph.new_vertex_property("float")
        self.graph.vp.reward = vp_reward

        vp_uct = self.graph.new_vertex_property("float")
        self.graph.vp.uct = vp_uct

        vp_nsims = self.graph.new_vertex_property("int")
        self.graph.vp.nsims = vp_nsims

        vp_turn_whose = self.graph.new_vertex_property("boolean")
        self.graph.vp.turn_whose = vp_turn_whose

        if visualize:
            vp_label = self.graph.new_vertex_property("string")
            self.graph.vp.label = vp_label

            vp_color = self.graph.new_vertex_property("string")
            self.graph.vp.color = vp_color

        # Edge properties
        ep_reward = self.graph.new_edge_property("float")
        self.graph.edge_properties.reward = ep_reward

        ep_action = self.graph.new_edge_property("string")
        self.graph.edge_properties.action = ep_action

        if visualize:
            ep_label = self.graph.new_edge_property("string")
            self.graph.edge_properties.label = ep_label

        # Dict to remember the conversion between index of the vertex in the graph and stateKey
        # This holds the dict of agent turn vertices
        # self.dict_stateKeyIndex_agent = {}
        # this holds the dict of universe turn vertices
        # self.dict_stateKeyIndex_universe = {}

    def addVertex(self, stateKey, turn):
        v = self.graph.add_vertex()
        self.graph.vp.state_key[v] = stateKey
        self.graph.vp.reward[v] = 0
        self.graph.vp.avg_reward[v] = 0
        self.graph.vp.cum_reward[v] = 0
        self.graph.vp.nsims[v] = 0
        self.graph.vp.uct[v] = np.inf
        self.graph.vp.turn_whose[v] = turn

        return v

    def hash_state(self, state):
        #The current state of mcts is given by the vertex node we are in, and the state-hash value.
        #This always has to be matched with the state of the universe we are handling.
        return state

    def unhash_state(self, hashed_state):
        # design: same as above
        # state1Darray = np.fromstring(hashed_state,self.stateArrayType)
        # return state1Darray.reshape(self.stateArrayShape)

        return hashed_state

    #
    # def create_graph(self):
    #     self.vertex_list = []
    #     self.edge_list = []

    def visualize_props(self):
        aiagent = 'turquoise'
        universe = 'sienna'
        win = 'green'
        lose = 'red'

        vertices = self.graph.get_vertices()
        # tag = '''<TITLE>Node Shapes</TITLE>'''

        for vertex in vertices:
            state = int(self.graph.vp.state_key[vertex])
            label_tag = self.world.get_stateDisplay(state)
            self.graph.vp.label[vertex] = label_tag
            if self.graph.vp.turn_whose[vertex]:
                self.graph.vp.color[vertex] = aiagent
            else:
                self.graph.vp.color[vertex] = universe

        return

    def rollout(self, curr_env,root_index):

        """
        :param curr_env:  The environment where the MCTS agent is called to act. We make
                          a copy of this object (courtesy of the .copy() method that the
                          object must provide), and use it to serve as a playground/gym
                          for our rollouts. Each rollout will require a different copy,
                          which is identical to a court in a playground/gym/stadium. The MCTS
                          agent will 'play' in this court until the game ends (a rollout). And then
                          it will create a new court from the master copy, and begin to play again.

        :return: Nothing. It just plays to learn.
        """

        backprop_info= []
        reward_info=[]


        world = curr_env.copy()

        begin_state = self.curr_state

        curr_state = deepcopy(begin_state)
        curr_stateIndex = self.curr_stateIndex
        begin_stateIndex = deepcopy(curr_stateIndex)

        backprop_info.append(curr_stateIndex)


        expandable_nodeIndex, bp_info, rw_info = self.select_expandableNode(curr_stateIndex,world)

        backprop_info+=bp_info
        reward_info+=rw_info

        #now select a new-node for expansion.
        legalactions = world.get_legalActions(self.graph.vp.turn_whose[expandable_nodeIndex])

        existingEdges = self.graph.get_out_edges(curr_stateIndex)
        existingExploredActions = [self.graph.edge_properties.action[edge[1]] for edge in existingEdges]

        for action in legalactions:
            if action in existingExploredActions:
                pass
            else:
                explorable_action = action
                break

        expansion_edge = existingEdges[existingExploredActions.index(explorable_action)]
        # explorable_realAction = world.unhash_action(explorable_action)

        #we found the action to take to move to the next state
        #now we act on it.
        curr_turnwhose = self.graph.vp.turn_whose[curr_stateIndex]
        if curr_turnwhose is AIAGENT:
            #so the agent is the one performing the action.
            reward, new_state = world.react(explorable_action)

        else:
            #the universe is performing this action.
            reward, new_state = world.act_external(explorable_action)




        newstate_vertex = self.addVertex(new_state,UNIVERSE)
        newstateIndex = self.graph.vertex_index[newstate_vertex]
        e = self.graph.add_edge(curr_stateIndex,newstateIndex)
        self.graph.edge_properties.reward[e]=reward
        self.graph.edge_properties.action[e]=explorable_action


        curr_stateIndex = newstateIndex
        curr_stateVertex = newstate_vertex

        # backprop_info.append(curr_stateIndex)
        reward_info+=reward


        #simulation stage. Go until you reach the terminal state

        rewardList_sim = []
        if not world.isterminal:
            #the expansion node is not terminal
            while not world.isterminal:
                turn_whose = self.graph.vp.turn_whose[curr_stateIndex]
                if turn_whose is AIAGENT:
                    random_action = world.get_legalAction_random()
                    r,next_state = world.react(random_action)
                    rewardList_sim.append(r)
                    turn_whose = UNIVERSE
                else:
                    r,next_state = world.act()
                    turn_whose = AIAGENT


            totalReward_simulation = 0
            sim_length = len(rewardList_sim)
            for i in range(sim_length):
                totalReward_simulation +=np.power(self.discount,i)*np.array(rewardList_sim[sim_length-i-1])

            #update the UCT of the expanded node.
            self.update_UCT(newstate_vertex,totalReward_simulation)
        else:
            #we don't want any simluation forward on the terminal node.
            totalReward_simulation = 100000
            self.update_UCT(newstate_vertex,totalReward_simulation)


        #backprop

        #we backprop only over the pre-expansion part.

        #backprop list in the order root-->newnode - hence we need to reverse it.
        backpropList = backprop_info.reverse()

        #rewards from newnode---r[-1]--->expandablenode----->2ndRoot-->r[0]-->root - hence we need to reverse it.
        rewardList = reward_info.reverse()


        discount = 1
        value_of_child = totalReward_simulation #this is the value of the child node of the expandable node,
        #where we start our backprop
        for nodeIndex,reward in zip(backpropList,rewardList):
            value_of_child+=(reward+self.discount*value_of_child)
            self.update_UCT(nodeIndex,value_of_child)

        #resetting the state
        self.curr_state = begin_state
        self.curr_stateIndex = begin_stateIndex

        return



    def update_UCT(self, stateIndex, reward):
        self.graph.vp.nsims[stateIndex] += 1
        self.graph.vp.cum_reward[stateIndex] += reward
        avg_reward = self.graph.vp.cum_reward[stateIndex] / self.graph.vp.nsims[stateIndex]
        self.graph.vp.avg_reward[stateIndex] = avg_reward

        parentIndex = self.graph.get_in_neighbors(stateIndex)[0]
        parent_nsims = self.graph.vp[parentIndex]

        uct = avg_reward + self.C * np.sqrt(((np.log(parent_nsims+1)) / self.graph.vp.nsims[stateIndex]))
        self.graph.vp.uct[stateIndex] = uct
        return



    def select_expandableNode(self,rootIndex,world):
        """

        :param rootIndex: Index of the root node at which the expansion begins. This is where the MCTS begins.
        :param world: The world object which we are using for current workout.
        :return exp_index: The index of the node which is expandable (i.e. whose children are not yet explored)
        :return backprop_info: list of indices travelled to reach the expandable node.
        :return reward_info: rewards obtained through each of the above transitions.
        """

        backprop_info=[]
        reward_info=[]
        EXPANDABLE = False
        currIndex = deepcopy(rootIndex)
        while not EXPANDABLE:
            #Check if the number of children node = number of legal actions.
            n_children = self.graph.get_out_degrees([currIndex])[0]

            turn_whose = self.graph.vp.turn_whose[currIndex]

            n_actions = world.get_legalActions_N(turn_whose)

            if n_children!=n_actions:
                EXPANDABLE = True

            else:
                #move to the next node by picking up the highest UCT, and then move the unvierse as well.
                action,childIndex = self.select_UCTNode(currIndex)

                #moving mcts to the next node
                currIndex = childIndex

                #moving world to the next state.
                # action_real = world.unhash(action)

                if turn_whose is AIAGENT:
                    reward,_ =self.world.react(action)
                else:
                    reward,_ = self.world.act_external(action) ##
                #todo:we can check if the action reward returned matches with ours at the edge.

                backprop_info.append(currIndex)
                reward_info.append(reward)

        return currIndex, backprop_info, reward_info





    def select_UCTNode(self,parentIndex):
        children = self.graph.get_out_neighbors(parentIndex)
        children_uct = [self.graph.vp.uct[child] for child in children]

        maxuct_childIndex = children[children_uct.index(max(children_uct))]
        maxuct_action = self.graph.edge(parentIndex,maxuct_childIndex)

        return maxuct_action,maxuct_childIndex








