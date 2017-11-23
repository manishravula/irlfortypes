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

global C
C = np.sqrt(2)
external_player = 0
internal_agent = 1

UNIVERSE = False
AIAGENT = True


# Todo: 1) Change all of these properties as graph-node's internal properties and add those functions as external methods
#
# class mcts_statenode():
#     def __init__(self,state,parent,childaction_array):
#         self.state = state
#         #Each state node is uniquely identified by this state definition. This could be a numpy array of the grid-world or whatever.
#         #The only requirement is that this is unique.
#         self.parent = parent
#         self.childaction_array = childaction_array
#         self.avg_reward = 0
#         self.cum_reward = 0
#         self.reward = 0
#         self.ucb = 0
#         self.n_sims = 0
#         self.turn_whose = AGENT
#
#         #The following two are tricky.
#         #First, we need to know how 'good' this position is, which is inferred from the first part of the UCT
#         #THis is calculated by a ratio of number of times this node played resulted in a win vs number of times this
#         #node is played at all
#         #The second part is to know how much this node is exploited. This is calculated by computing the total number of times
#         #simulations started at its parent node divided by the number of times this node particular node was traversed in the process.
#         self.n_sims = 0 #Number of simulations involving the current node.
#
#     def update_meanreward(self):
#         self.avg_reward = self.cum_reward/self.n_sims
#     def update_ucb(self):
#         self.update_meanreward()
#         self.ucb = self.avg_reward + C*np.sqrt(((np.log(self.parent.n_sims))/self.n_sims))
#         return



class mcts():
    def __init__(self, universe, number_of_playouts, name='Default', visualize=False):

        self.universe = universe

        # Todo: Hacky, fix this to mean better.
        self.world = universe
        self.number_of_playouts = number_of_playouts
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
        if visualize:
            ep_label = self.graph.new_edge_property("string")
            self.graph.edge_properties.label = ep_label

        # Dict to remember the conversion between index of the vertex in the graph and stateKey
        # This holds the dict of agent turn vertices
        self.dict_stateKeyIndex_agent = {}
        # this holds the dict of universe turn vertices
        self.dict_stateKeyIndex_universe = {}

    def addVertex(self, stateKey, turn):
        v = self.graph.add_vertex()
        self.graph.vp.state_key[v] = stateKey
        self.graph.vp.reward[v] = 0
        self.graph.vp.avg_reward[v] = 0
        self.graph.vp.cum_reward[v] = 0
        self.graph.vp.nsims[v] = 0
        self.graph.vp.uct[v] = 0
        self.graph.vp.turn_whose[v] = turn

        # Add v to the dictionary of the vertexindex - statekey pairs`
        if turn == UNIVERSE:
            self.dict_stateKeyIndex_universe[stateKey] = self.graph.vertex_index[v]
        else:
            self.dict_stateKeyIndex_agent[stateKey] = self.graph.vertex_index[v]
        return v

    def hash_state(self, state):
        # Custom hash function for the state.
        # if nothing is mentioned, by default use python's hash function on numpytostring function.

        # design: for our usecase, state is the numpy array of [locofmctsagent,locsofallotheragents,[heading,0]of all otheragents,
        # design: locsoffood].
        #
        # self.stateArrayShape = state.shape()
        # self.stateArrayType = state.dtype
        # return state.tostring()

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

    def rollout(self, state):

        # All rewards are interpreted as beneficial/adversarial for agent.
        # So a reward received in the node where it is the universe's turn, describes how good it is for the agent.

        self.world = self.universe.create_world()

        # design: we always start with our turn. The world is assumed to just have taken its turn, and then it is us.
        begin_state = state
        begin_stateKey = self.hash_state(begin_state)
        self.world.set_state(state)

        if not self.dict_stateKeyIndex_agent.has_key(begin_stateKey):
            self.addVertex(begin_stateKey, AIAGENT)

        # design: As we always begin first, we assume our start is after the world's action that resulted in world's curr_state
        curr_state = begin_state
        curr_stateKey = begin_stateKey
        curr_stateIndex = self.dict_stateKeyIndex_agent[curr_stateKey]
        curr_stateVertex = self.graph.vertex(curr_stateIndex)

        # list of list to hold information regarding backprop
        backprop_edgerewardlist = []

        i = 0
        print('Rollout- ---------------------------------------------------------- ')

        while not self.world.is_terminalstate(curr_state):
            start = time.time()
            print('While loop index is :' + str(i))
            i += 1
            j = 0

            # It is our turn
            self.graph.vp.turn_whose[curr_stateIndex] = AIAGENT

            # get legal actions from current state
            actions_legal = self.world.get_actionsLegal(curr_state)

            # List to hold valid states to transition into, as a one-to-one correspondence with actions.
            stateIndices_legal = []

            # Now retrieve the resulting state-objects.
            # This is a turn-by-turn simulation. We pick one action, obtain reward, store it, and ask the universe to do the same.
            # until we reach a terminal state.

            for action in actions_legal:
                print('Internal for loop index' + str(j))
                j += 1

                # Peek into a one-step future.
                next_state = self.world.react(action, curr_state, Transition=False)
                next_stateKey = self.hash_state(next_state)

                # See if the next-state has related node-objects already. If the node-objects don't exist, create them
                # otherwise, just associate them with our children_array.


                if not self.dict_stateKeyIndex_universe.has_key(next_stateKey):
                    self.addVertex(next_stateKey, UNIVERSE)

                next_stateIndex = self.dict_stateKeyIndex_universe[next_stateKey]

                # because it is always the case that the children node are universe's turn
                self.graph.vp.turn_whose[next_stateIndex] = UNIVERSE

                # Added directed edge from currstate to the child state
                _ = self.graph.add_edge(curr_stateIndex, next_stateIndex)  # to supress output

                stateIndices_legal.append(next_stateIndex)

            # notes: There is different strategy to be followed while exploring vs while actually playing the game.
            # notes: While exploring, we choose the next node on the following order"
            # notes: 1) If there are unexplored actions, pick one at random and explore it.
            # notes: 2) If there are no unexplored actions, pick the one with highest UCB.
            # notes: 3) If there are nodes with same UCB, resolve ties arbitrarily.

            # design: Pick the best action to explore.
            bestExploration_action, bestExploration_stateIndex = self.pick_actionBestExploration(actions_legal,
                                                                                                 stateIndices_legal)

            # Act on it and make the world react to your action
            _ = self.world.react(bestExploration_action, curr_state, Transition=True)

            bestExploration_stateKey = self.graph.vp.state_key[bestExploration_stateIndex]
            bestExploration_state = self.unhash_state(bestExploration_stateKey)

            # get the edge that connects the current state and the state that we are transitioning into
            exploreEdge = self.graph.edge(curr_stateIndex, bestExploration_stateIndex)

            # design: we add the reward as a property of the edge.
            reward = self.world.get_reward(curr_state, bestExploration_action, bestExploration_action)
            self.graph.edge_properties.reward[exploreEdge] = reward

            # remember this edge for use in backprop
            backprop_edgerewardlist.append([bestExploration_stateIndex, curr_stateIndex])

            # design: Update the properties of the newly formed node.
            # updates: nsim, cumreward, avgreward.
            self.update_newReward(bestExploration_stateIndex, reward)

            # ------------------------------------#
            # universe's turn#

            # design: Make the universe act on this current move of ours.
            worldactionReward = self.world.act(bestExploration_state, Transition=True)

            # design: After the world acts, make a new node in the graph to represent the transition and change the variables for the future.
            postWorld_state = self.world.get_state()
            postWorld_stateKey = self.hash_state(postWorld_state)

            # Retrieve the vertex node in the graph. If you don't have a vertex yet, create one accordingly.
            if not self.dict_stateKeyIndex_agent.has_key(postWorld_stateKey):
                postWorld_stateVertex = self.addVertex(postWorld_stateKey, AIAGENT)
                postWorld_stateIndex = self.graph.vertex_index[postWorld_stateVertex]
            else:
                postWorld_stateIndex = self.dict_stateKeyIndex_agent[postWorld_stateKey]
                postWorld_stateVertex = self.graph.vertex(postWorld_stateIndex)

            backprop_edgerewardlist.append([postWorld_stateIndex, bestExploration_stateIndex])
            self.update_newReward(postWorld_stateIndex, worldactionReward)

            # Design: Now add a new edge to the above to signify the transition of the world because of an action.
            transitionEdge = self.graph.add_edge(bestExploration_stateIndex, postWorld_stateIndex)
            self.graph.edge_properties.reward[transitionEdge] += worldactionReward

            print("In while loop transitioning from " + str(curr_state) + " " + str(postWorld_state))

            # prepare for the next iteration of the loop.
            curr_state = postWorld_state
            curr_stateKey = postWorld_stateKey
            curr_stateIndex = postWorld_stateIndex
            curr_stateVertex = postWorld_stateVertex
            print('Time for loop is ' + str(time.time() - start))

        # backpropagate the rewards.
        # design: After done with the forward simulation, go backwards and
        # start off with the terminal node, whose reward is given.


        # backprop is basically going through all the visited nodes, and updating the ucb values of visited nodes
        # + children nodes of visited nodes (because their parents' nsims changed)

        backprop_edgerewardlist.reverse()  # starting from reverse

        # Now update the UCT values
        for edge in backprop_edgerewardlist:
            # for each change visited state node, we need to update the child's UCT.
            # to update the child's uct, we need to first retrieve all the parents that the child has, and then
            # sum up their nsims to get the total number of times
            stateIndex_toUpdate = edge[0]  # The 'from' vertex.
            print("Updating: " + str(edge[0]))
            self.update_uct(stateIndex_toUpdate)
        # update UCT
        self.update_uct(0)

        print('backprop_length is' + str(len(backprop_edgerewardlist)))
        return

    def update_newReward(self, stateIndex, reward):
        self.graph.vp.nsims[stateIndex] += 1
        self.graph.vp.cum_reward[stateIndex] += reward
        self.graph.vp.avg_reward[stateIndex] = self.graph.vp.cum_reward[stateIndex] / self.graph.vp.nsims[stateIndex]
        return

    def update_uct(self, stateIndex):
        # Update UCT by computing new sum(parents' nsims)
        currVertex = self.graph.vertex(stateIndex)
        parents = self.graph.get_in_neighbors(currVertex)

        total_nsim = 0
        for parent in parents:
            total_nsim += self.graph.vp.nsims[parent]

        avg_reward = self.graph.vp.avg_reward[stateIndex]
        nsim = self.graph.vp.nsims[stateIndex]
        uct = avg_reward + self.C * np.sqrt(((np.log(total_nsim)) / nsim))
        self.graph.vp.uct[currVertex] = uct

    def pick_actionBestExploration(self, actions_legal, stateIndices_legal):
        # unexplored_stateIndices = [statenode if statenode.n_sims==0 else None for statenode in statenodes_legal]
        unexplored_stateIndices = [stateIndex for stateIndex in stateIndices_legal if
                                   not self.graph.vp.nsims[stateIndex]]
        if unexplored_stateIndices:
            # design: Means there are unexplored child-states, which is bad for UCB comparision, so go forward and pick them.
            next_stateIndex = random.choice(unexplored_stateIndices)
            next_action = actions_legal[stateIndices_legal.index(next_stateIndex)]
            return (next_action, next_stateIndex)
        else:
            # design: Means all the child-states are explored all-ready. Pick the highest UCB child state.
            # maxUCB_statenode = max(statenodes_legal,key=attrgetter('ucb'))
            uct_array = [self.graph.vp.uct[stateIndex] for stateIndex in stateIndices_legal]
            maxuct_index = uct_array.index(max(uct_array))
            maxUCB_stateIndex = stateIndices_legal[maxuct_index]
            maxUCB_action = actions_legal[maxuct_index]
            return (maxUCB_action, maxUCB_stateIndex)

    def pick_actionBestExploitation(self, actions_legal, stateIndices_legal):
        avgReward_array = [self.graph.vp.avg_reward[stateIndex] for stateIndex in stateIndices_legal]
        maxReward_index = avgReward_array.index(max(avgReward_array))
        maxReward_stateIndex = stateIndices_legal[maxReward_index]
        maxReward_action = actions_legal[maxReward_index]
        return (maxReward_action, maxReward_stateIndex)

    def act(self, state):
        # Here, given a state, we are asking the AI agent to pick an action and transition into a next state.

        # The universe has just taken the turn, so we retrieve its state.
        stateKey = self.hash_state(state)
        stateKeyIndex = self.dict_stateKeyIndex_universe(stateKey)
        stateKeyVertex = self.graph.vertex(stateKeyIndex)

        next_stateIndices = self.graph.get_out_neighbors(stateKeyVertex)
        next_actionsLegal = self.world.get_actionsLegal(state)
        best_action, best_nextStateIndex = self.pick_actionBestExploitation(next_actionsLegal, next_stateIndices)

        # select the state from the agent's turns' vertices only.
        best_nextState = self.unhash_state(self.dict_stateKeyIndex_agent(best_nextStateIndex))
        return best_action, best_nextState





