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
from src.global_const import UNIVERSE, AIAGENT

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
from graph_tool.all import *
from copy import deepcopy
from MCTS.environment import environment
from experiments import configuration as config

global C
C = np.sqrt(2)
external_player = 0
internal_agent = 1

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
    def __init__(self, env_object, visualize=True):


        self.discount = .95
        self.C = 5.95
        self.env_object = env_object

        # Graph properties
        self.graph = Graph()
        # gname = self.graph.new_graph_property("string")
        # self.graph.properties["config"] = gname
        # self.graph.properties["config"] = name

        # Vertex Properties
        vp_stateKey = self.graph.new_vertex_property("string")  # Key to identify, lookup states
        self.graph.vp.state_key = vp_stateKey

        vp_nlegalactions = self.graph.new_vertex_property("int") #Number of legal actions permissible by the env at this state.
        self.graph.vp.nlegalactions = vp_nlegalactions

        vp_parentIndex = self.graph.new_vertex_property("int")
        self.graph.vp.parentIndex = vp_parentIndex

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

        self.rootVertex = self.addVertex('root', AIAGENT, len(self.env_object.getActions_legalFromCurrentState(
            AIAGENT)), -1)
        self.rootVertex_index = self.graph.vertex_index[self.rootVertex]  # [TODO: Fix an error here]
        self.currState_vertexIndex = deepcopy(self.rootVertex_index)
        self.currState_vertex = self.rootVertex

        # Dict to remember the conversion between index of the vertex in the graph and stateKey
        # This holds the dict of agent turn vertices
        # self.dict_stateKeyIndex_agent = {}
        # this holds the dict of universe turn vertices
        # self.dict_stateKeyIndex_universe = {}

    def addVertex(self, stateKey, turn, n_legalactions,parentIndex):
        """
        :param stateKey: The name/repr of state
        :param turn: Who should act now? If the turn is AIAGENT, it means that the Universe just acted
                    and resulted in this vertex, and now the agent should act.
        :param n_legalactions: Number of legal actions possible from this particular state.
                    We are saving this so that we don't have to request this info from the
                    Universe everytime.
        :param parentIndex: The index of the parent this node came from. We are saving this because
                    we will use it in backprop. This is -1 if the vertex is root.
        :return:
        """
        v = self.graph.add_vertex()
        self.graph.vp.state_key[v] = stateKey
        self.graph.vp.nlegalactions[v] = n_legalactions
        self.graph.vp.reward[v] = 0
        self.graph.vp.avg_reward[v] = 0
        self.graph.vp.cum_reward[v] = 0
        self.graph.vp.nsims[v] = 0
        self.graph.vp.uct[v] = np.inf
        self.graph.vp.turn_whose[v] = turn
        self.graph.vp.parentIndex[v] = parentIndex

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

    def rollout(self, env, root_index, rolloutdepth=config.MAX_ROLLOUT_DEPTH):

        """
        :param rolloutdepth:
        :param env:  The environment where the MCTS agent is called to act. We make
                          a copy of this object (courtesy of the .copy() method that the
                          object must provide), and use it to serve as a playground/gym
                          for our rollouts. Each rollout will require a different copy,
                          which is identical to a court in a playground/gym/stadium. The MCTS
                          agent will 'play' in this court until the game ends (a rollout). And then
                          it will create a new court from the master copy, and begin to play again.

        :return: Nothing. It just plays to learn.
        """

        chain_statesTraversed = []
        chain_rewards =[]

        curr_env = env

        # ------------INITIALIZATION
        begin_state = self.graph.vp.state_key[self.currState_vertex]
        curr_state = deepcopy(begin_state)

        curr_stateIndex = self.currState_vertexIndex
        begin_stateIndex = deepcopy(curr_stateIndex)

        #backprop_info.append(curr_stateIndex)

        #------------SELECTION------------
        expandable_nodeIndex, states_info, rewards_info = self.select_expandableNode(curr_stateIndex,curr_env)

        chain_statesTraversed.extend(states_info)
        chain_rewards.extend(rewards_info)


        #------------EXPANSION------------
        #Select an action to expand on.
        legalactions = curr_env.getActions_legalFromCurrentState(self.graph.vp.turn_whose[expandable_nodeIndex])
        existingEdges = self.graph.get_out_edges(expandable_nodeIndex)
        existingExploredActions = [self.graph.edge_properties.action[edge] for edge in
                                   existingEdges]  # edge properties ta
        #ake edge objects as input

        for action in legalactions:
            if action in existingExploredActions:
                pass
            else:
                explorable_action = action
                break


        #Create an node for the state that results on acting on this action.
        curr_turnwhose = self.graph.vp.turn_whose[expandable_nodeIndex]
        if curr_turnwhose == AIAGENT:
            #so the agent is the one performing the action.
            reward_newState, new_state = curr_env.respond(explorable_action)

        else:
            #the universe is performing this action.
            reward_newState, new_state = curr_env.act_externalwill(explorable_action)

        #Add both the node and the edge to the graph.
        curr_turnwhose = not curr_turnwhose
        newstate_vertex = self.addVertex(new_state, curr_turnwhose,
                                         curr_env.getNumberOfActions_legalFromCurrentState(curr_turnwhose),
                                         expandable_nodeIndex)
        newstateIndex = self.graph.vertex_index[newstate_vertex]
        e = self.graph.add_edge(expandable_nodeIndex, newstateIndex)
        self.graph.edge_properties.reward[e]=reward_newState
        self.graph.edge_properties.action[e]=explorable_action

        #Adding the new node info to the backprop related stuff.
        chain_statesTraversed.append(newstate_vertex)
        chain_rewards.append(reward_newState)

        #Setting precedent to the rest of the loop.
        curr_stateIndex = newstateIndex
        curr_stateVertex = newstate_vertex

        # backprop_info.append(curr_stateIndex)
        # reward_info.append(reward_newState)


        #------------SIMULATION------------
        #simulation stage. Go until you reach the terminal state

        rewardList_sim = []
        if not curr_env.isterminal:
            #the expansion node is not terminal
            turn_whose = self.graph.vp.turn_whose[curr_stateIndex]
            rolloutidx =0
            while not curr_env.isterminal and rolloutidx < rolloutdepth:
                if turn_whose == AIAGENT:
                    random_action = curr_env.getAction_randomLegalFromCurrentState(turn_whose)
                    r,next_state = curr_env.respond(random_action)
                    rewardList_sim.append(r)
                    turn_whose = UNIVERSE
                else:
                    r,next_state = curr_env.act_freewill()
                    rewardList_sim.append(r)
                    turn_whose = AIAGENT
                rolloutidx+=1

            totalReward_simulation = 0
            sim_length = len(rewardList_sim)
            #todo: make this more efficient. Don't need so many multiplications
            for i in range(sim_length):
                totalReward_simulation +=np.power(self.discount,i)*np.array(rewardList_sim[i])

            #update the UCT of the expanded node.
            # self.update_UCT(newstate_vertex,reward_newState+(self.discount*totalReward_simulation))
            #Regarding the above, The UCT of the new_state vertex is representative of the idea
            #that its parent has about it's expected reward. UCT is closely tied with it's parent.
            #In the eyes of the parent, reward+(self.discount*totalReward_simulation) is what this
            #child's worth is.
        else:
            #we don't want any simluation forward on the terminal node.
            totalReward_simulation = curr_env.getvalue_terminalState()
            # self.update_UCT(newstate_vertex,reward_newState+(self.discount*totalReward_simulation))




        #---------BACKPROP-----------
        #backprop list in the order root-->newnode - hence we need to reverse it.
        backpropChain_stateIndexes = chain_statesTraversed
        backpropChain_stateIndexes.reverse()

        # Rewards from newnode---r[-1]--->expandablenode----->2ndRoot-->r[0]-->root - hence we need to reverse it.
        backpropChain_rewards = chain_rewards
        backpropChain_rewards.reverse()

        #This is the value of the child node of the expandable node, where we start our backprop
        value_of_child = totalReward_simulation

        if len(backpropChain_rewards) != 0:
            # Backprop phase one until the currstate with which this rollout was called.
            for nodeIndex, reward in zip(backpropChain_stateIndexes, backpropChain_rewards):
                value_of_child += (reward + self.discount * value_of_child)
                self.increment_UCT(nodeIndex, value_of_child)

        # Incrementing root's nsims, because the backprop doesn't do it already.
        assert begin_stateIndex == 0;
        'Root is not the beginning state index'
        self.graph.vp.nsims[begin_stateIndex] += 1

        # self.update_UCT(self.rootVertex_index,value_of_child)

        #Backprop phase two from currstate with which the rollout was called to the root of the search tree.
        #NOT NEEDED?
        """
        parent = self.graph.vp.parentIndex[nodeIndex]
        if (parent>=0):
        """
        #resetting the state
        self.curr_state = begin_state
        self.curr_stateIndex = begin_stateIndex

        return

    def increment_UCT(self, stateIndex, reward):
        """
        When the banditbox's hinge has been pulled, a new reward is accumulated, and hence we need to
        Update it's value.
        :param stateIndex:
        :param reward:
        :return:
        """
        self.graph.vp.nsims[stateIndex] += 1
        self.graph.vp.cum_reward[stateIndex] += reward
        avg_reward = self.graph.vp.cum_reward[stateIndex] / self.graph.vp.nsims[stateIndex]
        self.graph.vp.avg_reward[stateIndex] = avg_reward


        parentIndex = self.graph.get_in_neighbors(stateIndex)[0]
        parent_nsims = self.graph.vp.nsims[parentIndex]

        uct = avg_reward + self.C * np.sqrt(((np.log(parent_nsims+1)) / self.graph.vp.nsims[stateIndex]))
        self.graph.vp.uct[stateIndex] = uct
        siblings_indices = self.graph.get_out_neighbors(parentIndex)

        for sibling_index in siblings_indices:
            if sibling_index != stateIndex:
                self.refresh_UCT_becauseOfParent(sibling_index, parent_nsims + 1)

        # Now we that we took care of ourselves, we need to take care of the siblings.
        #Since the parent's nsims has increased, the siblings' UCT value change.
        return

    def refresh_UCT_becauseOfParent(self, stateIndex, n_parentsims):
        """
        If a node's parent has been run-through, then the node's UCT also changes
        on account of the parent_nsims part of the UCT formula. Hence we need to update this.
        :param stateIndex:
        :return:
        """
        avg_reward = self.graph.vp.avg_reward[stateIndex]
        new_uct = avg_reward + self.C * np.sqrt(((np.log(n_parentsims)) / self.graph.vp.nsims[stateIndex]))
        self.graph.vp.uct[stateIndex] = new_uct


    def select_expandableNode(self,rootIndex,curr_env):
        """
        :param rootIndex: Index of the root node at which the expansion begins. This is where the MCTS begins.
        :param world: The world object which we are using for current workout.
        :return exp_index: The index of the node which is expandable (i.e. whose children are not yet explored)
        :return backprop_info: list of indices travelled to reach the expandable node.
        :return reward_info: rewards obtained through each of the above transitions.
        """

        '''
        Chain of nodes traversed,beginning from one-node below the root.
        Chain of rewards obtained for going into corresponding state in the above list.
        Forexample, the first element is the reward obtained when we transition from root to 
        the first state.
        '''

        chain_statesTraversed=[]
        chain_rewards=[]

        EXPANDABLE = False
        currIndex = deepcopy(rootIndex)
        while not EXPANDABLE:
            #Check if the number of children node = number of legal actions.
            n_children = self.graph.vertex(currIndex).out_degree()

            turn_whose = self.graph.vp.turn_whose[currIndex]

            #TODO: Replace this with the node property
            # n_actions = len(curr_env.getActions_legalFromCurrentState(turn_whose))
            n_actions = self.graph.vp.nlegalactions[currIndex]

            if n_children!=n_actions:
                EXPANDABLE = True

            else:
                #move to the next node by picking up the highest UCT, and then move the unvierse as well.
                action,childIndex = self.select_UCTNode(currIndex)

                #moving mcts to the next node
                currIndex = childIndex

                #moving world to the next state.
                # action_real = world.unhash(action)
                if turn_whose == AIAGENT:
                    reward,_ =curr_env.respond(action)
                else:
                    reward,_ = curr_env.act_externalwill(action) ##
                #todo:we can check if the action reward returned matches with ours at the edge.

                chain_statesTraversed.append(currIndex)
                chain_rewards.append(reward)

        return currIndex, chain_statesTraversed, chain_rewards


    def select_UCTNode(self,parentIndex):
        children = self.graph.get_out_neighbors(parentIndex)
        children_uct = [self.graph.vp.uct[child] for child in children]

        maxuct_childIndex = children[children_uct.index(max(children_uct))]
        maxuct_action_edge = self.graph.edge(parentIndex, maxuct_childIndex)
        maxuct_action = self.graph.edge_properties.action[maxuct_action_edge]

        return maxuct_action,maxuct_childIndex

    def get_bestActionGreedy(self):
        #To be called after rollout. Now, once we have an average reward for everything,
        #We just go ahead and pick the most valuable action/state pair.

        #Retrieve all children from the current state.
        allChildren_stateVertexIndices = [vertex for vertex in self.currState_vertex.out_neighbors()]

        #Retrieve their corresponding values
        allChildren_stateValues = [self.graph.vp.avg_reward[vertexIndex] for vertexIndex in allChildren_stateVertexIndices]

        #Find the best child state vertex
        bestChild_stateVertexIndex = allChildren_stateVertexIndices[allChildren_stateValues.index(max(allChildren_stateValues))]

        #Retrieve the action to reach that state.
        bestAction_edgeIndex = self.graph.edge(self.curr_stateIndex,bestChild_stateVertexIndex)

        #Retrieve the string-name of the best action.
        bestAction_name = self.graph.edge_properties.action[bestAction_edgeIndex]

        #Set the current node to go to the right place.
        self.currState_vertex = self.graph.vertex(bestChild_stateVertexIndex)
        self.currState_vertexIndex = bestChild_stateVertexIndex

        return bestAction_name

env = environment()
