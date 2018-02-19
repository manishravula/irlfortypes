import itertools
import logging
import logging.config

import numpy as np

import src.arena as original_arena
import src.global_const
from MCTS import mcts as _mcts
from experiments import configuration as config
from src import generate_init as gi
from src.global_const import ACTIONS2CHAR, CHAR2ACTIONS

#if no action, then it is represented as 'n'

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

orea = original_arena.arena

class mcts_arena(orea):
    def __init__(self,grid_matrix,visualize):
        orea.__init__(self,grid_matrix,visualize)
        self.mctsagent_set = False


    def add_MCTSagent(self,mctsAgent):
        self.agents.append(mctsAgent)
        self.mcts_agent = mctsAgent
        self.mctsagent_set = True
        return

    def duplicate(self):
        return

    def hash_currstate(self):
        """
        Synthesize state definition from the current state.
        :return:
        """
        # return str(np.ravel_multi_index(np.argwhere(self.grid_matrix>0).T, self.grid_matrix.shape))

        # Optimize-we don't need big strings.
        return 'a'

    def hash_action(self,action_array):
        if len(action_array)==1:
            return ACTIONS2CHAR[action_array[0]]
        else:
            actionRepr = ''
            for action in action_array:
                actionRepr+=(ACTIONS2CHAR[action])
            return actionRepr


    def unhash_action(self,actionRepr):
        if len(actionRepr)==1:
            return CHAR2ACTIONS[actionRepr]
        else:
            action_array = []
            for char in actionRepr:
                action_array.append(CHAR2ACTIONS[char])
            return np.array(action_array)


    def getActions_legalFromCurrentState(self,turn_whose):
        """
        :param turn_whose: Legal actions of whose turn? The legal actions depend on whose turn this is.
        MCTS requests for all possible legal actions from the current state of the environment.
        :return: List of legal actions.

        #Debug: In: an arena equipped with one MCTS agent.
                Out: Visually check that the actions match.
        """
        # TODO; Make this a generator.
        # TODO: We only need one wrong example in most of the cases.
        if turn_whose == _mcts.UNIVERSE:  # universe's turn
            validactionString_list = []
            for agent in self.agents[:-1]:  # exclude MCTS agent
                validactionProb,_,_ = agent.get_legalActionProbs()
                valid_actions = np.argwhere(validactionProb != 0).reshape(-1)
                validAction_string = self.hash_action(valid_actions)
                validactionString_list.append(validAction_string)
            validactionString_list.append('n') #For the MCTS agent in the end.
            all_actionStringVectors = [''.join(ele) for ele in itertools.product(*validactionString_list)]
            return all_actionStringVectors
        else:
            valid_actionProbs_withLoad,_,_ = self.mcts_agent.get_legalActionProbs()
            valid_actionsChars = self.hash_action(np.argwhere(valid_actionProbs_withLoad>0).reshape(-1))

            all_actionStringVectors = []
            default_string = 'n'*(len(self.agents)-1)
            for actionChar in valid_actionsChars:
                all_actionStringVectors.append(default_string+actionChar)
            return all_actionStringVectors

    def getNumberOfActions_legalFromCurrentState(self,turn_whose):
        """
         :param turn_whose: Legal actions of whose turn? The legal actions depend on whose turn this is.
                MCTS requests for all possible legal actions from the current state of the environment.
         :return: number of legalactions

         #Debug: In: an arena equipped with one MCTS agent.
                 Out: Visually check that the actions match.
        """
        nlegalactions = 1
        if turn_whose == _mcts.UNIVERSE:  # universe's turn
            for agent in self.agents[:-1]:  # exclude MCTS agent
                validactionProb,_,_ = agent.get_legalActionProbs()
                n_valid_actions_currAgent = np.sum(validactionProb != 0)
                nlegalactions*=n_valid_actions_currAgent
            return nlegalactions
        else:
            valid_actionProbs_withLoad, _, _ = self.mcts_agent.get_legalActionProbs()
            return np.sum(valid_actionProbs_withLoad!=0)

    def getAction_randomLegalFromCurrentState(self, turn_whose):
        """
        Return a random action from the list of legal actions possible at this state.
        :param turn_whose: Whose turn is it now?
        :return:
        """
        if turn_whose == _mcts.UNIVERSE:  # universe's turn
            rand_validActionString = ''
            for agent in self.agents[:-1]:  # exclude MCTS agent
                validactionProb,_,_ = agent.get_legalActionProbs()
                random_valid_action = np.random.choice(np.argwhere(validactionProb != 0).reshape(-1)) #COULD GO WRONG
                random_validActionChar = ACTIONS2CHAR[random_valid_action]

                rand_validActionString+=random_validActionChar

            rand_validActionString+='n' #For the last MCTS agent.
            return rand_validActionString
        else:
            valid_actionProbs_withLoad,_,_ = self.mcts_agent.get_legalActionProbs()
            rand_valid_actionsChar = ACTIONS2CHAR[(np.random.choice(np.argwhere(valid_actionProbs_withLoad>0).reshape(-1)))]
            default_string = 'n'*(len(self.agents)-1)
            rand_validActionString = default_string+rand_valid_actionsChar
            return rand_validActionString



    def respond(self,action_agent):
        """
        MCTS proposes an action, called action_external and the environment applies the action
        and responds to the action by transitioning into a new state.
        :param action_external: the agent's action.
        :return:
        """
        #MCTS agent just requested to act action_agent.
        #It already carried out self.execute_action() so the grid matrix is also altered.
        #note this.

        #Should NEVER be CALLED BEFORE CALLING act_freewill or act_externalwill
        #Because the food consumption is evaluated here.

        init_nitems = len(self.items)
        if np.any([a.load for a in self.agents]):
            self.update_foodconsumption()
            n_itemsConsumed = init_nitems - len(self.items)
            reward = n_itemsConsumed
        else:
            reward = 0
        self.check_for_termination()

        new_state = self.hash_currstate()
        return reward, new_state

    def act_freewill(self):
        """
        The environment acts according to its will and transitions into a new state
        where the turn is now the agent's.
        :return:
        """
        init_nitems = len(self.items)

        for agent in self.agents[:-1]:
            action_probs = agent.behave(False)
            agent_action = agent.behave_act(action_probs)
            agent.execute_action(agent_action)

        reward = 0
        new_state = self.hash_currstate()
        self.check_for_termination()
        return reward,new_state

    def act_externalwill(self,action_externalRequested):
        """
        Although it's the environment's turn, it acts the action ordered by something else.
        The action here is action_external.
        :param action_external:
        :return:
        """
        #first convert the given action string to action-consequence-pairs.
        #HOWWW
        action_consequence_list= []

        for idx,actionChar in enumerate(action_externalRequested[:-1]):
            if actionChar=='n':
                raise Exception("Passed a no-do action to one of the agents")
            action = CHAR2ACTIONS[actionChar]
            movement = src.global_const.ACTION2MOVEMENTVECTOR[action] #Check
            final_next_position = self.agents[idx].curr_position+movement
            action_and_consequence = [action,movement,final_next_position]
            _ = self.agents[idx].behave(False)
            self.agents[idx].execute_action(action_and_consequence)

        reward=0
        new_state=self.hash_currstate()
        self.check_for_termination()
        return reward,new_state

    def getvalue_terminalState(self):
        """
        The terminal state has a value, for instance, if the agent succesfully completely
        finishes all tasks, the reward is great and good.
        If the agent fails, this terminal state is bitter and not needed.
        :return:
        """
        return 0



if __name__ == "__main__":
    def Itest_legalfuncs():
        n_agents = 4
        cvs = config.VISUALIZE_SIM
        config.VISUALIZE_SIM = False
        are = mcts_arena(gi.generate_arena_matrix(10, 23, )[0], False)
        agents = gi.generate_agents(4,are,from_save=False)
        are.init_add_agents(agents[:-1])
        are.add_MCTSagent(agents[-1])

        for i in range(10):


            #Universe's role

            #Test 1: Inspect all possible action strings.
            allLegalActions = are.getActions_legalFromCurrentState(_mcts.UNIVERSE)
            logger.debug("The generated random actions for UNI look like this \n {}".format(allLegalActions))
            n_allLegalActinos = are.getNumberOfActions_legalFromCurrentState(_mcts.UNIVERSE)

            assert len(allLegalActions[0]) == len(are.agents)
            assert len(allLegalActions)==n_allLegalActinos; 'Number of actions differ from {} to {}'.format(
                len(allLegalActions),n_allLegalActinos)

            logger.debug("PASS: Test 1 passed. The number of legal actions function works")

            #Test 2: Testing random action vector generation:
            for i in range(100):
                random_legalAction = are.getAction_randomLegalFromCurrentState(_mcts.UNIVERSE)
                logger.debug("Iter {}, turn UNI, randomaction: {}".format(i,random_legalAction))
                assert random_legalAction in allLegalActions; 'Random action generator UNI produced {} which is' \
                                                             'not in allLegalActions'.format(random_legalAction)
                random_legalAction2 = are.getAction_randomLegalFromCurrentState(_mcts.AIAGENT)
                logger.debug("Iter {}, turn AIG, randomaction: {}".format(i,random_legalAction2))
                #assert random_legalAction2 in allLegalActions; 'Random action generator AIG produced {} which is' \
                                                             #'not in allLegalActions'.format(random_legalAction2)
            logger.debug("PASS: Test 2 passed. The random action generator works for both turns.")

            #Test 2: Testing forced execution.
            r,nstate = are.act_freewill()
            logger.debug("PASS: After acting freewill, new state is {} with reward {}".format(nstate,r))

            #Test 3: Test
            random_action = are.getAction_randomLegalFromCurrentState(_mcts.AIAGENT)
            assert len(random_action) == len(are.agents)

            #MCTS agent's role
            logging.debug("Random action generated is {}".format(random_action))
            # logging.debug("Random action's index is {}".format(CHAR2ACTIONS[random_action[0]]))
            # logging.debug("Random movement is {}".format(original_agent.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[0]]]))
            random_movement = src.global_const.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[-1]]]
            random_nextPosition = are.mcts_agent.curr_position+random_movement

            r,next_state = are.respond(random_action)
            logging.debug("Reward returned for are.respond is {} and new state is {}".format(r,next_state))

            #Test 4: Acting external will
            random_externalAction = are.getAction_randomLegalFromCurrentState(_mcts.UNIVERSE)
            r,next_state = are.act_externalwill(random_externalAction)
            logging.debug("PASS: act_external with random action {} and returned reward {} with new state {}".format(
                random_externalAction,r,next_state
            ))

            #Test 5: Re-do respond, to finish 2 cycles inside one iteration.
            random_action = are.getAction_randomLegalFromCurrentState(_mcts.AIAGENT)
            assert len(random_action) == len(are.agents)

            # MCTS agent's role
            # logging.debug("Random action generated is {}".format(random_action))
            # logging.debug("Random action's index is {}".format(CHAR2ACTIONS[random_action[0]]))
            # logging.debug(
            #     "Random movement is {}".format(original_agent.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[0]]]))
            random_movement = src.global_const.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[-1]]]
            random_nextPosition = are.mcts_agent.curr_position + random_movement

            r, next_state = are.respond(random_action)
            logging.debug("Reward returned for are.respond is {} and new state is {}".format(r, next_state))


            #pick action to execute.


    def Jtest_legalfuncs():
        n_agents = 4
        cvs = config.VISUALIZE_SIM
        # config.VISUALIZE_SIM = False
        are = mcts_arena(gi.generate_arena_matrix(10, 10, )[0], True)
        agents = gi.generate_agents(4, are, from_save=False)
        are.init_add_agents(agents[:-1])
        are.add_MCTSagent(agents[-1])
        iter = 0

        while(not are.isterminal):
            iter+=1
            print(iter)
            # Universe's role

            # Test 2: Testing forced execution.
            r, nstate = are.act_freewill()
            logger.debug("PASS: After acting freewill, new state is {} with reward {}".format(nstate, r))

            # Test 3: Test
            random_action = are.getAction_randomLegalFromCurrentState(_mcts.AIAGENT)
            assert len(random_action) == len(are.agents)

            # MCTS agent's role
            logging.debug("Random action generated is {}".format(random_action))
            # logging.debug("Random action's index is {}".format(CHAR2ACTIONS[random_action[0]]))
            # logging.debug("Random movement is {}".format(original_agent.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[0]]]))
            random_movement = src.global_const.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[-1]]]
            random_nextPosition = are.mcts_agent.curr_position + random_movement

            r, next_state = are.respond(random_action)
            logging.debug("Reward returned for are.respond is {} and new state is {}".format(r, next_state))

            # Test 4: Acting external will
            # random_externalAction = are.getAction_randomLegalFromCurrentState(_mcts.UNIVERSE)
            # r, next_state = are.act_externalwill(random_externalAction)
            # logging.debug("PASS: act_external with random action {} and returned reward {} with new state {}".format(
            #     random_externalAction, r, next_state
            # ))
            #
            # # Test 5: Re-do respond, to finish 2 cycles inside one iteration.
            # random_action = are.getAction_randomLegalFromCurrentState(_mcts.AIAGENT)
            # assert len(random_action) == len(are.agents)
            #
            # # MCTS agent's role
            # # logging.debug("Random action generated is {}".format(random_action))
            # # logging.debug("Random action's index is {}".format(CHAR2ACTIONS[random_action[0]]))
            # # logging.debug(
            # #     "Random movement is {}".format(original_agent.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[0]]]))
            # random_movement = original_agent.ACTION2MOVEMENTVECTOR[CHAR2ACTIONS[random_action[-1]]]
            # random_nextPosition = are.mcts_agent.curr_position + random_movement
            #
            # r, next_state = are.respond(random_action)
            # logging.debug("Reward returned for are.respond is {} and new state is {}".format(r, next_state))

            # pick action to execute.


    def iterlengthtest():
        n_agents = 4
        cvs = config.VISUALIZE_SIM
        # config.VISUALIZE_SIM = False
        are = mcts_arena(gi.generate_arena_matrix(10, 25, )[0], True)
        agents = gi.generate_agents(4, are, from_save=False)
        are.init_add_agents(agents[:-1])
        are.add_MCTSagent(agents[-1])
        iter = 0

        while (not are.isterminal):
            iter += 1
            print(iter)
            # Universe's role

            # Test 2: Testing forced execution.
            r, nstate = are.act_freewill()
            logger.debug("PASS: After acting freewill, new state is {} with reward {}".format(nstate, r))

            are.update_foodconsumption()


    # iterlengthtest()
    Jtest_legalfuncs()



