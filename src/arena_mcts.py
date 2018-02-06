import arena as original_arena
import numpy as np
import itertools
from MCTS import mcts as _mcts


class mcts_arena(original_arena.arena):
    def __init__(self,grid_matrix,visualize):
        original_arena.__init__(grid_matrix,visualize)
        self.actionsToChar = ['u','d','l','r','p']
        self.charToActions = {'u':0,'d':1,'l':2,'r':3,'p':4}
        self.current_state = 0
        self.is_terminal = False
        self.turn_whose = True
        return

    def duplicate(self):
        return self.__init__()

    def hash_currstate(self):
        """
        Synthesize state definition from the current state.
        :return:
        """
        return hash(self.grid_matrix.tostring())




    def hash_action(self,action_array):
        if len(action_array)==1:
            return self.actionsToChar[action_array[0]]
        else:
            actionRepr = ''
            for action in action_array:
                actionRepr+=(self.actionsToChar[action])
            return actionRepr


    def unhash_action(self,actionRepr):
        if len(actionRepr)==1:
            return self.charToActions[actionRepr]
        else:
            action_array = []
            for char in actionRepr:
                action_array.append(self.charToActions[char])
            return np.array(action_array)




    def getActions_legalFromCurrentState(self,turn_whose):
        """
        :param turn_whose: Legal actions of whose turn? The legal actions depend on whose turn this is.
        MCTS requests for all possible legal actions from the current state of the environment.
        :return: List of legal actions.
        """
        if turn_whose is _mcts.UNIVERSE:  # universe's turn
            validactionString_list = []
            for agent in self.agents:  # exclude MCTS agent
                validactionProb = agent.get_legalActionProbs()
                valid_actions = np.argwhere(validactionProb != 0).reshape(-1)
                validAction_string = self.hash_action(valid_actions)
                validactionString_list.append(validAction_string)
            all_actionStringVectors = [list(ele) for ele in itertools.product(*validactionString_list)]
            return all_actionStringVectors
        else:
            final_pos = self.curr_position + self.action_to_movements

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

            validactionString_list = []
            default_string = ''.join(['n' for i in range(self.no_items - 1)])
            validactionProb = self.mcts_agent.get_legalActionProbs()
            valid_actions = np.argwhere(validactionProb != 0).reshape(-1)
            for action in valid_actions:
                validactionString_list.append(default_string + self.action_hashes[action])

            return validactionString_list
        return range(10)

        return range(10)

    def getAction_randomLegalFromCurrentState(self, turn_whose):
        """
        Return a random action from the list of legal actions possible at this state.
        :param turn_whose: Whose turn is it now?
        :return:
        """
        legalactions = self.getActions_legalFromCurrentState()
        return np.random.choice(legalactions)



    def respond(self,action_agent):
        """
        MCTS proposes an action, called action_external and the environment applies the action
        and responds to the action by transitioning into a new state.
        :param action_external: the agent's action.
        :return:
        """
        reward = 0
        new_state = 2
        return reward,new_state

    def act_freewill(self):
        """
        The environment acts according to its will and transitions into a new state
        where the turn is now the agent's.
        :return:
        """
        reward=0
        new_state=0
        return reward,new_state

    def act_externalwill(self,action_externalRequested):
        """
        Although it's the environment's turn, it acts the action ordered by something else.
        The action here is action_external.
        :param action_external:
        :return:
        """
        reward=0
        new_state=0
        return reward,new_state

    def getvalue_terminalState(self):
        """
        The terminal state has a value, for instance, if the agent succesfully completely
        finishes all tasks, the reward is great and good.
        If the agent fails, this terminal state is bitter and not needed.
        :return:
        """
        return 1000




