import numpy as np
from MCTS import mcts as _mcts


class environment():
    def __init__(self):
        self.current_state = 0
        self.is_terminal = False
        self.turn_whose = True
        return
    def duplicate(self):
        return self.__init__()

    def getActions_legalFromCurrentState(self,turn_whose):
        return

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



