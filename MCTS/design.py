class world():
    def __init__(self):
        self.__isterminal = False #Flag goes True if we are in the terminal state.

    def get_legalActions(self,turn):
        """
        :param turn: Boolean variable that specifies whose turn it is, to provide
                    the possible legal actions.
        :return: list of strings, with each string representin a hash of the true action.
        """

    def react(self,action):
        """
        :param action: str - hashed representation of true action.
        :return: reward, nextstate (hashed-str)

        """

    def act_external(self,action):
        """
        :param action: str - hashed representation of true action, requested by MCTS
        :return: reward, nextstate (hashed-str)

        """

    def act(self):
        """
        Just act according to will
        :return: reward, nextstate
        """



