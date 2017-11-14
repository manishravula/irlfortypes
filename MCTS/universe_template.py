
class universe():
    #Could be plug-and-play
    #A game should allow :
    #1) Tell all possible legal states
    #2) Evaluate if a state is terminal or not
    #3) If it is terminal, tell who won.

    def __init__(self):
        #TODO: set this to the beginning state of every-universe
        #design: This state will always give the turn to the AI agent.
        self.state = False

    def create_world(self):
        world = universe()
        return world

    def get_actionsLegal(self,state):
        #First get all moves allowed.
        #Apply them to the game and get the possible states.
        #Always generates a list of new legal next-state's features possible.
        return


    def react(self,action_external,state,Transition=False):
        """
        :param action_external: action taken by the external agent.
        :param state: state from which the agent is taking the action.
        :param Transition: Should the world transition into that state, or just peek and tell us what the state is.
        :return:
        """
        #This function gives the universe's reaction to a particular user action in a state.
        #This could be thought as the transition when the agent acts, pushing the universe into its turn-taking state.
        self.state = self.get_stateNext(self.state,action_external)
        return

    def act(self,state,Transition=False):
        """
        :param state: state from which the world should act
        :param Transition: Should the world transition into that state, or just peek and tell us what the state is.
        :return:
        """
        #This function gives the universe's response to a particular state when the turn is its.
        #This cold be thought of as universe's move.
        return

    def is_terminalstate(self,state):
        return True

    def get_reward(self,curr_state,action,next_state):
        return 0
        #return one if player 1 won #Whose action we are trying to build a tree for
        #return -1 if player 2 won, Whose actions are generated randomly.

    def get_stateNext(self,curr_state,action_curr):
        return self.state #Some dummy
