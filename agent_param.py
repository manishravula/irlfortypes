"""
This agent is for keeping track of likelihood calculations for different type and parameter
settings.

The situation is something like this:

We have an agent actually taking actions and foraging in the arena. We need to calculate
the likelihood of that agent having a particular set of type-parameter combinations.
That should basically be the product of what the agent with that set of type-parameter
setting's probability of action at each state.

As this probability is in-tractable unless we have actually have an agent behaving with
these particular settings in the original agent's place, we are writing this new class.
This new class does exactly that.

This new agent, which is dubbed as Agent_lh, will walk in the true agent's shoes, meaning,
it will act exactly as the original agent


As we need to be in a different agents' shoes to calculate likelihood for each type-parameter combo

Similarities between older agent and the new agent are:

Only difference lies in the execution policy. We note the execution of the agent shouldn't
really affect the gridmatrix.
"""


from agent import Agent
import numpy as np

class Agent_lh(Agent):
    def __init__(self,param_vector,tp,curr_pos,foraging_arena):

        Agent.__init__(self,param_vector[0],param_vector[1],param_vector[2],tp,curr_pos,foraging_arena)
        self.likelihood_total = 1 #INIT
        self.likelihood_array = []

    # def imitate_action(self,action_and_consequence):
    #     """
        #USELESS
        # optimize: We need to remove this method soon.
        # This method changes the
        # :param action_and_consequence: The action_and_consequence performed by  the agent we are
        #                                 trying to imitate.
        # :return:
        # """
        # imit_action_probs = np.zeros(5).astype('float')
        # imit_action = action_and_consequence[0]
        # imit_action_probs[imit_action] = 1.0 #forcing it use the actions of the true-agent.
        # action_and_consequence = self.behave_act(imit_action_probs)
        # execute dummy action.
        # self.execute_action_dummy(action_and_consequence)



    def execute_action_dummy(self,action_and_consequence):
        """
        Fake-executes the action. i.e this doesn't change ths state of gridmatrix,
        in other words, it won't let the arena (outer-world) know of its movements.
        It is like a shadow to the actual, acting agent.

        It's action only changes its internal state.

        :param action_and_consequence:  The action and the consequence to execute
        :return:

        """
        # The action is approved by the arena, and is being executed by the agent.
        # We can only move if it is not a load action.
        [final_action, final_movement, final_nextposition] = action_and_consequence
        if final_action != 4:
            self.curr_orientation = self.action_to_orientations[final_action]
            #Don't need the following line. The actual agent has already done this.
            # self.arena.grid_matrix[self.curr_position[0], self.curr_position[1]] = 0
            self.curr_position = final_nextposition
            #don't need this either.
            # self.arena.grid_matrix[self.curr_position[0], self.curr_position[1]] = 1
        else:
            # if this is a load action, this is probably already taken care of, by the arena.
            # Turn towards the item
            self.load = True
            if np.any(self.curr_destination):
                to_move = (self.curr_destination - self.curr_position)
                if np.sum(np.abs(to_move)) < 2:  # Only align orientation if the item is near by.
                    action_index = self.dict_actiontoIndices[str(to_move[0]) + str(to_move[1])]
                    orientation = self.action_to_orientations[action_index]
                    self.curr_orientation = orientation
                    # self.load = False
        return

    def behave_dummy(self):
        """
        Dummy function to calcualte probs
        :return:
        """
        self.behave(True)
        self.action_probs_are_fresh = True

    def calc_likelihood(self,action_and_consequence):
        """
        This method keeps track of the likelihood of the actions (performed by the agent we are observing), that this particular holding of parameters
        would produce.

        Can only be called when self.behave() has been called before the actual agent has taken any actions.
        This is because, before the agent actually takes a step, both the agent, and this dummy agent have some probabilites of
        actions to take. these probabilites are what signal likelihood, and we get that by retrieving the probability with which
        the dummy agen would have taken the original action.

        :param actual_action_and_consequence: [action, movement_delta, resulting_final_pos] the real action that the agent we are observing has taken.
        :return:
        """


        if self.action_probs_are_fresh:
            curr_prob = self.action_probability[action_and_consequence[0]]
            self.likelihood_curr = curr_prob
            self.likelihood_array.append(curr_prob)
            self.likelihood_total*=curr_prob
            self.action_probs_are_fresh = False
        else:
            raise Exception("Not really working on the curr state probabilites")







