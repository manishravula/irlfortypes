from experiments import configuration as config
from src.global_const import CHAR2ACTIONS
from src.utils import cloner as cloner
from src.estimation import update_state as update_state
from src.mcts import mcts_tree as mctstree
import src.global_const as globals

#if no action, then it is represented as 'n'

class mcts_agent(config.AGENT_CURR):
    """
    Class to wrap the MCTS planning functionality as an agent in the arena.
    This way, the main loop in any simulation doesn't need to bother about the details of this implementation.

    Also, now that the MCTS agent kinda encompasses our whole work, it makes sense to integrate estimation built into this.

    The principle is to override the behave, behave_act methods with rollout-algorithm, and straightforward
    action selection.

    Further, this agent has to be the last one in the arena.agents list, for ID purposes.

    As the agent needs to compute updated states, it needs the 'history', 'estimated_params' and the 'agent_ids' of agents
    that it is currently tracking, the design should accomodate stoing these variables.
    This sounds like the most logical place to do so, since storing it in
        1) Arena object doesn't make sense because, we reuse arena instances in so many ways, and it uses a lot of memory.
        2) Simulation loop doesn't make sense because then we would have to introduce a hook to pass those values to our agent.
    #TODO: Include the type in the estimation results too.

    When behave is called, this agent should:

      INIT
        0) Compute updated states using the history.

      ROLLOUT_INIT
        1) replicate the arena it is using into a mcts_arena wrapper -
        2) replicate the rest of the agents post updating their states using history.
        3) replicate itself into a regular agent. -
        4) register the replicated agents in the new arena.
        5) register itself as the mcts_agent in the new arena.
        6) should call in the inherited roll-out function, passing the arena as the environment.

      CHOOSING
        1) Choose the best action from your roll-outs.
        2) Set action probabilites accordingly.
    """

    def __init__(self, capacity_param, viewRadius_param, viewAngle_param, type, curr_pos, foraging_arena):
        config.AGENT_CURR.__init__(self,capacity_param,viewRadius_param,viewAngle_param,type,curr_pos,foraging_arena)
        self.rollout_itersLimit = 200
        self.history = [] #to hold the history from beginning to end
        self.targetAgentIds = [] #To save the target agent IDs
        self.estimated_params = [] #To have the current estimated agent params.
        self.estimators = []



    def behave(self,history,trackingAgentIds,trackingAgentParameterEstimates):
        """
        Over-ridden.

        :param history: History of states of the agents.
        :param trackingAgentIds: Which agents are we tracking
        :param trackingAgentParameterEstimate: The agents' config after we tracked. This is a dict with a subset of keys from
                                    the agent.__getstate__()
        :return:
        """

        #Sanity checks.
        if config.DEBUG:
            for id,pestimate in zip(trackingAgentIds,trackingAgentParameterEstimates):
                #keys in the pestimate
                assert isinstance(pestimate,dict), 'Dict is not passed'
                allkeys = pestimate.keys()
                sampleStateInHistory = history[1][1][id][0] #timestep-agentinfo-firstagentinfo-agentstate
                historyStateKeys = sampleStateInHistory.keys()
                for key in allkeys:
                    assert key in historyStateKeys, 'Estimated parameters not in the history state defs'
            assert len(self.arena.agents)-1 not in trackingAgentIds ,'The MCTS agent is tracking itself. '


        """Note:
        So behave is called after the execute action of every other agent in the arena is called.
        That means, at everystep, we use the most recent estimates from curr_step's actions, and the most recent
        state histories, that too from the current step. 
        
        A small glitch lies in the fact that the history list would be incomplete as we are using in right before finishing
        a full iteration. The last element holds the history of the MCTS agent.
        This state doesn't need to be accurate, as we only use an ensemble of the rest of the agents+arena
        to be used as an environment for the MCTS solver.
        
        So we just add a dummy element towards the end of the history inorder to be able to call the update_state.
        """


        history[-1][1].append(history[-1][1][1]) #just to have a filler value for the last MCTS agent's
        #state.
        corrected_states,_ = update_state.get_updatedStateForMultipleAgents(history,trackingAgentIds,trackingAgentParameterEstimates)

        init_arena_for_rollout, init_agents_for_rollout = self.generate_environment(corrected_states,trackingAgentIds)
        mcts_planner = mctstree.mcts(init_arena_for_rollout,False)
        #TODO parallelize
        for i in range(config.N_ROLLOUTS):
            arena_for_rollout, agents_for_rollout = self.generate_environment(corrected_states,trackingAgentIds)
            mctsAgent_forArena = cloner.clone_Agent(self.__getstate__(),arena_for_rollout)
            arena_for_rollout.add_MCTSagent(mctsAgent_forArena)
            mcts_planner.rollout(arena_for_rollout,mcts_planner.rootVertex_index)

        action_name = mcts_planner.get_bestActionGreedy()
        final_action = globals.CHAR2ACTIONS[action_name]
        final_movement = globals.ACTION2MOVEMENTVECTOR[final_action]
        final_position = self.curr_position + final_movement
        return [final_action,final_movement,final_position]


    def generate_environment(self,corrected_states,trackingAgentIds):
        #Replicating arena.
        arena_for_rollout = cloner.clone_MCTSArena(self.arena.__getstate__())

        #Replicating agents.
        agents_for_rollout = []
        target_agentIdx = 0
        for i in range(len(self.arena.agents)-1):
            if i not in trackingAgentIds:
                new_agent = cloner.clone_Agent(self.arena.agents[i].__getstate__(),arena_for_rollout)
            else:
                new_agent = cloner.clone_Agent(corrected_states[target_agentIdx],arena_for_rollout)
                target_agentIdx+=1
            agents_for_rollout.append(new_agent)

        return arena_for_rollout,agents_for_rollout


#TODO: Write tests for each of these individual functions.
#Todo: Write a wholistic experiment to use all this function and run simulations.

