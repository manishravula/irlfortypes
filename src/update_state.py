import numpy as np
from experiments import config_experiment as config
from agent_param import Agent_lh
import cloner as cloner
import generate_init as gi

"""
Given a history of states, we need to recalculate the agent's current state. This is the case because the
current defintion of type is not Markovian.
"""

def get_updatedStateForSingleAgent(history,agentIndex,targetConfig):
    """
    Agent state is a mis-nomer of sorts. It contains both its parameters and state.
    :param history:  History is basically a list of time-step-ordered-lists.
                    Each time step, we save the state of the system in [arenaState,[[agent1State,action],[agent2State,action],.....]
                    The states are dictionaries in themselves.

    :param agentIndex: Which agent's state to update.
    :param targetConfig: The config to which the state has to be updated.
    :return:
    """
    #Initializing state
    initstate_Arena = history[0][0]
    initstatelist_Agents = [agentinfo[0] for agentinfo in history[0][1]]
    n_agents = len(initstatelist_Agents)

    initstatelist_Agents[agentIndex] = targetConfig

    fresh_arena, fresh_AgentsList = cloner.clone_ArenaAndAgents(initstate_Arena,initstatelist_Agents)
    actionProbs_list = []
    iteridx = 0
    for timestep_state in history:
        print("Updating iteration no: {}".format(iteridx))
        iteridx+=1
        #First dummy behave to collect the possible
        currstep_AgentstateInfoList = timestep_state[1]
        currstep_ArenastateInfo = timestep_state[0]

        #Mandatory checks.
        comparestate(fresh_arena.__getstate__(),currstep_ArenastateInfo)
        #Set state now.
        fresh_arena.__setstate__(currstep_ArenastateInfo)

        for agent,m in zip(fresh_AgentsList,range(n_agents)):
            if m is not agentIndex:
                #Check if the state matches with what was in our books
                comparestate(agent.__getstate__(),currstep_AgentstateInfoList[m][0])
                #Change anyway.
                agent.__setstate__(currstep_AgentstateInfoList[m][0])


        #Updating
        curr_actionProbs = []
        for agent in fresh_AgentsList:
            actionProbs = agent.behave(False)
            curr_actionProbs.append(actionProbs)
        for agent,j in zip(fresh_AgentsList,range(n_agents)):
            agent.execute_action(currstep_AgentstateInfoList[j][1])
        fresh_arena.update_foodconsumption()

        actionProbs_list.append(curr_actionProbs)
    return fresh_AgentsList[agentIndex].__getstate__(), actionProbs_list


def comparestate(state1,state2):
    all_keys = state1.keys()
    for key in all_keys:
        print key
        assert np.all(state1[key] == state2[key]), "Key {} doesn't match from {} to {}.".format(key,state1[key],state2[key])
    return

if __name__ == '__main__':
    def extest1():
        n_agents = 3
        cvs = config.VISUALIZE_SIM
        config.VISUALIZE_SIM=False
        are, agents = gi.generate_all(10,20,n_agents)
        init_config_agent2 = agents[2].__getstate__()
        init_config_agent2['capacity_param'] = .5
        config.VISUALIZE_SIM=cvs
        history = []


        actionProbs_list = []

        for i in range(50):
            print('iter {}'.format(i))
            curr_arenastate = are.__getstate__()
            curr_agentstateList = [[agent.__getstate__(),None] for agent in agents]
            curr_actionProbList = []
            for agent in agents:
                actionProb = agent.behave(False)
                curr_actionProbList.append(actionProb)

            for agent,j in zip(agents,range(n_agents)):
                actionAndConsequence = agent.behave_act(curr_actionProbList[j])
                curr_agentstateList[j][1] = actionAndConsequence
                agent.execute_action(actionAndConsequence)

            are.update_foodconsumption()
            actionProbs_list.append(curr_actionProbList)
            history.append([curr_arenastate,curr_agentstateList])

        final_state, updated_actionProbsList = get_updatedStateForSingleAgent(history,2,init_config_agent2)

        comparestate(final_state,agents[2].__getstate__())
        assert np.all(np.array(updated_actionProbsList) == np.array(actionProbs_list))
    extest1()









