import time

import numpy as np

import src.generate_init as gi
from experiments import configuration as config
from src.utils import cloner as cloner

"""
Given a history of states, we need to recalculate the agent's current state. This is the case because the
current defintion of type is not Markovian.
"""


def get_updatedStateForSingleAgent(history, agentIndex, targetConfig, retActionProbs=False):
    """
    Agent state is a mis-nomer of sorts. It contains both its parameters and state.
    :param retActionProbs:
    :param history:  History is basically a list of time-step-ordered-lists.
                    Each time step, we save the state of the system in [arenaState,[[agent1State,action],[agent2State,action],.....]
                    The states are dictionaries in themselves.

    :param agentIndex: Which agent's state to update.
    :param targetConfig: The config to which the state has to be updated.
    :return:
    """
    # TODO change targetConfig from an equivalent representation of state to just the paramter values.
    #Initializing state
    initstate_Arena = history[0][0] #TODO change the datastructure for history
    initstatelist_Agents = [agentinfo[0] for agentinfo in history[0][1]]
    n_agents = len(initstatelist_Agents)

    initstatelist_Agents[agentIndex] = targetConfig

    fresh_arena, fresh_AgentsList = cloner.clone_ArenaAndAgents(initstate_Arena, initstatelist_Agents)
    actionProbs_list = []
    iteridx = 0
    for timestep_state in history:
        print("Updating iteration no: {}".format(iteridx))
        iteridx+=1
        #First dummy behave to collect the possible
        currstep_AgentstateInfoList = timestep_state[1]
        currstep_ArenastateInfo = timestep_state[0]

        #Set state now.
        fresh_arena.__setstate__(currstep_ArenastateInfo)

        for agent,m in zip(fresh_AgentsList,range(n_agents)):
            if m is not agentIndex:
                #Check if the state matches with what was in our books
                comparestate(agent.__getstate__(),currstep_AgentstateInfoList[m][0])
                #Change anyway.
                agent.__setstate__(currstep_AgentstateInfoList[m][0])

        curr_actionProbs = []
        for jdx,agent in enumerate(fresh_AgentsList):
            actionProbs = agent.behave(False)
            curr_actionProbs.append(actionProbs)
            agent.execute_action(currstep_AgentstateInfoList[jdx][1])

        actionProbs_list.append(curr_actionProbs)

    if retActionProbs:
        return fresh_AgentsList[agentIndex].__getstate__(), actionProbs_list
    else:
        return fresh_AgentsList[agentIndex].__getstate__()


def get_updatedStateForMultipleAgents(history, agentIndexList, targetConfigList, retActionProbs):
    """
    Agent state is a mis-nomer of sorts. It contains both its parameters and state.
    :param retActionProbs:
    :param history:  History is basically a list of time-step-ordered-lists.
                    Each time step, we save the state of the system in [arenaState,[[agent1State,action],[agent2State,action],.....]
                    The states are dictionaries in themselves.

    :param agentIndex: Which agent's state to update.
    :param targetConfig: The config to which the state has to be updated.
                        As the internal mechanism involves updating the dictionary,
                        this can just be a dict of a few elements of the state.
    :return:
    """
    #Initializing state
    initstate_Arena = history[0][0]
    initstatelist_Agents = [agentinfo[0] for agentinfo in history[0][1]]
    n_agents = len(initstatelist_Agents)

    for agent_indx,id in enumerate(agentIndexList):
        initstatelist_Agents[agent_indx] = targetConfigList[id]

    fresh_arena, fresh_AgentsList = cloner.clone_ArenaAndAgents(initstate_Arena, initstatelist_Agents)
    actionProbs_list = []
    iteridx = 0
    if len(history) is 0:
        return []
    for timestep_state in history:
        print("Updating iteration no: {}".format(iteridx))
        iteridx+=1
        #First dummy behave to collect the possible
        currstep_AgentstateInfoList = timestep_state[1]
        currstep_ArenastateInfo = timestep_state[0]

        #Set state now.
        fresh_arena.__setstate__(currstep_ArenastateInfo)

        for agent,m in zip(fresh_AgentsList,range(n_agents)):
            if m not in agentIndexList:
                #Check if the state matches with what was in our books
                comparestate(agent.__getstate__(),currstep_AgentstateInfoList[m][0])
                #Change anyway.
                agent.__setstate__(currstep_AgentstateInfoList[m][0])
            # else:
            #     Soft compare and don't raise assertions.
                # soft_comparestate(agent.__getstate__(),currstep_AgentstateInfoList[m][0])

        #Updating
        curr_actionProbs = []
        for jdx,agent in enumerate(fresh_AgentsList):
            actionProbs = agent.behave(False)
            curr_actionProbs.append(actionProbs)
            agent.execute_action(currstep_AgentstateInfoList[jdx][1])

        # if np.any([agent.load for agent in fresh_arena.agents]):
        #     fresh_arena.update_foodconsumption()

        actionProbs_list.append(curr_actionProbs)
    if retActionProbs:
        return [fresh_AgentsList[agentIndex].__getstate__() for agentIndex in agentIndexList], actionProbs_list
    else:
        return [fresh_AgentsList[agentIndex].__getstate__() for agentIndex in agentIndexList]

def comparestate(state1,state2):
    all_keys = state1.keys()
    for key in all_keys:
        assert np.all(state1[key] == state2[key]), "Key {} doesn't match from {} to {}.".format(key,state1[key],state2[key])
    return

def soft_comparestate(state1,state2):
    all_keys = state1.keys()
    for key in all_keys:
        if np.all(state1[key] == state2[key]):
            pass
        else:
            print("Key {} doesn't match with val1 {} and val2 {}".format(key,state1[key],state2[key]))
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
        st = time.time()

        for i in range(50):
            print('iter {}'.format(i))
            curr_arenastate = are.__getstate__()
            curr_agentstateList = [[agent.__getstate__(),None] for agent in agents]
            curr_actionProbList = []
            for jdx,agent in enumerate(agents):
                actionProb = agent.behave(False)
                curr_actionProbList.append(actionProb)
                actionAndConsequence = agent.behave_act(curr_actionProbList[jdx])
                agent.execute_action(actionAndConsequence)
                curr_agentstateList[jdx][1] = actionAndConsequence

            are.update_foodconsumption()
            actionProbs_list.append(curr_actionProbList)
            history.append([curr_arenastate,curr_agentstateList])
        et = time.time()
        print("Main loop time: {}".format(et-st))
        start_time = time.time()
        final_state, updated_actionProbsList = get_updatedStateForSingleAgent(history, 2, init_config_agent2)
        end_time = time.time()
        print("Time taken is {}".format(end_time-start_time))

        soft_comparestate(final_state,agents[2].__getstate__())
    extest1()








