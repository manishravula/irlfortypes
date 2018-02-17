from experiments import configuration as config
from src.estimation.agent_param import Agent_lh
import logging
from src.mcts import mcts_arena_wrapper as mcts_arena
logger = logging.getLogger(__name__)

def clone_ArenaAndAgents(arena_state, agentState_list):
    newArena = clone_Arena(arena_state)
    new_agentsList = [clone_Agent(agent_state,newArena) for agent_state in agentState_list]
    newArena.init_add_agents(new_agentsList)
    return newArena, new_agentsList

def clone_ArenaAndLhAgents(arena_state, agentState_list):
    newArena = clone_Arena(arena_state)
    new_agentsList = [clone_AgentLH(agent_state,newArena) for agent_state in agentState_list]
    newArena.init_add_agents(new_agentsList)
    return newArena, new_agentsList

def clone_Arena(arena_state):
    newArena = config.ARENA_CURR(arena_state['grid_matrix'],False)
    newArena.__setstate__(arena_state)
    return newArena

def clone_MCTSArena(arena_state):
    newMCTS_arena = mcts_arena.mcts_arena(arena_state['grid_matrix'],False)
    newMCTS_arena.__setstate__(arena_state)
    return newMCTS_arena

def clone_Agent(agentState,arena_obj):
    newAgent = config.AGENT_CURR(agentState['capacity_param'],agentState['viewRadius_param'],
                                 agentState['viewAngle_param'],agentState['type'],agentState['curr_position'],arena_obj)
    newAgent.__setstate__(agentState)
    logger.info("Cloned agent with params {}".format(agentState))
    return newAgent

def clone_AgentLH(agentState,arena_obj):
    newAgent = Agent_lh([agentState['capacity_param'],agentState['viewRadius_param'],
                                 agentState['viewAngle_param']],agentState['type'],agentState['curr_position'],arena_obj)
    newAgent.__setstate__(agentState)
    return newAgent

