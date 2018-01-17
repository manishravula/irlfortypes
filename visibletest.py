import numpy as np
from src.arena import arena
from src.agent import Agent

grid_matrix = np.random.random((10,10))

grid_matrix[5,5]=0




are = arena(grid_matrix,True)
param1 = .42
param2 = .46
param3 = np.linspace(0,1,9)
position = np.array([5,5])

agent_list = []
for param in param3:
    a = Agent(param1,param2,param,0,position,are)
    a.load = False
    a.curr_orientation=np.pi*3/2.0
    agent_list.append(a)



are.add_agents(agent_list)
g1= are.grid_matrix

are.update_vis()

for agent in are.agents:
    visible = agent.get_visibleAgentsAndItems()
    print(visible)
    print(len(visible[0])+len(visible[1]))
pass




