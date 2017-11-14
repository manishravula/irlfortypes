import numpy as np
from arena import arena
from agent import agent
import time


# grid_matrix = np.random.random((10,10))
# grid_matrix[3,4]=0
# grid_matrix[5,5]=0
# grid_matrix[6,7]=0
# grid_matrix[7,7]=0
# np.save('grid.npy',grid_matrix)
grid_matrix = np.load('grid.npy')

are = arena(grid_matrix,False)
a1 = agent(0.5,4,.25,0,np.array([3,4]),1,are)
a1.load = True

a2 = agent(0.2,3,.4,3,np.array([5,5]),2,are)
a2.load = True

a3 = agent(.4,5,.5,2,np.array([6,7]),.5,are)
a3.load = True


a4 = agent(.1,4,.6,0,np.array([7,7]),.9,are)
a4.load = False

are.add_agents([a1,a2,a3,a4])
g1= are.grid_matrix
# are.update_vis()
# time.sleep(1)
# are.visualizer.snapshot('base')



# res = a1.get_visibleAgentsAndItems()
# print(res)
# res2 = a2.get_visibleAgentsAndItems()
# print(res2)
# res3 = a3.get_visibleAgentsAndItems()
# print(res3)
res4 = a2.get_visibleAgentsAndItems()
print(res4)
res = a2.choosetarget(res4)
print(res)


# are.update()
# time.sleep(1)
# are.update_vis()
# g2=are.grid_matrix


