import numpy as np
from arena import arena
from agent import agent
import time
import pdb
import copy


# #
# grid_matrix = np.random.random((10,10))
# #
# #
# #
# g = grid_matrix.flatten()
# g[[np.random.choice(np.arange(100),90,replace=False)]]=0
# grid_matrix = g.reshape((10,10))
# grid_matrix[3,4]=0
# grid_matrix[5,5]=0
# grid_matrix[6,7]=0
# grid_matrix[7,7]=0
# np.save('grid.npy',grid_matrix)

grid_matrix = np.load('grid.npy')
grid_matrix/=2.0
g2 = copy.deepcopy(grid_matrix)


are = arena(grid_matrix,True)
a1 = agent(0.5,4,.25,0,np.array([3,4]),1,are)
a1.load = True

a2 = agent(0.2,3,4,3,np.array([5,5]),2,are)
a2.load = True

a3 = agent(.4,5,.5,2,np.array([6,7]),.5,are)
a3.load = True


a4 = agent(.1,4,.6,0,np.array([7,7]),.9,are)
a4.load = False

are.add_agents([a1,a2,a3,a4])
g1= are.grid_matrix

gm=[]
for i in range(80):
    print("iter "+str(i))
    gm1 = copy.deepcopy(are.grid_matrix)
    gm.append(gm1)
    are.update()
    are.get_agent_posarray()
    print (np.linalg.norm(are.grid_matrix-gm1))
    aposar = are.agent_pos_array
    ap1 = np.sum(np.linalg.norm(aposar-aposar[0],axis=1)==0) > 1
    ap2 = np.sum(np.linalg.norm(aposar-aposar[1],axis=1)==0) > 1
    ap3 = np.all(np.linalg.norm(aposar-aposar[2],axis=1)==0) > 1
    ap4 = np.all(np.linalg.norm(aposar-aposar[3],axis=1)==0) > 1
    if ap1 and ap2 and ap3 and ap4:
        pdb.set_trace()
        raise Exception("CROSSING PATHS!")

    are.get_item_posarray()
    ipos = are.item_pos_array
    final = False
    for agent_loc in aposar:
        for item_loc in ipos:
            if np.linalg.norm(item_loc-agent_loc) ==0:
                raise Exception("ITEM AND AGENT IN SAME PALCE")
                pass
    time.sleep(.4)




    if final:
        raise Exception("Items fucked up")

    c1 = aposar[0]
    # time.sleep(.2)
# are.update_vis()
# time.sleep(1)
# are.visualizer.snapshot('base')
# are2 = are.copy()
# print("DAAAAANG")
# ap=[]

# for i in range(4):
#     ap_n = [[are.agents[i].action_probability,are2.agents[i].action_probability] for i in range(4)]
#     ap.append(ap_n)
#     are.update()
#     are2.update()
#     time.sleep(.2)
#
# print(ap)
# res = a1.get_visibleAgentsAndItems()
# print(res)
# res2 = a2.get_visibleAgentsAndItems()
# print(res2)
# res3 = a3.get_visibleAgentsAndItems()
# print(res3)


# are.experiment()

# are.update()
# time.sleep(1)
# are.update_vis()
# g2=are.grid_matrix


