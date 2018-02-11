import numpy as np
from src.arena import arena
from src.agent import Agent
import time
import copy
# from MCTS import mcts_unique as mu


#
grid_matrix = np.random.random((10,10))
#
#

g = grid_matrix.flatten()
g[[np.random.choice(np.arange(100),83,replace=False)]]=0
grid_matrix = g.reshape((10,10))
grid_matrix[3,4]=0
grid_matrix[5,5]=0
grid_matrix[6,7]=0
grid_matrix[7,7]=0
np.save('grid.npy',grid_matrix)

grid_matrix = np.load('grid.npy')
grid_matrix/=2.0
g2 = copy.deepcopy(grid_matrix)


are = arena(grid_matrix,False)


# are = arena(grid_matrix,True)
a1 = Agent(0.42,.46,.25,0,np.array([3,4]),are)
a1.load = False

# a2 = Agent(0.2,3,4,3,np.array([5,5]),2,are)
# a2.load = True

a3 = Agent(.49,.2,.2,2,np.array([6,7]),are)
a3.load = False

a2 = Agent(.3,.25,.3,0,np.array([7,7]),are)
a2.load = False

# are.add_agents([a4,a2,a3,a1])
are.add_agents([a1,a2,a3])
g1= are.grid_matrix

gm=[]
time_array = []
i=0
prob_lh = []
prob_lh2 = []
prob_ori = []
while not are.isterminal:
    print("iter "+str(i))
    i+=1

    start = time.time()
    # print ad.curr_position, ad.curr_orientation
    # print a4.curr_position, a4.curr_orientation

    # lh_action_probs = ad.behave()
    # lh_a2 = ad2.behave()
    # if lh_action_probs[4]==0:
    #     pass

    # prob_array.append(lh_action_probs)
    agent_actions_list,action_probs = are.update()

    # ad.imitate_action(agent_actions_list[0])
    # ad2.imitate_action(agent_actions_list[0])

    # prob_lh.append(lh_action_probs[agent_actions_list[0][0]])
    # prob_lh2.append(lh_a2[agent_actions_list[0][0]])

    # prob_ori.append(a4.action_probability[agent_actions_list[0][0]])

    # if lh_action_probs[agent_actions_list[0][0]]==0:
    #     print("This happens when "+str(agent_actions_list[0][0]))
    #     raise Exception("This ignoring legal actions")
    #     pass

    # are.update()
    # prob.append(action_probs)
    # for agent in are.agents:
    #     print agent.action_probability
    # print(are.isterminal)
    # print (are.no_items)
    are.check_for_termination()
    # print (are.isterminal)
    time_array.append(time.time()-start)
    are.get_agent_posarray()

    are.get_item_posarray()
    ipos = are.item_pos_array
    final = False
    time.sleep(.2)
    # print(np.mean(np.array(time_array)))

# print time_array
prob_lh = np.array(prob_lh).astype('float32')
prob_ori = np.array(prob_ori)


print np.where(prob_lh==0)
print(np.product(prob_lh))
print(np.sum(np.log(prob_lh)))
print(np.sum(np.log(prob_lh2)))
if np.all(prob_lh):
    print('All set')
else:
    print("This is not right")

if np.all(prob_ori):
    print('All set here too')
else:
    print("this is not right either.")

# print prob_array
# print prob3
# print prob_array_2
# m = mu.mcts(visualize=False)
#
#
# m.addVertex('root',False)
# m.curr_stateIndex = 0
# m.curr_state = 'root'
# m.rollout(are,0)


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



