import numpy as np
from src.arena import arena
from src.agent import Agent
import time
import copy
# from MCTS import mcts_unique as mu
from src.agent_param import Agent_lh
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from src import rejection_sampler as rs
from champ import champ as ch
from src.ABU_estimator import  ABU


# #
# grid_matrix = np.random.random((8,8))
# #
# #
#
# g = grid_matrix.flatten()
# g[[np.random.choice(np.arange(64),50,replace=False)]]=0
# grid_matrix = g.reshape((8,8))
# grid_matrix = np.lib.pad(grid_matrix,(1,1),'constant',constant_values=(0,0))
#
#
# grid_matrix[3,4]=0
# # grid_matrix[5,5]=0
# grid_matrix[6,7]=0
# grid_matrix[7,7]=0
# np.save('grid.npy',grid_matrix)



def test(index):# #

    grid_matrix = np.random.random((8,8))

    g = grid_matrix.flatten()
    g[[np.random.choice(np.arange(64),50,replace=False)]]=0
    grid_matrix = g.reshape((8,8))
    grid_matrix = np.lib.pad(grid_matrix,(1,1),'constant',constant_values=(0,0))


    grid_matrix = np.load('grid.npy')
    grid_matrix/=2.0
    g2 = copy.deepcopy(grid_matrix)

    config_name = "config_{}".format(index)

    are = arena(grid_matrix,False)
    p1 = np.random.uniform(.1,1,3)
    p2 = np.random.uniform(.1,1,3)
    p3 = np.random.uniform(.1,1,3)
    t = np.random.randint(0,4,3)
    ls = np.random.randint(0,9,(3,2))

    grid_matrix[ls[:,0],ls[:,1]]=0

    config_dict = {}
    config_dict['grid_matrix']=grid_matrix
    config_dict['p1'] = p1
    config_dict['p2'] = p2
    config_dict['p3'] = p3
    config_dict['types']=t
    config_dict['locations']=ls
    np.save(config_name,config_dict)


    a1 = Agent(p1[0],p2[0],p3[0],t[0],ls[0],are)
    a1.load = False

    # a2 = Agent(0.2,3,4,3,np.array([5,5]),2,are)
    # a2.load = True

    a3 = Agent(p1[1],p2[1],p3[1],t[1],ls[1],are)
    a3.load = False

    a2 = Agent(p1[2],p2[2],p3[2],t[2],ls[2],are)
    a2.load = False

    # ad = Agent_lh(.1,4,.6,0,np.array([7,7]),are)
    # ad.load = False
    #
    # ad2 = Agent_lh(.1,4,.6,0,np.array([7,7]),are)
    # ad2.load = False

    # are.add_agents([a4,a2,a3,a1])
    are.add_agents([a1,a2,a3])
    abu = ABU(a1,are,False)


    g1= are.grid_matrix

    gm=[]
    time_array = []
    i=0
    j=0
    estimates_array = []
    # are.visualize=True#we want to update ourselves


    fitters = [abu.estimate_segmentForChamp_type0,abu.estimate_segmentForChamp_type1,abu.estimate_segmentForChamp_type2,abu.estimate_segmentForChamp_type3]

    config = ch.Config(fitters, length_min=5, length_mean=20, length_sigma=10.0, max_particles=1000, resamp_particles=1000)
    champ = ch.Champ(config)
    curr_items = len(are.items)
    changepoint_occured = False
    changepoint_time = 0
    itemconsumed_time = []
    nitems = len(are.items)

    while not are.isterminal and i<70:
        print("iter "+str(i))
        # print("fail "+str(j))

        start = time.time()

        #first estimate the probability
        abu.all_agents_behave()


        agent_actions_list,action_probs = are.update()
        action_and_consequence = agent_actions_list[0]

        #then let fake-act on by imitating this true-action
        abu.all_agents_imitate(action_and_consequence) #zero because we are following the first agent.

        abu.all_agents_calc_likelihood(action_and_consequence)
        abu.likelihood_polyCoeff_typesList.append(abu.fit_likelihoodPolynomial_allTypes(action_and_consequence))

        abu.calculate_modelEvidence(i)

        estimates,posterior=abu.estimate_allTypes(i)
        abu.posterior_polyCoeff_typesList.append(posterior)
        estimates_array.append(estimates)

        abu.total_simSteps+=1

        champ.observe(i,action_and_consequence)
        # print(estimates)
        # are.update_vis() #now we want to see what happened

        are.check_for_termination()

        delta = time.time()-start
        time_array.append(delta)


        if i>20 and curr_items!=len(are.items) and not changepoint_occured:
            new_type = np.random.randint(1,4,1)[0]
            # a1.curr_destination = None
            # a1.curr_memory = None

            print('Changepoint occured at {} to type {}'.format(i,new_type))
            a1.type= new_type
            changepoint_occured=True
            changepoint_time = i

        if nitems!=len(are.items):
            itemconsumed_time.append(i)
        curr_items = len(are.items)
        i+=1


    res = champ.backtrack(i-1)

    logl=[]
    logl.append(abu.estimate_segmentForChamp_type0(0,i-1))
    logl.append(abu.estimate_segmentForChamp_type1(0,i-1))
    logl.append(abu.estimate_segmentForChamp_type2(0,i-1))
    logl.append(abu.estimate_segmentForChamp_type3(0,i-1))



    ## ENTROPY STUFF
    entropy_stuff=True
    if entropy_stuff:
        entropy_set = []
        for likelihood_set in abu.likelihood_polyCoeff_typesList:
            var_set = []
            for likelipoly in likelihood_set:
                xval = abu.x_pointsDense
                probval = np.polyval(likelipoly,xval)
                var = np.var(probval)
                var_set.append(var)
            entropy_set.append(var_set)
        entropy_set = np.array(entropy_set)
        for l in range(4):
            plt.plot(entropy_set[:,l],label='Entropy vs Time of model {}'.format(l))
        plt.legend()
        plt.axvline(changepoint_time)
        plt.title("Changepoint at {} from type 0 to type {}".format(changepoint_time,a1.type))
        image_name = config_name+'_entropy.png'
        plt.savefig(image_name)

   #MODEL EVIDENCE STUFF
    mev_stuff = True
    if mev_stuff:
        abu.model_evidence = np.array(abu.model_evidence)
        for tp in abu.types:
            plt.plot(np.cumsum(abu.model_evidence[:,tp]),label='Evidence of model {}'.format(tp))
            plt.ylim(0,30)

        plt.axvline(changepoint_time)
        plt.legend()
        plt.title("Model Evidence across time")
        image_name = config_name+'_mevidence.png'
        plt.savefig(image_name)


    #ESTIMATES_STUFF:
    est_stuff = True
    if est_stuff:
        est_array = np.array(estimates_array)
        for tp in abu.types:
            plt.plot(est_array[:,tp][0],label='for type {}'.format(tp))
        plt.axvline(changepoint_time)
        plt.legend()
        plt.title("Evolution of estimates by ABU")
        image_name = config_name+'_estimates.png'
        plt.savefig(image_name)




    #LOGL across time for estimates:
    logl_stuff = True
    logl_list = []
    if logl_stuff:
        for likelilist,estimatelist in zip(abu.likelihood_polyCoeff_typesList,estimates_array):
            tp_list = []
            for tp in abu.types:
                tp_list.append(np.polyval(likelilist[tp],[estimatelist[tp][0]])[0])
            logl_list.append(tp_list)
        logl_list = np.array(logl_list)
        for tp in abu.types:
            plt.plot(logl_list[:,tp],label='type {}'.format(tp))
        plt.axvline(changepoint_time)
        plt.legend()
        plt.title("Evolution of estimated parameter's loglikelihood")
        image_name = config_name+'_likelihood.png'
        plt.savefig(image_name)


    results = {}
    results['likelihood'] = logl_list
    results['estimates_info']=est_array
    results['changepoint_time']=changepoint_time
    results['final_loglikelihood']=np.array(logl)
    results['item_consumption_times']=np.array(itemconsumed_time)
    results['champ_result']=np.array(res)
    np.save(config_name+'_simres',results)
    return results



   #

#plot variances


res_list = []
for i in range(20):
    res1 = test(i)
    res_list.append(res1)

np.save('res.npy',np.array(res_list))


