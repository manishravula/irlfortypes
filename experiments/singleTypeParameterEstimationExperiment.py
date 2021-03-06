
import matplotlib.pyplot as plt
from src.arena import arena
from src.agent_originaltypes import Agent
from src.ABU_estimator import ABU
import numpy as np
import copy
import time
import numpy.polynomial.polynomial as poly
from matplotlib.animation import FuncAnimation
from tests import tests_helper as Tests

grid_matrix = np.random.random((8,8))
#
#

g = grid_matrix.flatten()
g[[np.random.choice(np.arange(64),50,replace=False)]]=0
grid_matrix = g.reshape((8,8))
grid_matrix = np.lib.pad(grid_matrix,(1,1),'constant',constant_values=(0,0))


grid_matrix[3,4]=0
# grid_matrix[5,5]=0
grid_matrix[6,7]=0
grid_matrix[7,7]=0


# grid_matrix = np.load('grid.npy')
grid_matrix/=2.0
g2 = copy.deepcopy(grid_matrix)


are = arena(grid_matrix,True)
a1 = Agent(0.42,.46,.25,0,np.array([3,4]),are)
a1.load = False

# a2 = Agent(0.2,3,4,3,np.array([5,5]),2,are)
# a2.load = True

a3 = Agent(.49,.2,.2,2,np.array([6,7]),are)
a3.load = False

a2 = Agent(.3,.25,.3,0,np.array([7,7]),are)
a2.load = False

# ad = Agent_lh(.1,4,.6,0,np.array([7,7]),are)
# ad.load = False
#
# ad2 = Agent_lh(.1,4,.6,0,np.array([7,7]),are)
# ad2.load = False

# are.add_agents([a4,a2,a3,a1])
are.add_agents([a3,a1,a2])
abu_param_dict = {'resolution':9,
              'refit_density':20,
              'likelihood_polyDegree':15,
              'posterior_polyDegree':15,
              'prior_polyDegree':15,
              'visualize':True,
              'saveplots':True}

abu = ABU(a3,are,abu_param_dict)
g1= are.grid_matrix
gm=[]
i=0
j=0



time_array = []
prob_lh = []
prob_lh2 = []
prob_ori = []

estimates_array = []
itemconsumed_time = []
nitems = len(are.items)
time_array = []



while not are.isterminal and i < 70:
    print("iter " + str(i))
    # print("fail "+str(j))

    start = time.time()

    # first estimate the probability
    abu.all_agents_behave()

    agent_actions_list, action_probs = are.update()
    action_and_consequence = agent_actions_list[0]

    # then let fake-act on by imitating this true-action
    abu.all_agents_imitate(action_and_consequence)  # zero because we are following the first agent.

    abu.all_agents_calc_likelihood(action_and_consequence)
    _ = abu.fit_likelihoodPolynomial_allTypes(action_and_consequence)

    abu.calculate_modelEvidence(i)

    estimates, posterior = abu.estimate_allTypes(i)
    # abu.posterior_polyCoeff_typesList.append(posterior)
    estimates_array.append(estimates)

    abu.total_simSteps += 1

    # champ.observe(i, action_and_consequence)
    # print(estimates)
    # are.update_vis() #now we want to see what happened

    are.check_for_termination()

    delta = time.time() - start
    time_array.append(delta)

    # if i > 20 and curr_items != len(are.items) and not changepoint_occured:
    #     new_type = np.random.randint(1, 4, 1)[0]
    #     # a1.curr_destination = None
    #     # a1.curr_memory = None
    #
    #     print('Changepoint occured at {} to type {}'.format(i, new_type))
    #     a1.type = new_type
    #     changepoint_occured = True
    #     changepoint_time = i

    if nitems != len(are.items):
        itemconsumed_time.append(i)
    curr_items = len(are.items)
    i += 1

# res = champ.backtrack(i - 1)

# logl = []
# logl.append(abu.estimate_segmentForChamp_type0(0, i - 1))
# logl.append(abu.estimate_segmentForChamp_type1(0, i - 1))
# logl.append(abu.estimate_segmentForChamp_type2(0, i - 1))
# logl.append(abu.estimate_segmentForChamp_type3(0, i - 1))

## ENTROPY STUFF
# estimates_array = abu.posteriorEstimates_sample
entropy_stuff = True
if entropy_stuff:
    entropy_set = []
    for likelihood_set in abu.likelihood_polyCoeff_typesList:
        var_set = []
        for likelipoly in likelihood_set:
            xval = abu.x_pointsDense
            probval = np.polyval(likelipoly, xval)
            var = np.var(probval)
            var_set.append(var)
        entropy_set.append(var_set)
    entropy_set = np.array(entropy_set)
    for l in range(4):
        plt.plot(entropy_set[:, l], label='Entropy vs Time of model {}'.format(l))
    plt.legend()
    # plt.axvline(changepoint_time)
    # plt.title("Changepoint at {} from type 0 to type {}".format(changepoint_time, a1.type))
    image_name = '_entropy.png'
    plt.savefig(image_name)
    plt.close()

# MODEL EVIDENCE STUFF
mev_stuff = True
if mev_stuff:
    abu.model_evidence = np.array(abu.model_evidence)
    for tp in abu.types:
        plt.plot(np.cumsum(abu.model_evidence[:, tp]), label='Evidence of model {}'.format(tp))
        plt.ylim(0, 30)

    # plt.axvline(changepoint_time)
    plt.legend()
    plt.title("Model Evidence across time")
    image_name = '_mevidence.png'
    plt.savefig(image_name)
    plt.close()

# ESTIMATES_STUFF:
est_stuff = True
if est_stuff:
    est_array = np.array(estimates_array)
    for tp in abu.types:
        plt.plot(est_array[:, tp,0], label='for type {}'.format(tp))
    # plt.axvline(changepoint_time)
    plt.legend()
    plt.title("Evolution of estimates by ABU")
    image_name = '_estimates.png'
    plt.savefig(image_name)
    plt.close()

# LOGL across time for estimates:
logl_stuff = True
logl_list = []
if logl_stuff:
    for likelilist, estimatelist in zip(abu.likelihood_polyCoeff_typesList, estimates_array):
        tp_list = []
        for tp in abu.types:
            tp_list.append(np.polyval(likelilist[tp], [estimatelist[tp][0]])[0])
        logl_list.append(tp_list)
    logl_list = np.array(logl_list)
    for tp in abu.types:
        plt.plot(logl_list[:, tp], label='type {}'.format(tp))
    # plt.axvline(changepoint_time)
    plt.legend()
    plt.title("Evolution of estimated parameter's loglikelihood")
    image_name = '_likelihood.png'
    plt.savefig(image_name)
    plt.close()

#Likelihood vs param value for individual type; progression across time.
anim_stuff = True
n_iterations = copy.deepcopy(i)
if anim_stuff:
    def animatePosterior_individualType(tp):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        x = abu.x_pointsDense
        line, = ax.plot(x,poly.polyval(x,abu.posterior_polyCoeff_typesList[0][tp]))
        def update(i):
            label = 'timestep {0}'.format(i)
            Tests.test_for_normalization(abu.posterior_polyCoeff_typesList[i][tp],abu.xrange)
            line.set_ydata(poly.polyval(x,abu.posterior_polyCoeff_typesList[i][tp]))
            ax.set_xlabel(label)
            return line,ax
        anim = FuncAnimation(fig,update,frames=n_iterations,interval=10)
        save=True
        if save:
            anim.save('posterior_type_{}.gif'.format(tp),dpi=100,writer='imagemagick')
        else:
            plt.show()
    for tp in range(len(abu.types)):
        animatePosterior_individualType(tp)



anim_stuff = True
n_iterations = copy.deepcopy(i)
if anim_stuff:
    def animateLikelihood_individualType(tp):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        x = abu.x_pointsDense
        line, = ax.plot(x,poly.polyval(x,abu.likelihood_polyCoeff_typesList[0][tp]))
        def update(i):
            label = 'timestep {0}'.format(i)
            # Tests.test_for_normalization(abu.likelihood_polyCoeff_typesList[i][tp],abu.xrange)
            line.set_ydata(poly.polyval(x,abu.likelihood_polyCoeff_typesList[i][tp]))
            ax.set_xlabel(label)
            return line,ax
        anim = FuncAnimation(fig,update,frames=n_iterations,interval=10)
        save=True
        if save:
            anim.save('likelihood_type_{}.gif'.format(tp),dpi=100,writer='imagemagick')
        else:
            plt.show()
    for tp in range(len(abu.types)):
        animateLikelihood_individualType(tp)





# prob_lh = np.array(prob_lh).astype('float32')
# prob_ori = np.array(prob_ori)
#
#
# print np.where(prob_lh==0)
# print(np.product(prob_lh))
# print(np.sum(np.log(prob_lh)))
# print(np.sum(np.log(prob_lh2)))
# if np.all(prob_lh):
#     print('All set')
# else:
#     print("This is not right")
#
# if np.all(prob_ori):
#     print('All set here too')
# else:
#     print("this is not right either.")