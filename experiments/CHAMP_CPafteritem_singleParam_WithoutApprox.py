
import matplotlib.pyplot as plt
# from src.arena import arena
# from src.agent_originaltypes import Agent
import src.champ as champ
from src.estimation.ABU_estimator_noapproximation import ABU
import numpy as np
import copy
import time
import numpy.polynomial.polynomial as poly
from matplotlib.animation import FuncAnimation
from tests import tests_helper as Tests
import configuration as config
from src.utils import generate_init as genInit
import seaborn as sns
sns.set()
import logging.config

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

logger.info("Logging configuration of the file")
for item in dir(config):
    if not item.startswith('__'):
        logger.info(item,getattr(config,item))




if config.INIT_TYPE == config.FROM_MEMORY:
    are, [a1,a2,a3] = genInit.generate_reload(3)
    logger.info("Reloading initial parameters")
else:
    are, [a1,a2,a3] = genInit.generate_all(10,20,3)
    logger.info("Generated new inital parameters")
curr_type = copy.deepcopy(a1.type)


abu_param_dict = {
              'radius_range': [.1,1],
              'angle_range':[.1,1],
              'resolution':5,
              'refit_density':20,
              'likelihood_polyDegree':9,
              'posterior_polyDegree':9,
              'prior_polyDegree':9,
              'visualize':config.VISUALIZE_ESTIMATION,
              'saveplots':config.VISUALIZE_ESTIMATION_SAVE}

abu = ABU(a1,are,abu_param_dict)
g1= are.grid_matrix
gm=[]
i=0
j=0

fitters = [abu.estimate_segmentForChamp_type0_withoutApprox, abu.estimate_segmentForChamp_type1_withoutApprox, abu.estimate_segmentForChamp_type2_withoutApprox,
           abu.estimate_segmentForChamp_type3_withoutApprox]
champ_config = champ.Config(fitters, length_min=5, length_mean=20, length_sigma=10.0, max_particles=1000, resamp_particles=1000)
champ = champ.Champ(champ_config)

time_array = []
prob_lh = []
prob_lh2 = []
prob_ori = []

estimates_array = []
estimates_array_noApprox = []
itemconsumed_time = []
curr_items = len(are.items)
time_array = []
mse_list = []
changepoint_time = []
changepoint_occured = False
resarray=[]


while not are.isterminal and j < 70:
    print("iter " + str(j))
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
    abu.get_likelihoodValues_allTypes()
    abu.calculate_modelEvidence(j)

    estimates, posterior = abu.estimate_allTypes(j)
    estimates_withoutApproximation, posterior_withoutApproximation = abu.estimate_parameter_allTypes_withoutApproximation(j, False)
    estimates_array.append(estimates)
    estimates_array_noApprox.append(estimates_withoutApproximation)

    mse_curr = abu.calculate_differenceInProbability()
    mse_list.append(mse_curr)
    # plt.plot(mse_curr,'-ro',label='mse vs parameter')
    # plt.show()
    abu.total_simSteps += 1

    champ.observe(j, action_and_consequence)
    print(estimates)
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
    if curr_items != len(are.items):
        itemconsumed_time.append(j)
        logger.info('Item consumed at time step {}'.format(j))
        if j>20 and j<50 and not changepoint_occured:
            newtp = np.random.randint(0,3,1)[0]
            while newtp == a1.type:
                newtp = np.random.randint(0,3,1)[0]
            changepoint_occured = True
            changepoint_time = j
            logger.info('Changepoint occured at {} and agent 1 changed from type {} to type {} '.format(j,curr_type,newtp))
            a1.type = newtp
            are.agents[0].tp = newtp

    curr_items = len(are.items)
    j += 1



    res = champ.backtrack(j - 1)
    logger.info('Results from CHAMP: {}'.format(res))
    logger.info("Main reported changepoint at {} from type {} to {}".format(changepoint_time,curr_type,a1.type))
    changepoint_time=0 #for not disturbing the plots
    resarray.append(res)

raise Exception()


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
    plt.axvline(changepoint_time)
    plt.title("Changepoint at {} from type 0 to type {}".format(changepoint_time, a1.type))
    image_name = './results/_entropy.png'
    plt.savefig(image_name,dpi=config.DPI)
    plt.close()

# MODEL EVIDENCE STUFF
mev_stuff = True
if mev_stuff:
    abu.model_evidence = np.array(abu.model_evidence)
    for tp in abu.types:
        plt.plot(np.cumsum(abu.model_evidence[:, tp]), label='Evidence of model {}'.format(tp))
        # plt.ylim(0, 30)

    plt.axvline(changepoint_time)
    plt.legend()
    plt.title("Model Evidence across time with true model {} and changed to {}".format(curr_type,are.agents[0].type))

    image_name = './results/_mevidence.png'

    plt.savefig(image_name, dpi=config.DPI)
    plt.close()

# ESTIMATES_STUFF:
est_stuff = True
if est_stuff:
    est_array = np.array(estimates_array)
    est_array_approx = np.array(estimates_array_noApprox)
    for tp in abu.types:
        # if tp==are.agents[0].type:
        plt.plot(est_array[:,tp,0], label='for type {}'.format(tp))
        plt.plot(est_array_approx[:,tp,0],label='for type {} without approximation'.format(tp))
    plt.axvline(changepoint_time)
    plt.legend()
    if abu.estimating_parameter=='view_radius':
        plt.title("ABU estim-evolution parameter {} with tv {} ".format(abu.estimating_parameter,are.agents[0].viewRadius_param))
        plt.axhline(linewidth=2,y=are.agents[0].viewRadius_param)
    else:
        plt.title("ABU estim-evolution parameter {} with tv {} ".format(abu.estimating_parameter,are.agents[0].viewAngle_param))
        plt.axhline(linewidth=2,y=are.agents[0].viewAngle_param)
    image_name = './results/_estimates.png'

    plt.savefig(image_name, dpi=config.DPI)

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
    plt.axvline(changepoint_time)
    plt.legend()
    plt.title("Evolution of estimated parameter's loglikelihood with changepoint from {} to {}".format(curr_type,a1.type))

    image_name = './results/_likelihood.png'

    plt.savefig(image_name,dpi=config.DPI)
    plt.close()

#Likelihood vs param value for individual type; progression across time.
anim_stuff = False
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
            anim.save('./results/posterior_type_{}.gif'.format(tp),dpi=100,writer='imagemagick')
        else:
            plt.show()
    for tp in range(len(abu.types)):
        animatePosterior_individualType(tp)



anim_stuff = False
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
            anim.save('./results/likelihood_type_{}.gif'.format(tp),dpi=100,writer='imagemagick')
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