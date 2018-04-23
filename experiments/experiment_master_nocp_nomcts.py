import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import copy
import seaborn as sns
import pickle
import glob
sns.set()
import logging.config

from src import arena
from src.mcts import mcts_arena_wrapper, mcts_sourcealgo, mcts_agent_wrapper
from src.utils import cloner, generate_init
from src.estimation import update_state, ABU_estimator_noapproximation

import configuration as config
from src.utils import banner
import logging

logging.config.dictConfig(config.LOGGING_CONFIG)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--settingsfolder",type=str,help='Setings folder for initializations during experiments \n')
args = parser.parse_args()
expfile = args.settingsfolder

# possible_types = ['nocp-correstim','nocp-wrongestim']
# possible_mctssettings = ['heuristic','absolute']

#sanity checks

if expfile is None:
    raise Exception("No experiment conditions folder given")


experimentID = int(time.time())

logger = logging.getLogger(str(experimentID))
logger.info("-----------------------------Experiment ID {} begins--------------------------".format(experimentID))


#Find number of data files.
n_files = len(glob.glob(expfile+'/*'))

no_experiments = n_files
logger.info("Configuration of the experiment: no_experiments: {}".format(no_experiments))

n_max_iters_in_experiment = config.N_MAXITERS_IN_EXPERIMENTS
logger.info("Configurations of the experiment: n_max_iters_experiment: {}".format(n_max_iters_in_experiment))

n_agents = 4 #including MCTS
logger.info("Configurations of the experiment: n_agents: {}".format(n_agents))

n_agents_tracking = 1
logger.info("Configurations of the experiment: n_agents_tracking: {}".format(n_agents_tracking))


abu_param_dict = {
              'radius_range': [.1,1],
              'angle_range':[.1,1],
              'resolution':10,
              'refit_density':20,
              'likelihood_polyDegree':9,
              'posterior_polyDegree':9,
              'prior_polyDegree':9,
              'visualize':config.VISUALIZE_ESTIMATION,
              'saveplots':config.VISUALIZE_ESTIMATION_SAVE}
final_results = []

class result(object):
    def __init__(self,ID):
        self.experimentID = ID

try:

    for i in range(no_experiments):
        r = result(i)
        logger.info(banner.horizontal('Experiment {}'.format(i)))
        #Conducting individual experiments now.
        main_arena, agents = generate_init.generate_from_savedexperiment(expfile,i+4,n_agents)
        r.ini_number_items = main_arena.no_items

        #Setting up ABU.
        abu = ABU_estimator_noapproximation.ABU(agents[0],main_arena,abu_param_dict)

        #Setting up the required lists. - reusable because of the iters.
        history = []
        est = []
        mevd = []
        tpost_list = []
        ppost_list = []
        esta_list = []
        estv_list = []
        estimates = []
        itemcons_time =[]
        mse_list = []


        #Setting up MCTS agent.
        _dummy_agent_for_config = main_arena.agents.pop()
        mctsagent = mcts_agent_wrapper.mcts_agent(_dummy_agent_for_config.capacity_param, _dummy_agent_for_config.viewRadius_param, _dummy_agent_for_config.viewAngle_param, _dummy_agent_for_config.type,
                              _dummy_agent_for_config.curr_position, main_arena)

        main_arena.agents.append(mctsagent)

        j=0
        #Beginning loop
        while (j<n_max_iters_in_experiment) and not main_arena.isterminal:
            logger.info('iter {} in experiment {}'.format(j,i))
            abu.all_agents_behave()
            currstep_arenaState = main_arena.__getstate__()
            currstep_agentStates = [ag.__getstate__() for ag in main_arena.agents[:-1]]
            currstep_agentActions = [None for ag in main_arena.agents[:-1]] #initialization


            for m,ag in enumerate(main_arena.agents[:-1]):
                actionprobs = ag.behave(False)
                action_and_consequence = ag.behave_act(actionprobs)
                currstep_agentActions[m] = action_and_consequence

                if ag is main_arena.agents[0]:
                    abu.all_agents_imitate(action_and_consequence)
                    abu.all_agents_calc_likelihood(action_and_consequence)
                    _ = abu.fit_likelihoodPolynomial_allTypes(action_and_consequence)
                    abu.get_likelihoodValues_allTypes()
                    mse_list.append(abu.calculate_differenceInProbability())

                    # if args.type == 'cp-oracle' and j == changepoint_time:
                        # resetting
                        # abu.reset(0,1,True)

                    # abu.calculate_modelEvidence(j)
                    # _, _ = abu.estimate_allTypes(j)
                    # estimates _ = abu.estimate_parameter_allTypes_withoutApproximation(j, True)
                    #
                    # mev = abu.calc_modelEvidence_noApproximation(j,estimates)
                    # estimated_type = abu.estimate_type_withoutApproximation(j,estimates)
                    #
                    # estimated_param = estimates[estimated_type][0]
                    inference_result,[tpost,ppost] = abu.infer_typeAndParameter()
                    estimated_params = [inference_result.param_res_list[edx].estim_sample for edx in range(4)]
                    estimated_params_variance = [inference_result.param_res_list[edx].estim_variance for edx in range(4)]
                    estimated_type = inference_result.type
                    estimated_param = estimated_params[estimated_type]

                    mev = inference_result.mevd
                    est.append([estimated_type,estimated_param])
                    mevd.append(mev)

                    ####
                    #DEBUG BREAKS
                    tpost_list.append(tpost)
                    ppost_list.append(ppost)
                    esta_list.append(estimated_params)
                    estv_list.append(estimated_params_variance)
                    ####

                ag.execute_action(action_and_consequence)

            currstep_agentStates.append(currstep_agentStates[-2]) #like a dummy so that the mcts caller won't be upset.
            currstep_agentActions.append(currstep_agentActions[-2]) #Like a dummy so that the mcts caller won't be upset.

            currstep_history = update_state.history_step(currstep_arenaState,currstep_agentStates,currstep_agentActions)
            history.append(currstep_history)

            i1 = len(main_arena.items)
            main_arena.update_foodconsumption()
            i2 = len(main_arena.items)
            if i2<i1:
                itemcons_time.append(j)

            main_arena.check_for_termination()
            j+=1

        r.left_over_items = len(main_arena.items)
        r.game_length = j
        r.tags = args
        for key,value in zip(r.__dict__.keys(),r.__dict__.values()):
            logging.info('result {}, {}'.format(key,value))

        logger.info("End of experiment {}".format(i))
        final_results.append(r)


        tpost_list = np.array(tpost_list)
        esta_list = np.array(esta_list)
        estv_list = np.array(estv_list)
        ll_list = np.array(abu.likelihood_dense_typesList)
        mse_list = np.array(mse_list)
        #MSE plots between
        plt.figure()
        for gdx in range(4):
            plt.subplot(4, 1, gdx + 1)
            xvals = np.arange(j)
            yvals = np.mean(mse_list[:, gdx], axis=1)
            std = np.std(mse_list[:, gdx],axis=1)
            ymin = yvals - std / 2
            ymax = yvals + std / 2
            if gdx == agents[0].type:
                c = 'g'
            else:
                c = 'b'
            plt.title("msd for individual types. True type {} and true param {}".format(agents[0].type, agents[0].viewRadius_param))
            plt.plot(xvals, yvals, 'o--', color=c, alpha=1, linewidth=5)
            plt.ylim(0,1.5)
            plt.fill_between(xvals, ymin, ymax, color=c, alpha=.4)

        #likelihood plots
        plt.figure()
        plt.title("mean likelihoods for individual types. True type {} and true param {}".format(agents[0].type,agents[0].viewRadius_param))
        for gdx in range(4):
            plt.subplot(4,1,gdx+1)
            xvals = np.arange(j)
            yvals = np.mean(ll_list[:,gdx],axis=1)
            std = np.std(ll_list[:,gdx],axis=1)
            ymin = yvals - std/2
            ymax = yvals + std/2
            if gdx==agents[0].type:
                c = 'g'
            else:
                c = 'b'
            plt.plot(xvals,yvals,'o--',color=c,alpha=1,linewidth=5)
            plt.ylim(0,1)
            plt.fill_between(xvals,ymin,ymax,color=c, alpha=.4)

        #PLOTTING POSTERIOR OVER TYPES
        plt.interactive(False)
        plt.figure()
        plt.title("Posterior over types vs time. True type: {}".format(agents[0].type))

        plt.ylim((0,1))
        for gdx in range(4):
            plt.plot(tpost_list[:,gdx],'*--',markersize=8,linewidth=5,label='type {}'.format(gdx))
            for time in itemcons_time:
                plt.axvline(time,color='r',linewidth=1)

        plt.legend()



        #parameter estimate plots
        plt.figure()
        plt.title("Parameter estimates for individual types. True type {} and true param {}".format(agents[0].type,agents[0].viewRadius_param))
        for gdx in range(4):
            plt.subplot(4,1,gdx+1)
            plt.ylim((0,1))
            plt.title("Estimates vs time for type {}".format(gdx))
            x_points = np.arange(j)
            y = esta_list[:,gdx]
            var = estv_list[:,gdx]
            ymin = y-var/2
            ymax = y+var/2
            plt.plot(x_points,esta_list[:,gdx],'+--',color = '#339caf', alpha = 1,linewidth=6)
            plt.fill_between(x_points,ymin,ymax,color = '#339cbf', alpha = 0.4)
            if gdx==agents[0].type:
                plt.axhline(agents[0].viewRadius_param)
            for time in itemcons_time:
                plt.axvline(time,color='r',linewidth=1)
        plt.show()






    # config.SMSClient.messages.create(to=config.to_number,from_=config.from_number,body="Experiments ID:{} with args {} finished succesfully".format(experimentID,args))
    resultname = str(experimentID)+'_'
    config_forsaving = {}
    p1 = config.__dict__
    for key in p1.keys():
        p1[key] = str(p1[key])
    config_forsaving.update(p1)
    config_forsaving.update(args.__dict__)
    final_results.append(config_forsaving)
    with open(resultname,'wb') as handle:
        pickle.dump(final_results,handle,protocol=pickle.HIGHEST_PROTOCOL)

except Exception as e:
    logging.exception("Experiment failed")
    # config.SMSClient.messages.create(to=config.to_number,from_=config.from_number,body="Experiment with args {} exception {} ! Check logs!".format(args,e))
