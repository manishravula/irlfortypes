import copy
import pickle
import time

import numpy as np
import seaborn as sns
import pickle
import glob
sns.set()
import logging.config

from src.mcts import mcts_agent_wrapper
from src.utils import generate_init
from src.estimation import update_state, ABU_estimator_noapproximation

import configuration as config
from src.utils import banner
import logging

logging.config.dictConfig(config.LOGGING_CONFIG)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str,
help="The type of experiment being performed. cp-oracle\n,cp-nooracle\n")
parser.add_argument("--settingsfolder", type=str, help="File specifying initial conditions for each experiment to be performed.")
parser.add_argument("--mcts_setting",type=str,help="Setting about using heuristic for MCTS")
parser.add_argument("--ch_length_min",type=int,help="CHAMP's minimum length of the segment parameter")
parser.add_argument("--ch_length_mean",type=int,help="CHAMP's mean length of the segment")
parser.add_argument("--ch_length_sigma",type=int,help="CHAMP's variance in length of the segment")
parser.add_argument("--ch_maxparticles",type=int,help="CHAMP's maximum number of particles")
parser.add_argument("--ch_resample_particles",type=int,help="Number of resampling in CHAMP's particles")


args = parser.parse_args()

exptype = args.type
expfile = args.settingsfolder

possible_types = ['cp-oracle','cp-nooracle']
possible_mctssettings = ['heuristic','absolute']

#sanity checks
if exptype is None:
    raise Exception("No valid experiment type given")
elif exptype not in possible_types:
    raise Exception("Given type {} doesn't match any one of the possible types".format(exptype))


if expfile is None:
    raise Exception("No experiment conditions file given")

if args.mcts_setting not in possible_mctssettings:
    raise Exception("Wrong MCTS settings described")

#default values
if args.ch_length_min is None:
    args.ch_length_min = 5
if args.ch_length_mean is None:
    args.ch_length_mean = 20
if args.ch_length_sigma is None:
    args.ch_length_sigma = 10
if args.ch_maxparticles is None:
    args.ch_maxparticles = 1000
if args.ch_resample_particles is None:
    args.ch_resample_particles = 1000

experimentID = int(time.time())

logger = logging.getLogger(args.type+str(experimentID))
logger.info("-----Experiment type {} ------ ".format(args.type))
logger.info("-----------------------------Experiment ID {} begins--------------------------".format(experimentID))


#Find number of data files.
n_files = len(glob.glob(expfile+'*'))

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
changepoint_tmin = 40
changepoint_tmax = 70
changepoint_set = False

class result(object):
    def __init__(self,ID):
        self.experimentID = ID

try:

    for i in range(no_experiments):
        r = result(i)
        logger.info(banner.horizontal('Experiment {}'.format(i)))
        #Conducting individual experiments now.
        main_arena, agents = generate_init.generate_from_savedexperiment(expfile,i,n_agents)
        r.ini_number_items = main_arena.no_items
        r.precp_type = copy.deepcopy(agents[0].type)

        #Setting up ABU.
        abu = ABU_estimator_noapproximation.ABU(agents[0],main_arena,abu_param_dict)

        #Setting up the required lists. - reusable because of the iters.
        history = []
        est = []

        #Setting up MCTS agent.
        _dummy_agent_for_config = main_arena.agents.pop()
        mctsagent = mcts_agent_wrapper.mcts_agent(_dummy_agent_for_config.capacity_param, _dummy_agent_for_config.viewRadius_param, _dummy_agent_for_config.viewAngle_param, _dummy_agent_for_config.type,
                              _dummy_agent_for_config.curr_position, main_arena)

        main_arena.agents.append(mctsagent)

        changepoint_time = 10000
        newtp = np.random.randint(0,3,1)[0]
        while newtp == main_arena.agents[0].type:
            newtp = np.random.randint(0,3,1)[0]
        changepoint_postType = newtp

        # logger.info("Preset changepoint is at {}".format(changepoint_time))

        j=0
        #Beginning loop
        while (j<n_max_iters_in_experiment) and not main_arena.isterminal:
            logger.info('iter {} in experiment {}'.format(j,i))
            #Changing type
            if j==changepoint_time:
                main_arena.agents[0].type = changepoint_postType
                if args.type == 'cp-nooracle':
                    pass
                else:
                    history = [] #resetting history.
                logger.info("Agent 0/1's type changed from to {}".format(changepoint_postType))
                r.postcp_time = changepoint_postType

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

                    if args.type == 'cp-oracle' and j == changepoint_time:
                        # resetting
                        abu.parameter_posterior = abu.reset_get_fresh_prior()
                    abu.calculate_modelEvidence(j)
                    _, _ = abu.estimate_allTypes(j)
                    estimates, _ = abu.estimate_parameter_allTypes_withoutApproximation(j, False)

                    est.append(estimates)

                ag.execute_action(action_and_consequence)

            currstep_agentStates.append(currstep_agentStates[-2]) #like a dummy so that the mcts caller won't be upset.
            currstep_agentActions.append(currstep_agentActions[-2]) #Like a dummy so that the mcts caller won't be upset.

            currstep_history = update_state.history_step(currstep_arenaState,currstep_agentStates,currstep_agentActions)
            history.append(currstep_history)



            estimated_type = np.argmax(abu.model_evidence[-1])
            estimated_param = estimates[estimated_type][0]

            trackingAgentIds = [0]
            #TODO: WHAT ABOUT THE TYPE??????
            #TODO: Change the last number to grid_matrix_size
            tainfo = {'viewRadius_param':estimated_param,'type':estimated_type,'view_radius':estimated_param*main_arena.grid_matrix.shape[0]}
            # trackingAgentParameterEstimates = copy.deepcopy(history[0][1][0][0])

            #Dummy chaning the state of only one agent(0).
            trackingAgentParameterEstimates = [copy.deepcopy(history[0].agent_states[0])]
            trackingAgentParameterEstimates[0].update(tainfo)

            mcts_state = mctsagent.__getstate__()

            if args.mcts_setting == 'heuristic':
                if config.N_MAXITERS_IN_EXPERIMENTS-j>config.MAX_HEURISTIC_ROLLOUT_DEPTH:
                    rolloutdepth = config.MAX_HEURISTIC_ROLLOUT_DEPTH
                else:
                    rolloutdepth = config.N_MAXITERS_IN_EXPERIMENTS - j
                action_and_consequence = mctsagent.behave(history,trackingAgentIds,trackingAgentParameterEstimates,rolloutdepth)
            else:
                action_and_consequence = mctsagent.behave(history,trackingAgentIds,trackingAgentParameterEstimates,config.MAX_ROLLOUT_DEPTH)


            #Now that we have the true state of the last agent we need to rewrite over the dummy values.
            history[-1].agent_states.pop()
            history[-1].agent_actions.pop()
            history[-1].agent_states.append(mcts_state)
            history[-1].agent_actions.append(action_and_consequence)

            mctsagent.execute_action(action_and_consequence)
            n_items = len(main_arena.items)
            main_arena.update_foodconsumption()
            n_items_new = len(main_arena.items)
            if n_items_new!=n_items:
                logger.info("Item consumed")
                if j>changepoint_tmin and j<changepoint_tmax and not changepoint_set:
                    changepoint_time = j+1
                    changepoint_set = True


            main_arena.check_for_termination()
            j+=1

        r.left_over_items = len(main_arena.items)
        r.game_length = j
        r.tags = args
        for key,value in zip(r.__dict__.keys(),r.__dict__.values()):
            logging.info('result {}, {}'.format(key,value))

        logger.info("End of expriment {}".format(i))
        final_results.append(r)
    resultname = str(experimentID)+'_'+args.type+'_'+args.mcts_setting

    config_forsaving = {}
    p1 = config.__dict__
    for key in p1.keys():
        p1[key] = str(p1[key])
    config_forsaving.update(p1)
    config_forsaving.update(args.__dict__)
    final_results.append(config_forsaving)
    with open(resultname,'wb') as handle:
        pickle.dump(final_results,handle,protocol=pickle.HIGHEST_PROTOCOL)

    config.SMSClient.messages.create(to=config.to_number,from_=config.from_number,body="Experiments ID:{} with args {} finished succesfully".format(experimentID,args))

except Exception as e:
    logging.exception("Experiment failed")
    config.SMSClient.messages.create(to=config.to_number,from_=config.from_number,body="Experiment with args {} exception {} ! Check logs!".format(args,e))


