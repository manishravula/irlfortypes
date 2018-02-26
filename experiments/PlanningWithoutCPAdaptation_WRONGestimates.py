import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import copy
import seaborn as sns

import pickle
import tempfile

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
logger = logging.getLogger('wrongEstimates')


experimentID = int(time.time())
logger.info("----Experiment without passing the right estimates to MCTS")
logger.info("-----------------------------Experiment {} begins--------------------------".format(experimentID))

no_experiments = config.N_EXPERIMENTS
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
        main_arena, agents = generate_init.generate_all(10,25,n_agents)
        r.ini_number_items = main_arena.no_items

        #Setting up ABU.
        abu = ABU_estimator_noapproximation.ABU(agents[0],main_arena,abu_param_dict)

        #Setting up the required lists. - reusable because of the iters.
        history = []

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
                    abu.calculate_modelEvidence(j)
                    _,_ = abu.estimate_allTypes(j)
                    estimates, _ = abu.estimate_allTypes_withoutApproximation(j)
                ag.execute_action(action_and_consequence)

            currstep_agentStates.append(currstep_agentStates[-2]) #like a dummy so that the mcts caller won't be upset.
            currstep_agentActions.append(currstep_agentActions[-2]) #Like a dummy so that the mcts caller won't be upset.

            currstep_history = update_state.history_step(currstep_arenaState,currstep_agentStates,currstep_agentActions)
            history.append(currstep_history)



            estimated_type = np.argmax(abu.model_evidence[-1])
            estimated_param = estimates[estimated_type][0]

            trackingAgentIds = [0]
            #TODO: Change the last number to grid_matrix_size
            tainfo = {'viewRadius_param':estimated_param,'type':estimated_type,'view_radius':estimated_param*main_arena.grid_matrix.shape[0]}
            # trackingAgentParameterEstimates = copy.deepcopy(history[0][1][0][0])

            #Dummy chaning the state of only one agent(0).
            trackingAgentParameterEstimates = [copy.deepcopy(history[0].agent_states[0])]
            trackingAgentParameterEstimates[0].update(tainfo)

            #REVERSING TO GIVE RANDOM ESTIMATES
            trackingAgentParameterEstimates.reverse()

            mcts_state = mctsagent.__getstate__()
            action_and_consequence = mctsagent.behave(history,trackingAgentIds,trackingAgentParameterEstimates)

            #Now that we have the true state of the last agent we need to rewrite over the dummy values.
            history[-1].agent_states.pop()
            history[-1].agent_actions.pop()
            history[-1].agent_states.append(mcts_state)
            history[-1].agent_actions.append(action_and_consequence)

            mctsagent.execute_action(action_and_consequence)

            main_arena.update_foodconsumption()
            main_arena.check_for_termination()
            j+=1

        r.left_over_items = len(main_arena.items)
        r.game_length = j
        for key,value in zip(r.__dict__.keys(),r.__dict__.values()):
            logging.info('result {}, {}'.format(key,value))

        logger.info("End of expriment {}".format(i))
        final_results.append(r)
    config.SMSClient.messages.create(to=config.to_number,from_=config.from_number,body="Experiments ID:{} with false information finished succesfully".format(experimentID))
    resultname = str(experimentID)+'_resultswithWrongEstimation'

    with open(resultname,'wb') as handle:
        pickle.dump(final_results,handle)

except Exception as e:
    logging.exception('Experiment failed')
    config.SMSClient.messages.create(to=config.to_number,from_=config.from_number,
                                     body="Experiment with false information exception occured {}! Check logs!".format(e))

