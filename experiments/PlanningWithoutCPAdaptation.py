import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import copy
import seaborn as sns
sns.set()
import logging.config

from src import arena
from src.mcts import mcts_arena_wrapper, mcts_sourcealgo, mcts_agent_wrapper
from src.utils import cloner, generate_init
from src.estimation import update_state, ABU_estimator_noapproximation

import configuration as config


no_experiments = 10
n_agents = 4 #including MCTS
n_agents_tracking = 1
n_max_iters_in_experiment = 150

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



for i in range(no_experiments):
    #Conducting individual experiments now.
    main_arena, agents = generate_init.generate_all(10,25,n_agents)

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
                _,_ = abu.estimate_allTypes(i)
                estimates, _ = abu.estimate_allTypes_withoutApproximation(i)
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
