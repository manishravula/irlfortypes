import time
import inspect


DEBUG=False

NO_MOVEMENTS = 4
LOAD_ACTION_PROBABILITY = []



#Agents to use
from src.agents.agent_originaltypes import Agent as AGENT_CURR
from src.arena import arena as ARENA_CURR

#Arena to use
#src.generation_init config
COOPERATION_INDEX = .8 # index*grid_min
NO_TYPES = 4
N_AGENTS = 3
#Initialization to use
FROM_MEMORY = 1
FROM_NEW = 0
INIT_TYPE = FROM_MEMORY
# INIT_TYPE = 'random'




#Visualization stuff
#visualization of the estimation process
VISUALIZE_ESTIMATION = False
#Save figures or not switch
VISUALIZE_ESTIMATION_SAVE = False
DPI = 300
#Visualize the simulation or not.
VISUALIZE_SIM = False


#MCTS Related
N_ROLLOUTS = 30
ROLLOUT_DEPTH = 30





#logging config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'infoterminal': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            # 'stream': 'sys.stdout'
        },
        'Debug_File':{
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': '../logging/debug_log_{}-{}.txt'.format(time.asctime().replace(' ',''),inspect.stack()[-1][1].replace('/',''))
        },
    },
    'loggers': {
        '': {
            'handlers': ['infoterminal','Debug_File'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}
