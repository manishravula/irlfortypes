import numpy as np

DICT_MOVES2ACTIONPROBS = {'-10': [1, 0, 0, 0, 0], '10': [0, 1, 0, 0, 0], '01': [0, 0, 1, 0, 0],
                                '0-1': [0, 0, 0, 1, 0],
                                '00': [0, 0, 0, 0, 1]}  # If the key is the difference between dest and curr,
ACTIONS = np.arange(5)
DICT_ACTION2INDEX = {'-10': 0, '10': 1, '01': 2, '0-1': 3,
                             '00': 4}  # Get the action index given the desired movement.
DICT_INDEX2ACTION = {0: '-10', 1: '10', 2: '01', 3: '0-1', 4: '00'}
ACTION2MOVEMENTVECTOR = np.array(
    [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])  # Given an action index, this array gives us the vector
ACTION2ORIENTATION = np.array(
    [np.pi / 2, 1.5 * np.pi, 0, np.pi])  # Given an action index, this array gives us the
ACTION2INDEX = {'u':0,'d':1,'l':2,'r':3,'a':4}
ACTIONHASHES =  [action for action in ACTION2INDEX.iterkeys()]
UNIVERSE = False
AIAGENT = True
ACTIONS2CHAR = ['u','d','l','r','p','n'] #p is for pick- which is a rename for load. n is for none
CHAR2ACTIONS = {'u':0,'d':1,'l':2,'r':3,'p':4,'n':5}