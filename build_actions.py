import numpy as np
from experiments import configuration as config

n_movs = config.NO_MOVEMENTS

action_probability = .25 * np.ones(n_movs+1)


# up,down,right,left, up-right, up-left, down-right, down-left

if n_movs==8:
    dict_moves_actionsProbs = {'-10': [1, 0, 0, 0,0,0,0],
                           '10': [0, 1, 0, 0,0,0,0,0],
                           '01': [0, 0, 1, 0,0,0,0,0],
                           '0-1': [0, 0, 0, 1,0,0,0,0],
                           '-11':[0,0,0,0,1,0,0,0],
                           '-1-1':[0,0,0,0,0,1,0,0],
                           '11':[0,0,0,0,0,0,1,0],
                           '1-1':[0,0,0,0,0,0,0,1]
                           }  # If the key is the difference between dest and curr,
else:
    dict_moves_actionsProbs={'-10': [1, 0, 0, 0],
                           '10': [0, 1, 0, 0],
                           '01': [0, 0, 1,0],
                           '0-1': [0, 0, 0, 1]}



# The list returns the action probs.
actions = np.arange(n_movs+1)


# Get the action index given the desired movement.
if n_movs==4:
    dict_actiontoIndices = {'-10': 0, '10': 1, '01': 2, '0-1': 3}
else:
    dict_actiontoIndices = {'-10': 0, '10': 1, '01': 2, '0-1': 3, '-11':4,'-1-1':5,'11':6,'1-1':7}


# Given an action index, this array gives us the vector to add to get the new position

if n_movs==4:
    action_to_movements = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])
else:
    action_to_movements = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0],[-1,1],[-1,-1],[1,1],[1,-1]])

if n_movs==4:
    action_to_orientations = np.array([np.pi / 2, 1.5 * np.pi, 0, np.pi])  # Given an action index, this array gives us what the orientation should be.
else:
    action_to_orientations = np.array([np.pi / 2, 1.5 * np.pi, 0, np.pi, np.pi/4, np.pi*3/4.0, np.pi*7/4.0, np.pi*5/4.0])
