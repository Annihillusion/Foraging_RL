import numpy as np
import os

file = 'logs/' + '2024-05-14 15-14' + '.npz'
npzfile = np.load(file)
position, action, reward, energy = npzfile['position'], npzfile['action'], npzfile['reward'], npzfile['energy']
np.savez(file, position=position.squeeze(0), action=action.squeeze(0), energy=energy.squeeze(0), reward=reward.squeeze(0))

