import numpy as np
import Env
import Agents
from Train import train_agent


STEPS = 10000
EPOCHS = 20  # i.e. number of worms
record_dir = './records/'
model_dir = './models/'

roam_mat = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])
dwell_mat = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
state_reward = np.array([9, 4, 1])

env = Env.TwoStep(roam_mat, dwell_mat, state_reward)
agent = Agents.QAgent()

train_agent(agent, env, EPOCHS, STEPS, record_dir+'train_rec.npy')
