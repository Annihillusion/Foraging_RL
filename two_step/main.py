import numpy as np
import Env
import Agents
from Train import train_agent


STEPS = 10000
EPOCHS = 20  # i.e. number of worms
record_dir = './records/'
model_dir = './models/'

roam_mat = np.array([[0.92708333, 0.07291667], [0.02909091, 0.97090909]])
dwell_mat = np.array([[0.95854922, 0.04145078], [0.01104972, 0.98895028]])
state_reward = np.array([10, 4])

env = Env.TwoStep(roam_mat, dwell_mat, state_reward)
agent = Agents.QAgent()

train_agent(agent, env, EPOCHS, STEPS, record_dir+'train_rec.npy')
