import numpy as np
import Agents
import Env
from matplotlib import pyplot as plt


STEPS = 10000
EPOCHS = 20

roam_mat = np.array([[0.3, 0.2, 0.5], [0.2, 0.7, 0.1], [0.4, 0.3, 0.3]])
dwell_mat = np.array([[0.2, 0.2, 0.6], [0.3, 0.1, 0.6], [0.5, 0.2, 0.3]])
state_reward = np.array([3, 2, 1])

env = Env.TwoStep(roam_mat, dwell_mat, state_reward)
agent = Agents.QAgent()

state = env.reset()
state_collector = np.empty([EPOCHS, STEPS])
action_collector = np.empty([EPOCHS, STEPS])
reward_collector = np.empty([EPOCHS, STEPS])

for i in range(EPOCHS):
    for j in range(STEPS):
        state_collector[i, j] = state
        decay_coefficient = j/(STEPS*0.75) if j/(STEPS*0.75) < 1 else 1
        action = agent.choose_action(state, decay_coefficient)
        next_state, reward = env.step(state, action)
        agent.update(state, action, reward, next_state)
        state = next_state

        action_collector[i, j] = action
        reward_collector[i, j] = reward

print(agent.q_table)
# rec[0]: state, rec[1]: action, rec[3]: reward
# [s/a/r, epochs, time]
np.save('train_rec.npy', np.array([state_collector, action_collector, reward_collector]))

#fig, ax = plt.subplots()
#ax.plot(range(100), reward_collector[0:100])
plt.show()
