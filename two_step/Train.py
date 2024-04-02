import numpy as np
import Agents
import Env


def train_agent(agent, env, epochs, steps, record_path, model_path=None):
    """
    :param agent:
    :param env:
    :param epochs: the number of simulated C. elegans
    :param steps:
    :param record_path: directory & filename to save training records
    :param model_path: directory & filename to save agents' Q_tables
    :return:
    """
    state_collector = np.empty([epochs, steps])
    action_collector = np.empty([epochs, steps])
    reward_collector = np.empty([epochs, steps])

    for i in range(epochs):
        state = env.reset()
        agent.reset()
        for j in range(steps):
            state_collector[i, j] = state
            decay_coefficient = j/(steps*0.75) if j/(steps*0.75) < 1 else 1
            action = agent.choose_action(state, decay_coefficient)
            next_state, reward = env.step(state, action)
            agent.update(state, action, reward, next_state)
            state = next_state

            action_collector[i, j] = action
            reward_collector[i, j] = reward

    # rec[0]: state, rec[1]: action, rec[3]: reward
    # [s/a/r, epochs, time]
    np.save(record_path, np.array([state_collector, action_collector, reward_collector]))

