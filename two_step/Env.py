import numpy as np


class TwoStep:
    def __init__(self, roam_mat, dwell_mat, reward, state_dim=3):
        self.trans_mat = np.array([roam_mat, dwell_mat])
        self.state_dim = roam_mat.shape[0]
        self.random_gen = np.random.default_rng(202434)
        self.state_reward = reward
        self.state_dim = state_dim
        self.curr_state = 0

    def step(self, state, action):
        # state = 0: patch edge
        # action = 0: roaming, action = 1: dwelling
        new_state = self.random_gen.choice(self.state_dim, 1, p=self.trans_mat[action, state])
        return new_state[0], self.state_reward[new_state]

    def reset(self):
        self.curr_state = 0
        return self.curr_state


