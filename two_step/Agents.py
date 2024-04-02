import numpy as np


class Agent:
    def __init__(self, env, alpha, gamma, epsilon):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon


class SARSA(Agent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        super(SARSA, self).__init__(env, alpha, gamma, epsilon)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # 随机选择动作
        else:
            return np.argmax(self.Q[state, :])  # 选择Q值最大的动作

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.Q[state, action]
        next_q = self.Q[next_state, next_action]
        self.Q[state, action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)


class QAgent:
    def __init__(self, state_dim=3, action_dim=2, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_dim, action_dim))
        self.np_ramdom = np.random.default_rng(202434)

    def choose_action(self, state, decay_coefficient):
        exploration_rate = self.exploration_rate * (1 - decay_coefficient)
        if np.random.uniform(0, 1) < exploration_rate:
            # 探索：随机选择动作
            action = self.np_ramdom.choice(self.action_dim)
        else:
            # 利用 Q 表选择最优动作
            action = np.argmax(self.q_table[state])
        return action

    def update(self, state, action, reward, next_state):
        # 更新 Q 表
        old_q_value = self.q_table[state, action]
        next_max_q_value = np.max(self.q_table[next_state])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q_value - old_q_value)
        self.q_table[state, action] = new_q_value

    def save_q_table(self, path):
        np.save(path, self.q_table)

    def load_q_table(self, path):
        q_table = np.load(path)
        self.q_table = q_table

    def reset(self):
        self.q_table = np.zeros((self.state_dim, self.action_dim))
