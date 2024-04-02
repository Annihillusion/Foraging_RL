import numpy as np
from Environment import CircularEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

num_envs = 2

# Parallel environments
# check_env(CircularEnv(render_mode=None))
env_config = {'render_mode': None}
env = make_vec_env(CircularEnv, n_envs=num_envs, env_kwargs=env_config)

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
# model.save("ppo_cartpole")

obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None

# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
