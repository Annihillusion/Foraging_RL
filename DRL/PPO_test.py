import numpy as np
from Environment import CircularEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO

num_envs = 2

# Parallel environments
# check_env(CircularEnv())
env_config = {'render_mode': None}
env = make_vec_env(CircularEnv, n_envs=num_envs, env_kwargs=env_config)

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)

# model.save("ppo_cartpole")

obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None

# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
for _ in range(100):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones

env.close()
