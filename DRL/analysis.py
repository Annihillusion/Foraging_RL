import numpy as np
import torch
import matplotlib.pyplot as plt

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


def smooth(y, radius, mode='two_sided', valid_only=False):
    """
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    """
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def draw_train_reward(file_path):
    npzfile = np.load(file_path)
    rewards = npzfile['reward']
    rewards = rewards.reshape([-1, *rewards.shape[2:]])
    mean = rewards.mean(axis=1)
    mean = smooth(mean, radius=10)
    std = rewards.std(axis=1)
    time_steps = rewards.shape[0]
    plt.plot(np.arange(time_steps), mean, '-')
    #plt.fill_between(np.arange(time_steps), mean - std, mean + std, alpha=0.2)
    plt.show()
    pass


def draw_average_reward(file_path):
    npzfile = np.load(file_path)
    rewards = npzfile['reward']
    avg_rewards = rewards.mean(axis=1)
    mean = avg_rewards.mean(axis=1)
    error = avg_rewards.std(axis=1) / np.sqrt(rewards.shape[2])
    x = np.arange(rewards.shape[0])
    plt.plot(x, mean, '-')
    plt.fill_between(x, mean - error, mean + error, alpha=0.2)
    plt.title(file_path[5:21])
    plt.show()


if __name__ == '__main__':
    file = 'logs/' + '2024-05-10 16-31' + '.npz'
    draw_average_reward(file)
