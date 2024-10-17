import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir, device):
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
    """
    暂时用多个envs的平均替代多个agent的平均
    """
    npzfile = np.load(file_path)
    reward = npzfile['reward']
    mean = reward.mean(axis=1)
    error = reward.std(axis=1) / np.sqrt(reward.shape[1])
    mean = smooth(mean, radius=500)
    error = smooth(error, radius=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(reward.shape[0])
    ax.plot(x, mean, '-')
    # ax.fill_between(x, mean - error, mean + error, alpha=0.5)
    plt.title(file_path[5:21])
    plt.tight_layout()
    plt.show()


def roam_time_baseline(patch_size: str, time_bin=8, record_dir='records_5_18'):
    if patch_size == 'small':
        prefix = 'small_'
    elif patch_size == 'large':
        prefix = 'large_'
    else:
        raise ValueError(f'Choose from small and large')

    file_list = [prefix + str(i+1) + '.npz' for i in range(50)]
    collector = []
    for file in file_list:
        file_path = os.path.join(record_dir, file)
        npzfile = np.load(file_path)
        action = npzfile['action'][80000:, :]
        len_bin = action.shape[0] // time_bin
        roam_ratio = np.empty(time_bin)

        for i in range(time_bin):
            tmp = 1 - np.mean(action[i * len_bin: (i + 1) * len_bin, :], axis=0)
            roam_ratio[i] = np.mean(tmp)
        collector.append(roam_ratio)

    roam_data = np.array(collector)
    mean = np.mean(roam_data, axis=0)
    err = np.std(roam_data, axis=0) / np.sqrt(50*4)
    return mean*100, err*100


def small_large_contrast():
    small_mean, small_err = roam_time_baseline('small')
    large_mean, large_err = roam_time_baseline('large')

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(small_mean))
    ax.errorbar(x, large_mean, label='大菌斑', color='red')
    ax.errorbar(x, small_mean, label='小菌斑', color='blue')
    ax.set_xlabel('训练步数')
    ax.set_ylabel('平均漫游率(%)')
    ax.set_ylim([0, 20])
    ax.set_xticklabels(['', '80k-82.5k', '82.5k-85k', '85k-87.5k', '87.5k-90k', '90k-92.5k', '92.5k-95k', '95k-97.5k', '97.5k-100k'])
    plt.legend()
    plt.tight_layout()
    plt.show()


def roam_time_collector(prefix: str, time_bin=4, record_dir='records_5_18'):
    file_list = [prefix + '_' + str(i + 1) + '.npz' for i in range(50)]
    collector = []
    for file in file_list:
        file_path = os.path.join(record_dir, file)
        npzfile = np.load(file_path)
        action = npzfile['action']
        len_bin = action.shape[0] // time_bin
        roam_ratio = np.empty(time_bin)

        for i in range(time_bin):
            tmp = 1 - np.mean(action[i * len_bin: (i + 1) * len_bin, :], axis=0)
            roam_ratio[i] = np.mean(tmp)
        collector.append(roam_ratio)

    roam_data = np.array(collector)
    mean = np.mean(roam_data, axis=0)
    err = np.std(roam_data, axis=0) / np.sqrt(50)
    return mean*100, err*100


def roam_time(baseline: str, time_bin=4):
    # if baseline == 'small':
    #     mean, err = roam_time_baseline('small')
    # elif baseline == 'large':
    #     mean, err = roam_time_baseline('large')
    # else:
    #     raise ValueError('Choose from small and large')

    mean_large_to, err_large_to = roam_time_collector('large' + '_' + baseline, time_bin=time_bin)
    mean_small_to, err_small_to = roam_time_collector('small' + '_' + baseline, time_bin=time_bin)

    fig, ax = plt.subplots()
    # draw baseline

    ax.axvline(x=1, color='grey', linestyle='--')
    # draw transfer data
    mean_baseline, err_baseline = roam_time_baseline('large', time_bin=1)
    mean_baseline = mean_baseline.squeeze(0)
    err_baseline = err_baseline.squeeze(0)
    ax.plot([0, 2], [mean_baseline, mean_large_to[0]], linestyle=':', color='grey')
    ax.errorbar([0], [mean_baseline], yerr=[err_baseline], marker='^', markersize=8, markerfacecolor='none',
                label='基准值(大菌斑)', color=np.array([255,191,204,255])/255, capsize=3)
    ax.errorbar(np.arange(time_bin) + 2, mean_large_to, yerr=err_large_to, marker='^',
                label='大菌斑—>小菌斑' if baseline == 'small' else '大菌斑—>大菌斑',
                markersize=8, capsize=3, color=np.array([255,191,204,255])/255)

    mean_baseline, err_baseline = roam_time_baseline('small', time_bin=1)
    mean_baseline = mean_baseline.squeeze(0)
    err_baseline = err_baseline.squeeze(0)
    ax.plot([0, 2], [mean_baseline, mean_small_to[0]], linestyle=':', color='grey')
    ax.errorbar([0], [mean_baseline], yerr=[err_baseline], marker='^', markersize=8, markerfacecolor='none',
                label='基准值(小菌斑)', color=np.array([135,206,255,255])/255, capsize=3)
    ax.errorbar(np.arange(time_bin) + 2, mean_small_to, yerr=err_small_to, marker='^',
                label='小菌斑—>小菌斑' if baseline == 'small' else '小菌斑—>大菌斑',
                markersize=8, capsize=3, color=np.array([135,206,255,255])/255)

    ax.set_ylabel('漫游率(%)')
    ax.set_xlabel('训练步数')
    ax.set_xticklabels(['', '基准值', '', '0-5k', '5k-10k', '10k-15k', '15k-20k'])
    ax.set_ylim(5, 30)
    plt.legend()
    plt.tight_layout()
    plt.title(baseline)
    plt.show()


def roam_pos(file_path, time_bin=3, area_bin=3):
    """
    暂时用多个envs的平均替代多个agent的平均
    """
    npzfile = np.load(file_path)
    position, action = npzfile['position'], npzfile['action']
    num_steps = position.shape[0]
    position = position[0: num_steps//time_bin*time_bin].reshape([time_bin, -1])
    action = action.reshape([time_bin, -1])

    for i in range(time_bin):
        pass


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    file = 'logs/' + 'small_1' + '.npz'
    # draw_average_reward(file)
    # roam_time_baseline('small')
    #roam_time('logs/small_large_4.npz', time_bin=4)
    #roam_time('logs/large_small_4.npz', time_bin=4)
    #small_large_contrast()
    roam_time('small')
    roam_time('large')
