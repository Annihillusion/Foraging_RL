import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def judge_area(position, radius, num_area):
    # dist_from_edge = radius - np.linalg.norm(position)
    ratio = np.linalg.norm(position) / radius
    for i in range(num_area):
        if ratio > np.sqrt(i) / np.sqrt(num_area) and ratio <= np.sqrt(i+1) / np.sqrt(num_area):
            return num_area - i - 1
    # return int(ratio // (1 / num_area))


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


def draw_average_reward(file_path):
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


def roam_time_baseline(patch_size: str, records_dir: str, repeat_num: int, time_bin=1):
    if patch_size == 'small':
        prefix = 'small_'
    elif patch_size == 'large':
        prefix = 'large_'
    else:
        raise ValueError(f'Choose from small and large')

    file_list = [prefix + str(i+1) + '.npz' for i in range(repeat_num)]
    collector = []
    for file in file_list:
        file_path = os.path.join(records_dir, file)
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
    err = np.std(roam_data, axis=0) / np.sqrt(repeat_num * 4)
    return mean*100, err*100


def small_large_contrast():
    small_mean, small_err = roam_time_baseline('small')
    large_mean, large_err = roam_time_baseline('large')

    fig, ax = plt.subplots()
    x = np.arange(len(small_mean)) + 1
    ax.errorbar(x, small_mean*100, yerr=small_err*100, label='small patch')
    ax.errorbar(x, large_mean*100, yerr=large_err*100, label='large patch')
    #ax.set_ylim([0, 20])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def roam_time_transfer(prefix: str, records_dir: str, repeat_num: int, time_bin=4):
    file_list = [prefix + '_' + str(i + 1) + '.npz' for i in range(repeat_num)]
    collector = []
    for file in file_list:
        file_path = os.path.join(records_dir, file)
        npzfile = np.load(file_path)
        action = npzfile['action']
        len_bin = action.shape[0] // time_bin
        roam_ratio = np.empty(time_bin)

        # 若干并行线程的平均值作为该次实验的数据，后续有待改进
        for i in range(time_bin):
            tmp = 1 - np.mean(action[i * len_bin: (i + 1) * len_bin, :], axis=0)
            roam_ratio[i] = np.mean(tmp)
        collector.append(roam_ratio)

    roam_data = np.array(collector)
    mean = np.mean(roam_data, axis=0)
    err = np.std(roam_data, axis=0) / np.sqrt(repeat_num)
    return mean*100, err*100


def get_repeat_number(file_dir: str):
    max_num = 0
    file_list = os.listdir(file_dir)
    for file_name in file_list:
        pattern = re.search(r'\d+', file_name)
        number = 0 if pattern is None else int(pattern.group())
        if number > max_num:
            max_num = number
    return max_num


def roam_time(records_dir, baseline, time_bin=4):
    # 获取实验重复次数，计算standard_error需要
    repeat_num = get_repeat_number(records_dir)

    if baseline == 'small':
        mean_baseline, err_baseline = roam_time_baseline('small', records_dir, repeat_num, time_bin=1)
    elif baseline == 'large':
        mean_baseline, err_baseline = roam_time_baseline('large', records_dir, repeat_num, time_bin=1)
    else:
        raise ValueError('Choose from small and large')
    mean_baseline = mean_baseline.squeeze(0)
    err_baseline = err_baseline.squeeze(0)

    mean_to_large, err_to_large = roam_time_transfer(baseline + '_large', records_dir, repeat_num, time_bin=time_bin)
    mean_to_small, err_to_small = roam_time_transfer(baseline + '_small', records_dir, repeat_num, time_bin=time_bin)

    fig, ax = plt.subplots(dpi=200)
    # draw baseline
    ax.errorbar([0], [mean_baseline], yerr=[err_baseline], marker='^', markersize=8, markerfacecolor='none',
                label='baseline(small)' if baseline == 'small' else 'baseline(large)', capsize=3)
    ax.axvline(x=1, color='grey', linestyle='--')
    ax.plot([0, 2], [mean_baseline, mean_to_small[0]], linestyle=':', color='grey')
    ax.plot([0, 2], [mean_baseline, mean_to_large[0]], linestyle=':', color='grey')
    # draw transfer data
    color_large = [np.array([164, 48, 42, 255]) / 255, np.array([245, 194, 204, 255]) / 255]
    color_small = [np.array([0, 0, 237, 255]) / 255, np.array([135, 206, 255, 255]) / 255]
    ax.errorbar(np.arange(time_bin) + 2, mean_to_large, yerr=err_to_large, marker='^',
                label='small—>large' if baseline == 'small' else 'large—>large',
                markersize=8, capsize=3, color=color_large[0] if baseline == 'large' else color_small[0])
    ax.errorbar(np.arange(time_bin) + 2, mean_to_small, yerr=err_to_small, marker='^',
                label='small—>small' if baseline == 'small' else 'large—>small',
                markersize=8, capsize=3, color=color_large[1] if baseline == 'large' else color_small[1])

    ax.set_ylabel('Roam rate(%)')
    ax.set_xlabel('Steps after transfer')
    ax.set_xticklabels(['', 'baseline', '', '0-5k', '5k-10k', '10k-15k', '15k-20k'])
    #ax.set_ylim(5, 25)
    plt.title('Roam rate after transfer ' + f'({baseline})')
    plt.legend()
    plt.tight_layout()
    _, exp_name = os.path.split(records_dir)
    plt.savefig(os.path.join('exp_plots', exp_name, 'roam_time_' + baseline))
    plt.show()


def time_pos_collector(file_path, radius, time_bin=3, area_bin=3):  # 0: edge, 1: middle, 2: center
    npzfile = np.load(file_path)
    position = npzfile['position'][1:]
    num_steps = position.shape[0]
    position = position[0: num_steps//time_bin*time_bin].reshape([time_bin, -1, 2])

    result = np.empty([time_bin, area_bin])
    for i in range(time_bin):
        tmp = np.zeros(area_bin)
        for j in range(position.shape[1]):
            area = judge_area(position[i, j], radius, area_bin)
            tmp[area] += 1
        result[i] = tmp / position.shape[1]
    return result


def time_pos(pre, post, time_bin=4, area_bin=4):
    repeat_num = 50
    file_list = [pre + '_' + str(i + 1) + '.npz' for i in range(repeat_num)]
    collector = []
    # for file in file_list:
    #     file_path = os.path.join('exp_records', 'origin', file)
    #     tmp = time_pos_collector(file_path, 6 if pre == 'small' else 12)
    #     tmp = tmp[-1]
    #     collector.append(tmp)
    # baseline = np.array(collector)
    # baseline_data = np.mean(baseline, axis=0)
    # baseline_data = np.expand_dims(baseline_data, axis=0)

    file_list = [pre + '_' + post + '_' + str(i + 1) + '.npz' for i in range(repeat_num)]
    collector = []
    for file in file_list:
        file_path = os.path.join('exp_records', 'origin', file)
        tmp = time_pos_collector(file_path, 6 if post == 'small' else 12, time_bin=time_bin, area_bin=area_bin)
        collector.append(tmp)
    all_data = np.array(collector)
    transfer_data = np.mean(all_data, axis=0)
    # plot_data = np.concatenate((baseline_data, transfer_data), axis=0).transpose(1, 0)
    plot_data = transfer_data.transpose(1, 0)
    fig, ax = plt.subplots()
    bar_width = 0.2 * 0.4
    x = np.arange(time_bin)
    # location_labels = ['Edge', 'Middle', 'Center']
    for i in range(area_bin):
        x_0 = x - bar_width * area_bin / 2
        x_plot = x_0 + i * bar_width + bar_width / 2
        ax.bar(x_plot, plot_data[i], width=bar_width)
    # 中文标注，运行前需使用matplotlib相关函数设置字体
    # ax.set_xticks(x, ['基准值(小菌斑)' if pre == 'small' else '基准值(大菌斑)', '阶段1', '阶段2', '阶段3'])
    # ax.set_xticks(x, ['baseline(small)' if pre == 'small' else 'baseline(large)', 'period 1', 'period 2', 'period 3'])
    ax.set_xticks(x, ['period 1', 'period 2', 'period 3', 'period 4'])
    ax.set_ylabel('normalized frequency')
    ax.set_ylim([0, 0.75])
    ax.set_title(pre + '-' + post)

    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 训练记录目录：exp_records，绘图保存目录：exp_plots
    experiment_name = 'edge_continue'
    records_dir = os.path.join('exp_records', experiment_name)

    roam_time(records_dir, 'small')
    # roam_time(records_dir, 'large')

    # file = 'logs/' + 'small_1' + '.npz'
    # draw_average_reward(file)
    # roam_time_baseline('small')
    #roam_time('logs/small_large_4.npz', time_bin=4)
    #roam_time('logs/large_small_4.npz', time_bin=4)
    #time_pos('small', 'large', area_bin=10)
    #time_pos('small', 'small')
    #time_pos('large', 'small')
    #time_pos('large', 'large')
    #small_large_contrast()

