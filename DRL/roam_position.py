import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from analysis import get_repeat_number, judge_area


def roam_time_pos_transfer(prefix: str, records_dir: str, repeat_num: int, radius: int, time_bin: int, area_bin: int):
    file_list = [prefix + '_' + str(i + 1) + '.npz' for i in range(repeat_num)]
    collector = []
    for file in file_list:
        file_path = os.path.join(records_dir, file)
        npzfile = np.load(file_path)
        pos, action = npzfile['position'][1:], npzfile['action']
        num_steps = action.shape[0]
        num_agents = action.shape[1]
        action = action.transpose([1, 0])
        pos = pos.transpose([1, 0, 2])

        mean = np.empty([time_bin, area_bin])
        err = np.empty([time_bin, area_bin])
        len_time_bin = num_steps // time_bin

        # 若干并行线程的平均值作为该次实验的数据，后续有待改进
        for i in range(time_bin):
            shift = i * len_time_bin
            roam_rate = np.empty([area_bin, num_agents])
            for j in range(num_agents):
                roam_cnt, total_cnt = np.zeros(area_bin), np.zeros(area_bin)
                for k in range(len_time_bin):
                    area = judge_area(pos[j, shift + k], radius, area_bin)
                    roam_cnt[area] += 1 - action[j, shift + k]
                    total_cnt[area] += 1
                tmp = roam_cnt / total_cnt
                roam_rate[:, j] = tmp
            time_mean = np.mean(roam_rate, axis=1)
            time_err = np.std(roam_rate, axis=1) / np.sqrt(num_agents)
            mean[i, :] = time_mean
            err[i, :] = time_err
        mean, err = np.nan_to_num(mean), np.nan_to_num(err)
        mean, err = mean.transpose([1, 0]), err.transpose([1, 0])
        collector.append(mean)

    roam_data = np.array(collector)
    mean = np.mean(roam_data, axis=0)

    return mean * 100, None


def roam_time_pos(records_dir: str, baseline: str, transfer: str, time_bin=3, area_bin=3):
    repeat_num = get_repeat_number(records_dir)

    if baseline == 'small':
        pass
    elif baseline == 'large':
        pass
    else:
        raise ValueError('Choose from small and large')
    mean_baseline = None
    err_baseline = None

    patch_size = 12 if transfer == 'large' else 6
    mean, err = roam_time_pos_transfer(baseline+'_'+transfer, records_dir, repeat_num, patch_size, time_bin, area_bin)

    fig, ax = plt.subplots(dpi=150)
    # fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.2
    x = np.arange(time_bin)
    time_labels = [f'Period {i + 1}' for i in range(time_bin)]
    #location_labels = ['Edge', 'Middle', 'Center']
    for i in range(area_bin):
        x_0 = x - bar_width * area_bin / 2
        x_plot = x_0 + i * bar_width + bar_width / 2
        plt.bar(x_plot, mean[i], width=bar_width)
        # plt.bar(x_plot, mean[i], width=bar_width, color=plt.cm.viridis(i / area_bin))
        # plt.bar(x_plot, mean[i], width=bar_width, color=plt.cm.viridis(i / area_bin))
        # plt.errorbar(x_plot, mean[i], yerr=err[i], color='b')

    ax.set_ylim([0, 30])
    # plt.xlabel('Time')
    plt.ylabel('Prob. of roaming')
    plt.xticks(x, time_labels)
    plt.title(baseline+'_'+transfer)
    # plt.legend()
    _, exp_name = os.path.split(records_dir)
    plt.savefig(os.path.join('exp_plots', exp_name, 'roam_time_pos_' + baseline + '_' + transfer))
    plt.show()


if __name__ == '__main__':
    experiment_name = 'dwell_stay'
    records_dir = os.path.join('exp_records', experiment_name)
    roam_time_pos(records_dir, 'small', 'small', time_bin=4, area_bin=4)
    # roam_time_pos(records_dir, 'small', 'large', time_bin=4, area_bin=10)
    # roam_time_pos(records_dir, 'large', 'small')
    # roam_time_pos(records_dir, 'large', 'large')
