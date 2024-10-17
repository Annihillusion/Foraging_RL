import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch

from analysis import get_repeat_number


def activation_baseline(prefix: str, records_dir: str):
    repeat_num = get_repeat_number(records_dir)
    file_list = [prefix + '_' + str(i + 1) + '.npz' for i in range(repeat_num)]
    rnn_x_roam = []
    rnn_x_dwell = []
    actor_feature_roam = []
    actor_feature_dwell = []

    for file in file_list:
        file_path = os.path.join(records_dir, file)
        npzfile = np.load(file_path)
        # 取相对稳定的最后1/5部分
        n_steps = int(npzfile['action'].shape[0] * 0.8)
        action = npzfile['action'][n_steps:].transpose([1, 0])
        rnn_x = npzfile['rnn_x'][n_steps:].transpose([1, 0, 2])
        actor_feature = npzfile['actor_feature'][n_steps:].reshape([4, -1, 4])

        for i in range(action.shape[0]):
            indices_roam = np.argwhere(action[i] == 0).squeeze(-1)
            indices_dwell = np.argwhere(action[i] == 1).squeeze(-1)
            rnn_x_roam_tmp = rnn_x[i][indices_roam].mean(axis=0)
            rnn_x_dwell_tmp = rnn_x[i][indices_dwell].mean(axis=0)
            actor_feature_roam_tmp = actor_feature[i][indices_roam].mean(axis=0)
            actor_feature_dwell_tmp = actor_feature[i][indices_dwell].mean(axis=0)
            rnn_x_roam.append(rnn_x_roam_tmp)
            rnn_x_dwell.append(rnn_x_dwell_tmp)
            actor_feature_roam.append(actor_feature_roam_tmp)
            actor_feature_dwell.append(actor_feature_dwell_tmp)

    rnn_x_roam = np.array(rnn_x_roam)
    rnn_x_dwell = np.array(rnn_x_dwell)
    actor_feature_roam = np.array(actor_feature_roam)
    actor_feature_dwell = np.array(actor_feature_dwell)

    activation_roam = np.hstack([rnn_x_roam.mean(axis=0), actor_feature_roam.mean(axis=0)])
    activation_dwell = np.hstack([rnn_x_dwell.mean(axis=0), actor_feature_dwell.mean(axis=0)])

    return activation_roam, activation_dwell


def activation_transfer(prefix: str, records_dir: str, time_bin=4):
    repeat_num = get_repeat_number(records_dir)
    file_list = [prefix + '_' + str(i + 1) + '.npz' for i in range(repeat_num)]
    rnn_x_roam_all = []
    rnn_x_dwell_all = []
    actor_feature_roam_all = []
    actor_feature_dwell_all = []

    for file in file_list:
        file_path = os.path.join(records_dir, file)
        npzfile = np.load(file_path)
        action = npzfile['action'].transpose([1, 0])
        rnn_x = npzfile['rnn_x'].transpose([1, 0, 2])
        actor_feature = npzfile['actor_feature'].reshape([4, -1, 4])

        a, b = action.shape
        action = action.reshape([a, time_bin, b // time_bin])
        a, b, c = rnn_x.shape
        rnn_x = rnn_x.reshape([a, time_bin, b // time_bin, c])
        actor_feature = actor_feature.reshape([a, time_bin, b // time_bin, c])

        rnn_x_roam = []
        rnn_x_dwell = []
        actor_feature_roam = []
        actor_feature_dwell = []
        # 依次处理每个trial中的4个并行线程的数据
        for i in range(action.shape[0]):
            rnn_x_roam_time = []
            rnn_x_dwell_time = []
            actor_feature_roam_time = []
            actor_feature_dwell_time = []
            # 依次处理每个时间段的数据
            for j in range(time_bin):
                indices_roam = np.argwhere(action[i, j] == 0).squeeze(-1)
                indices_dwell = np.argwhere(action[i, j] == 1).squeeze(-1)
                rnn_x_roam_tmp = rnn_x[i, j][indices_roam].mean(axis=0)
                rnn_x_dwell_tmp = rnn_x[i, j][indices_dwell].mean(axis=0)
                actor_feature_roam_tmp = actor_feature[i, j][indices_roam].mean(axis=0)
                actor_feature_dwell_tmp = actor_feature[i, j][indices_dwell].mean(axis=0)
                rnn_x_roam_time.append(rnn_x_roam_tmp)
                rnn_x_dwell_time.append(rnn_x_dwell_tmp)
                actor_feature_roam_time.append(actor_feature_roam_tmp)
                actor_feature_dwell_time.append(actor_feature_dwell_tmp)
            rnn_x_roam.append(rnn_x_roam_time)
            rnn_x_dwell.append(rnn_x_dwell_time)
            actor_feature_roam.append(actor_feature_roam_time)
            actor_feature_dwell.append(actor_feature_dwell_time)
        rnn_x_roam_all.append(np.array(rnn_x_roam).mean(axis=0))
        rnn_x_dwell_all.append(np.array(rnn_x_dwell).mean(axis=0))
        actor_feature_roam_all.append(np.array(actor_feature_roam).mean(axis=0))
        actor_feature_dwell_all.append(np.array(actor_feature_dwell).mean(axis=0))

    rr = np.array(rnn_x_roam_all).mean(axis=0)
    rd = np.array(rnn_x_dwell_all).mean(axis=0)
    ar = np.array(actor_feature_roam_all).mean(axis=0)
    ad = np.array(actor_feature_dwell_all).mean(axis=0)
    activation_roam = np.hstack([rr, ar])
    activation_dwell = np.hstack([rd, ad])
    return activation_roam, activation_dwell


def draw_activation_map(exp_name: str, baseline: str, time_bin=4):
    records_dir = os.path.join('exp_records', exp_name)
    baseline_roam, baseline_dwell = activation_baseline(baseline, records_dir)
    transfer_roam_small, transfer_dwell_small = activation_transfer(baseline + '_' + 'small', records_dir)
    transfer_roam_large, transfer_dwell_large = activation_transfer(baseline + '_' + 'large', records_dir)

    blank_row = np.zeros(baseline_roam.shape[0])
    map_data_small = np.vstack([baseline_roam, baseline_dwell, blank_row,
                                transfer_roam_small[0], transfer_dwell_small[0], blank_row,
                                transfer_roam_small[1], transfer_dwell_small[1], blank_row,
                                transfer_roam_small[2], transfer_dwell_small[2], blank_row,
                                transfer_roam_small[3], transfer_dwell_small[3]
                                ]).transpose([1, 0])
    map_data_large = np.vstack([baseline_roam, baseline_dwell, blank_row,
                                transfer_roam_large[0], transfer_dwell_large[0], blank_row,
                                transfer_roam_large[1], transfer_dwell_large[1], blank_row,
                                transfer_roam_large[2], transfer_dwell_large[2], blank_row,
                                transfer_roam_large[3], transfer_dwell_large[3]
                                ]).transpose([1, 0])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    x = np.arange(0, 15, 1)
    y = np.arange(0, 9, 1)

    vmin = min(map_data_small.min(), map_data_small.min())
    vmax = max(map_data_large.max(), map_data_large.max())
    abs_max = max(abs(vmin), abs(vmax))
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap('RdBu')
    ax1.pcolormesh(x, y, map_data_small, cmap=cmap, norm=norm)
    im = ax2.pcolormesh(x, y, map_data_large, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=[ax1, ax2])
    ax1.set_title(f'{baseline} to small')
    ax2.set_title(f'{baseline} to large')
    plt.show()


if __name__ == '__main__':
    exp_name = 'baseline'
    draw_activation_map(exp_name, 'small')
    # draw_activation_map(exp_name, 'small', 'large')
    # draw_activation_map(exp_name, 'large', 'large')
    # draw_activation_map(exp_name, 'large', 'small')
