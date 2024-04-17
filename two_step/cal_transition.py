import numpy as np


def cal_transition(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    # 找到包含 NaN 的行
    nan_rows = np.isnan(data).any(axis=1)
    # 去除含有 NaN 的行
    data = data[~nan_rows]
    cur_id = data[0, 0]
    # 0: dwell, 1: roam
    cur_state = data[0, 4]
    # 0: edge, 1: center
    cur_loc = 0 if data[0, 5] > 6 else 1
    center_roam, center_dwell, edge_roam, edge_dwell = [], [], [], []
    for line in data:
        if line[0] != cur_id:
            cur_id = line[0]
            cur_state = line[4]
            continue
        if line[4] == cur_state:
            continue
        else:
            next_loc = 0 if line[5] > 6 else 1
            next_state = line[4]
            if cur_loc == 1:
                if cur_state == 1:
                    center_roam.append(next_loc)
                else:
                    center_dwell.append(next_loc)
            else:
                if cur_state == 1:
                    edge_roam.append(next_loc)
                else:
                    edge_dwell.append(next_loc)
            cur_loc = next_loc
            cur_state = next_state
    # [prob_to_edge, prob_to_center]
    prob_ = lambda x: [1 - (sum(x) / len(x)), sum(x) / len(x)]
    center_roam = prob_(center_roam)
    center_dwell = prob_(center_dwell)
    edge_roam = prob_(edge_roam)
    edge_dwell = prob_(edge_dwell)
    roam_mat = np.array([edge_roam, center_roam])
    dwell_mat = np.array([edge_dwell, center_dwell])
    return roam_mat, dwell_mat


if __name__ == '__main__':
    file_path = '/Users/annihillusion/Workflow/Foraging_RL/data/1108_med1x_bas_01_luyi.csv'
    roam_mat, dwell_mat = cal_transition(file_path)
    print(roam_mat)
    print(dwell_mat)
