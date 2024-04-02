import numpy as np
import matplotlib.pyplot as plt


def cal_percentage(seq, num_window=4):
    length = seq.shape[0]
    width = length // num_window
    roam_hist = np.zeros(num_window)
    dwell_hist = np.zeros(num_window)
    for i in range(num_window - 1):
        for j in range(width * i, width * (i + 1)):
            if seq[j] == 0:
                roam_hist[i] += 1
            else:
                dwell_hist[i] += 1
    for j in range(width * (num_window - 1), length):
        if seq[j] == 0:
            roam_hist[-1] += 1
        elif seq[j] == 1:
            dwell_hist[-1] += 1
    return roam_hist / (dwell_hist + roam_hist)


def get_single_hist(states, actions, num_window, num_state):
    hist = np.zeros([num_state, num_window])
    for i in range(num_state):
        act_collector = []
        for j, state in enumerate(states):
            if state == i:
                act_collector.append(actions[j])
        hist[i] = cal_percentage(np.array(act_collector), num_window)
    return hist


def draw_roaming_percentage(file_name, num_window=4, num_state=3):
    data = np.load(file_name)
    states = data[0].astype(int)
    actions = data[1].astype(int)
    num_worm = states.shape[0]
    hist = np.zeros([num_worm, num_state, num_window])

    for i in range(num_worm):
        hist[i] = get_single_hist(states[i], actions[i], num_window, num_state)
    mean = np.mean(hist, axis=0)
    # std = np.std(hist, axis=0)

    plt.figure(figsize=(8, 6))
    bar_width = 0.2
    x = np.arange(num_window)
    time_labels = [f'Period {i + 1}' for i in range(num_window)]
    location_labels = ['Edge', 'Middle', 'Center']
    for i in range(num_state):
        x_0 = x - bar_width * num_state / 2
        x_plot = x_0 + i * bar_width + bar_width / 2
        plt.bar(x_plot, mean[i] * 100, width=bar_width, label=location_labels[i], color=plt.cm.viridis(i/num_state))

    plt.xlabel('Time')
    plt.ylabel('Proportion %')
    plt.xticks(x, time_labels)
    plt.legend()
    plt.show()


def draw_location_distribution(file_name, num_state=3):
    data = np.load(file_name)
    states = data[0].astype(int)
    num_Celegans, time_steps = states.shape
    loc_distrib = np.zeros([time_steps, num_state])
    for i in range(time_steps):
        for item in states[:, i]:
            loc_distrib[i, item] += 1
    loc_distrib = loc_distrib / num_Celegans * 100

    plt.figure(figsize=(8, 6))
    plt.stackplot(range(time_steps), np.transpose(loc_distrib), labels=['Edge', 'Middle', 'Center'])
    plt.title('Untitled')
    plt.xlabel('Steps')
    plt.ylabel('Percentage')
    plt.legend(loc='upper left')  # 添加图例
    plt.show()


if __name__ == '__main__':
    draw_location_distribution('records/train_rec.npy')
    draw_roaming_percentage('records/train_rec.npy')
