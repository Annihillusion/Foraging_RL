import os
import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt

from Environment import make_vec_envs, CircularEnv
from params import parse_args
from analysis import get_repeat_number
from a2c_ppo_acktr.utils import get_render_func


def draw_track(pos_trajectory, state_trajectory, save_mode=True, fig_path=None):
    radius = round(np.linalg.norm(pos_trajectory[0]))
    pygame.init()
    # 设置图形间隔
    gap = 50
    # 设置屏幕尺寸
    screen_size = 800
    screen = pygame.display.set_mode((screen_size * 2 + gap, screen_size))
    # 渲染环境
    screen.fill((255, 255, 255))
    # 绘制圆形区域
    pygame.draw.circle(screen, (0, 0, 0), (screen_size / 2, screen_size / 2),
                       screen_size * radius / (2 * radius), 2)
    # pygame.draw.circle(screen, (0, 0, 0), (screen_size / 2, screen_size / 2),
    #                    screen_size * radius / (2 * radius) * 0.75, 1)
    pygame.draw.circle(screen, (0, 0, 0), (screen_size / 2 + screen_size + gap, screen_size / 2),
                       screen_size * radius / (2 * radius), 2)
    # pygame.draw.circle(screen, (0, 0, 0), (screen_size / 2 + screen_size + gap, screen_size / 2),
    #                    screen_size * radius / (2 * radius) * 0.75, 1)

    length = len(pos_trajectory)
    for i in range(length - 1):
        color = np.array(plt.cm.viridis(i / length))
        color = (color * 255).astype(np.int32)
        start_pos = [int(screen_size * (pos_trajectory[i][0] + radius) / (2 * radius)),
                     int(screen_size * (pos_trajectory[i][1] + radius) / (2 * radius))]
        end_pos = [int(screen_size * (pos_trajectory[i + 1][0] + radius) / (2 * radius)),
                   int(screen_size * (pos_trajectory[i + 1][1] + radius) / (2 * radius))]
        pygame.draw.line(screen, color, start_pos, end_pos, 2)
        start_pos[0] += screen_size + gap
        end_pos[0] += screen_size + gap
        pygame.draw.line(screen, (0, 47, 167, 255) if state_trajectory[i] else (242, 5, 5, 255), start_pos, end_pos, 2)
    # 更新显示
    pygame.display.flip()

    if save_mode:
        pygame.image.save(screen, fig_path)
        return

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 检测按键，比如按下ESC键退出
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False


def test_and_plot(actor_critic, args, record_name, episode_id, save_mode=True):
    device = torch.device("cpu")
    env = make_vec_envs(args.env_name, args.seed, 1,
                        args.gamma, args.log_dir, device, True, args.num_episode_steps, args.radius)
    render_func = get_render_func(env)

    obs = env.reset()
    recurrent_hidden_states = torch.randn(1, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.ones(1, 1, device=device)

    for i in range(14400):
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = actor_critic.act(
                obs,
                recurrent_hidden_states,
                masks,
                deterministic=False)
        obs, _, done, infos = env.step(action)
    pos_trajectory, state_trajectory, energy_trajectory = render_func()
    env.close()

    num_window = 4
    roam_rate = np.empty(num_window)
    len_window = len(state_trajectory) // num_window
    for i in range(num_window):
        roam_rate[i] = (len_window - sum(state_trajectory[i * len_window: (i+1) * len_window])) / len_window

    fig, ax = plt.subplots()
    ax.plot(np.arange(num_window), roam_rate, marker='o', linestyle='-')
    if save_mode:
        save_dir = os.path.join('exp_traces', args.exp_name, record_name)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'roamrate' + '_episode' + str(episode_id) + '.png'))
        fig_name = os.path.join(save_dir, 'trajectory' + '_episode' + str(episode_id) + '.png')
        draw_track(pos_trajectory, state_trajectory, args.radius, save_mode=save_mode, fig_path=fig_name)
    else:
        draw_track(pos_trajectory, state_trajectory, args.radius, save_mode=save_mode)
        plt.show()

    return pos_trajectory, state_trajectory, energy_trajectory


def load_data(file_path: str):
    npzfile = np.load(file_path)

    pos_data = npzfile['position']
    state_data = npzfile['action']
    pos_data = pos_data.transpose([1, 0, 2])
    state_data = state_data[1:].transpose([1, 0])
    return pos_data, state_data


def draw_track_all(exp_name: str):
    records_dir = os.path.join('exp_records', exp_name)
    os.makedirs(os.path.join('exp_plots', exp_name, 'tracks'), exist_ok=True)
    repeat_num = get_repeat_number(records_dir)
    prefix = ['small', 'large', 'small_small', 'large_small', 'small_large', 'large_large']
    file_list = []
    for p in prefix:
        # 由于绘图程序缓慢，可选择前若干次实验，每次实验中的一个线程进行可视化
        for i in range(1):
            file_list.append(p + '_' + str(i + 1))
    for file_name in file_list:
        pos_data, state_data = load_data(os.path.join(records_dir, file_name+'.npz'))
        #save_path = os.path.join('exp_plots', exp_name, 'tracks', file_name + '.png')
        #draw_track(pos_data[0], state_data[0], save_mode=True, fig_path=save_path)
        num_process = pos_data.shape[0]
        for i in range(num_process):
            save_name = f'{file_name}({i+1})'
            save_path = os.path.join('exp_plots', exp_name, 'tracks', save_name+'.png')
            draw_track(pos_data[i], state_data[i], save_mode=True, fig_path=save_path)


if __name__ == '__main__':
    args = parse_args()
    record_name = "/Users/annihillusion/Workflow/Foraging_RL/DRL/exp_records/origin/small_small_2.npz"
    # actor_critic, _ = torch.load(os.path.join(args.save_dir, record_name + ".pt"), map_location="cpu")
    # test_and_plot(actor_critic, args, record_name, 0, save_mode=False)
    # pos_data, state_data = load_data(record_name)
    # radius = round(np.linalg.norm(pos_data[0, 0]))
    # draw_trace(pos_data[0], state_data[0], radius, save_mode=False)
    # records_dir = '/Users/annihillusion/Workflow/Foraging_RL/DRL/exp_records/origin'
    draw_track_all('baseline')
