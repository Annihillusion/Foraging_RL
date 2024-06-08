import os
import sys

import numpy as np
import torch
import pygame
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces
from scipy.stats import multivariate_normal

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor


class CircularEnv(gym.Env):
    def __init__(self, render_mode=None, screen_size=1000, seed=202434, num_episode_steps=0, radius=6):
        super(CircularEnv, self).__init__()

        self.radius = radius
        self.render_mode = render_mode
        self.np_ramdom = np.random.default_rng(seed=seed)
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        # 方向角
        self.head_angle = None
        self.clockwise = None
        # 定义动作空间，turn & move
        self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.Box(np.array([0.0, -np.pi/2]), np.array([0.1, np.pi/2]))
        # 定义观察空间
        self.observation_space = spaces.Box(np.array([0, -100, 0]), np.array([4, 100, 1]))
        # 定义奖励函数
        self.concentration_func = np.poly1d([4.642225e+05,
                                             -2.852648e+06,
                                             7.726167e+06,
                                             -1.212951e+07,
                                             1.221637e+07,
                                             -8.258561e+06,
                                             3.818097e+06,
                                             -1.213596e+06,
                                             2.665742e+05,
                                             -4.167904e+04,
                                             5.021120e+03,
                                             -4.865155e+02,
                                             2.994606e+01,
                                             1.504611e-01,
                                             4.419983e-01
                                             ])
        self.energy = 0
        # 步长与偏转角的分布
        self.dist_roam = multivariate_normal([16.80526863, 0.11663735], [[3.14002146e+01, -1.08002093e-01], [-1.08002093e-01, 1.18060633e-03]])
        self.dist_dwell = multivariate_normal([1.79725427e+02, 4.28807042e-02], [[1.49774232e+03, -8.79897963e-02], [-8.79897963e-02, 2.34201939e-04]])
        # 记录运动轨迹与状态
        self.pos_trajectory = []
        self.state_trajectory = []
        self.energy_trajectory = []

        self.pygame_init = False
        self.screen_size = screen_size
        self.screen, self.screen2 = None, None
        # 用于绘制轨迹时调制颜色
        self.num_episode_steps = num_episode_steps
        self.cur_step = 0

    @property
    def observation(self):
        # Normalized to [0, 1]
        r = np.linalg.norm(self.agent_position) / self.radius
        concentration = self.concentration_func(r)
        df = np.polyder(self.concentration_func)
        theta = np.arctan2(self.agent_position[1], self.agent_position[0])
        alpha = np.abs(theta - self.head_angle)
        gradient = df(r) * np.cos(alpha)
        return np.array([concentration, gradient])

    def reset(self, seed=None, *args):
        # 将Agent放在菌斑边缘
        self.head_angle = self.np_random.uniform(0, 2 * np.pi)
        x = -np.cos(self.head_angle) * self.radius
        y = -np.sin(self.head_angle) * self.radius
        self.agent_position = np.array([x, y], dtype=np.float32)
        # 重置轨迹
        self.pos_trajectory = []
        self.state_trajectory = []
        self.energy_trajectory = []
        self.pos_trajectory.append(self.agent_position.copy())
        self.energy = 0
        info = {}
        return np.append(self.observation, 0), info

    def step(self, action):
        self.move(action)

        distance_from_center = np.linalg.norm(self.agent_position)
        # 定义奖励
        consumption = 1.1 if action == 0 else 0.0
        if distance_from_center > self.radius:
            food = 0
        else:
            food = self.concentration_func(distance_from_center / self.radius)
        self.energy = self.energy * 0.9 + (food - consumption) * 0.1
        reward = food + 2 * self.energy
        # 限制Agent在圆形区域内移动
        if distance_from_center > self.radius:
            self.agent_position /= distance_from_center  # 将位置归一化到圆形边界
            self.agent_position *= self.radius

            theta = np.arctan2(self.agent_position[1], self.agent_position[0])
            beta = np.abs(theta - self.head_angle)
            if self.head_angle > theta:
                self.head_angle += np.pi - 2 * beta
            else:
                self.head_angle -= np.pi - 2 * beta
            self.head_angle %= 2 * np.pi
        # 更新轨迹
        self.pos_trajectory.append(self.agent_position.copy())
        self.state_trajectory.append(action)
        self.energy_trajectory.append(self.energy)
        # 定义是否终止的条件
        terminated, truncated, info = False, False, {'position': self.agent_position, 'energy': self.energy}

        return np.append(self.observation, action), reward, terminated, truncated, info

    def move(self, action):
        # roaming
        if action == 0:
            turn_angle, step_size = self.dist_roam.rvs(size=1)
        # dwelling
        elif action == 1:
            turn_angle, step_size = self.dist_dwell.rvs(size=1)
            # step_size = 0
        else:
            raise NotImplementedError

        turn_angle = turn_angle / 180 * np.pi
        self.clockwise = np.random.choice([-1, 1], 1)[0]
        self.head_angle += turn_angle * self.clockwise
        self.head_angle %= 2 * np.pi
        self.agent_position[0] += step_size * np.cos(self.head_angle)
        self.agent_position[1] += step_size * np.sin(self.head_angle)

    def render(self):
        return self.pos_trajectory, self.state_trajectory, self.energy_trajectory
        # # if self.render_mode != 'human':
        # #     return
        # if not self.pygame_init:
        #     # 初始化 Pygame
        #     pygame.init()
        #     # 设置图形间隔
        #     gap = 50
        #     # 设置屏幕尺寸
        #     self.screen = pygame.display.set_mode((self.screen_size * 2 + gap, self.screen_size))
        #     self.pygame_init = True
        # # 渲染环境
        # self.screen.fill((255, 255, 255))
        #
        # # 将Agent的位置映射到屏幕坐标
        # # screen_x = int(self.screen_size * (self.agent_position[0] + self.radius) / (2 * self.radius))
        # # screen_y = int(self.screen_size * (self.agent_position[1] + self.radius) / (2 * self.radius))
        #
        # # 绘制圆形区域
        # pygame.draw.circle(self.screen, (0, 0, 0), (int(self.screen_size / 2), int(self.screen_size / 2)),
        #                    int(self.screen_size * self.radius / (2 * self.radius)), 1)
        # pygame.draw.circle(self.screen, (0, 0, 0), (int(self.screen_size / 2 + self.screen_size + gap), int(self.screen_size / 2)),
        #                    int(self.screen_size * self.radius / (2 * self.radius)), 1)
        # # 绘制Agent
        # # pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), 5)
        # # pygame.draw.circle(self.screen, (255, 0, 0), (screen_x + self.screen_size + gap, screen_y), 5)
        #
        # # 绘制轨迹线
        # # if len(self.trajectory) > 1:
        # #     # color: int[r, g, b, alpha], 0~255
        # #     color = np.array(plt.cm.viridis(self.cur_step / self.num_episode_steps))
        # #     color = (color * 255).astype(np.int32)
        # #     self.cur_step += 1
        # #     pygame.draw.lines(self.screen, color, False, [(int(self.screen_size * (x + self.radius) / (
        # #                 2 * self.radius)), int(self.screen_size * (y + self.radius) / (2 * self.radius))) for x, y in
        # #                                                       self.trajectory], 2)
        # length = len(self.pos_trajectory)
        # for i in range(length - 1):
        #     color = np.array(plt.cm.viridis(i / length))
        #     # r = i / length
        #     # color = np.array([0, 0, 0.8, 1])*(1-r) + np.array([0, 0, 0, 1])*r
        #     color = (color * 255).astype(np.int32)
        #     start_pos = [int(self.screen_size * (self.pos_trajectory[i][0] + self.radius) / (2 * self.radius)),
        #                  int(self.screen_size * (self.pos_trajectory[i][1] + self.radius) / (2 * self.radius))]
        #     end_pos = [int(self.screen_size * (self.pos_trajectory[i + 1][0] + self.radius) / (2 * self.radius)),
        #                int(self.screen_size * (self.pos_trajectory[i + 1][1] + self.radius) / (2 * self.radius))]
        #     pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
        #     start_pos[0] += self.screen_size + gap
        #     end_pos[0] += self.screen_size + gap
        #     pygame.draw.line(self.screen, (0, 47, 167, 255) if self.state_trajectory[i] else (242, 5, 5, 255), start_pos, end_pos, 2)
        #
        # # 更新显示
        # pygame.display.flip()
        #
        # running = True
        # while running:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
        #         # 检测按键，比如按下ESC键退出
        #         elif event.type == pygame.KEYDOWN:
        #             if event.key == pygame.K_ESCAPE:
        #                 running = False
        # return self.state_trajectory

    def close(self):
        if self.pygame_init:
            pygame.quit()


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


def make_env(env_id, seed, rank, log_dir, allow_early_resets, num_episode_steps, radius):
    def _thunk():
        if env_id == 'CircularEnv':
            env = CircularEnv(seed=seed + rank, num_episode_steps=num_episode_steps, radius=radius)
        else:
            raise NotImplementedError

        if log_dir is not None:
            env = Monitor(env,
                          os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_episode_steps,
                  radius):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, num_episode_steps, radius)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, norm_reward=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)
    return envs
