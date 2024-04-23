import os
import sys

import numpy as np
import torch
import pygame
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor


class CircularEnv(gym.Env):
    def __init__(self, radius=5.0, step_size=0.1, render_mode=None, screen_size=1000, seed=202434, num_episode_steps=0):
        super(CircularEnv, self).__init__()

        self.radius = radius
        self.step_size = step_size
        self.render_mode = render_mode
        self.np_ramdom = np.random.default_rng(seed=seed)
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        # 方向角
        self.head_angle = 0.0
        self.clockwise = 1
        # 定义动作空间，turn & move
        self.action_space = spaces.Discrete(2)
        self.state = 0
        # 定义观察空间
        self.observation_space = spaces.Box(0, 1, shape=[2], dtype=np.float32)
        # 记录运动轨迹
        self.trajectory = []
        self.pygame_init = False
        self.screen_size = screen_size
        self.screen = None
        # 用于绘制轨迹时调制颜色
        self.num_episode_steps = num_episode_steps
        self.cur_step = 0

    @property
    def observation(self):
        distance_from_origin = np.linalg.norm(self.agent_position)
        # return np.array(distance_from_origin / self.radius * 9)
        return np.zeros(2)

    def reset(self, seed=None, *args):
        # 将Agent放在圆形区域中心
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        self.head_angle = self.np_random.uniform(0, 2 * np.pi)
        self.trajectory.append(self.agent_position.copy())
        # 重置轨迹
        self.trajectory = []
        info = {}
        return self.observation, info

    def step(self, action):
        # if action == 0:
        #     yaw_angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        #     self.head_angle += yaw_angle
        #
        # self.agent_position[0] += self.step_size * np.cos(self.head_angle)
        # self.agent_position[1] += self.step_size * np.sin(self.head_angle)
        self.move(action)

        distance_from_origin = np.linalg.norm(self.agent_position)
        # 定义奖励
        if distance_from_origin > self.radius:
            reward = -1
        else:
            reward = distance_from_origin / self.radius
            # reward = 1
        # 限制Agent在圆形区域内移动
        if distance_from_origin > self.radius:
            self.agent_position /= distance_from_origin  # 将位置归一化到圆形边界
            self.agent_position *= self.radius
        # 更新轨迹
        self.trajectory.append(self.agent_position.copy())

        # 定义是否终止的条件
        terminated, truncated, info = False, False, {}

        return self.observation, reward, terminated, truncated, info

    def move(self, action):
        # roaming
        if action == 0:
            # 不是连续的roaming，重新选择偏转方向
            # if self.state == 1:
            #     self.state = 0
            self.clockwise = np.random.choice([-1, 1], 1)
            turn_angle = np.random.normal(30/180*np.pi, 1)
            self.head_angle += turn_angle * self.clockwise
            self.agent_position[0] += self.step_size * np.cos(self.head_angle)
            self.agent_position[1] += self.step_size * np.sin(self.head_angle)
        # dwelling
        elif action == 1:
            self.state = 1
        else:
            raise NotImplementedError

    def render(self):
        # if self.render_mode != 'human':
        #     return
        if not self.pygame_init:
            # 初始化 Pygame
            pygame.init()
            # 设置屏幕尺寸
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.pygame_init = True
        # 渲染环境
        self.screen.fill((255, 255, 255))

        # 将Agent的位置映射到屏幕坐标
        screen_x = int(self.screen_size * (self.agent_position[0] + self.radius) / (2 * self.radius))
        screen_y = int(self.screen_size * (self.agent_position[1] + self.radius) / (2 * self.radius))

        # 绘制圆形区域
        pygame.draw.circle(self.screen, (0, 0, 0), (int(self.screen_size / 2), int(self.screen_size / 2)), int(self.screen_size * self.radius / (2 * self.radius)), 1)
        # 绘制Agent
        pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), 5)
        # 绘制轨迹线
        # if len(self.trajectory) > 1:
        #     # color: int[r, g, b, alpha], 0~255
        #     color = np.array(plt.cm.viridis(self.cur_step / self.num_episode_steps))
        #     color = (color * 255).astype(np.int32)
        #     self.cur_step += 1
        #     pygame.draw.lines(self.screen, color, False, [(int(self.screen_size * (x + self.radius) / (
        #                 2 * self.radius)), int(self.screen_size * (y + self.radius) / (2 * self.radius))) for x, y in
        #                                                       self.trajectory], 2)
        length = len(self.trajectory)
        for i in range(length - 1):
            color = np.array(plt.cm.viridis(i / length))
            r = i / length
            # color = np.array([0, 0, 0.8, 1])*(1-r) + np.array([0, 0, 0, 1])*r
            color = (color * 255).astype(np.int32)
            start_pos = [int(self.screen_size * (self.trajectory[i][0] + self.radius) / (
                        2 * self.radius)), int(self.screen_size * (self.trajectory[i][1] + self.radius) / (2 * self.radius))]
            end_pos = [int(self.screen_size * (self.trajectory[i+1][0] + self.radius) / (
                        2 * self.radius)), int(self.screen_size * (self.trajectory[i+1][1] + self.radius) / (2 * self.radius))]
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
        # 更新显示
        pygame.display.flip()

        # 处理手动退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
        sys.exit()


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


def make_env(env_id, seed, rank, log_dir, allow_early_resets, num_episode_steps):
    def _thunk():
        if env_id == 'CircularEnv':
            env = CircularEnv(seed=seed+rank, num_episode_steps=num_episode_steps)
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
                  num_episode_steps):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, num_episode_steps)
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


if __name__ == '__main__':
    # 创建圆形环境
    radius = 5.0
    env = CircularEnv(radius, step_size=0.1, render_mode='human')

    for _ in range(2000):
        action = env.action_space.sample()  # 随机选择一个动作
        observation, reward, done, _, _ = env.step(action)
        env.render()

    env.close()
