import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys


class CircularEnv(gym.Env):
    def __init__(self, radius=1, step_size=0.1, render_mode='human', screen_size=1000):
        super(CircularEnv, self).__init__()

        self.radius = radius
        self.step_size = step_size
        self.render_mode = render_mode
        # [x, y]表示位置
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        # 方向角
        self.head_angle = np.pi
        # 定义动作空间，turn & move
        self.action_space = spaces.Discrete(2)
        # 定义状态空间，roam state & dwell state
        self.state_space = spaces.Discrete(10)
        # 记录运动轨迹
        self.trajectory = []
        # 初始化 Pygame
        pygame.init()
        # 设置屏幕尺寸
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

    def reset(self):
        # 将Agent放在圆形区域中心
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)
        self.head_angle = np.random.uniform(0, 2 * np.pi)
        self.trajectory.append(self.agent_position.copy())
        # 重置轨迹
        self.trajectory = []
        return 0

    def step(self, action):
        if action == 0:
            yaw_angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            self.head_angle += yaw_angle

        self.agent_position[0] += self.step_size * np.cos(self.head_angle)
        self.agent_position[1] += self.step_size * np.sin(self.head_angle)

        # 限制Agent在圆形区域内移动
        distance_from_origin = np.linalg.norm(self.agent_position)
        if distance_from_origin > self.radius:
            self.agent_position /= distance_from_origin  # 将位置归一化到圆形边界
            self.agent_position *= self.radius
        # 更新轨迹
        self.trajectory.append(self.agent_position.copy())
        # 定义奖励，中心0.5，边界1，线性变化；界外-1
        if distance_from_origin > self.radius:
            reward = -1
        else:
            reward = 0.5 / self.radius * distance_from_origin + 0.5
        # 定义是否终止的条件
        done = False

        return int(distance_from_origin/self.radius*9), reward, done, {}

    def render(self, mode='human'):
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
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, (0, 0, 0), False, [(int(self.screen_size * (x + self.radius) / (
                        2 * self.radius)), int(self.screen_size * (y + self.radius) / (2 * self.radius))) for x, y in
                                                              self.trajectory], 2)
        # 更新显示
        pygame.display.flip()

        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    # 创建圆形环境
    radius = 5.0
    env = CircularEnv(radius, step_size=0.1)

    # 测试环境
    for _ in range(2000):
        action = env.action_space.sample()  # 随机选择一个动作
        observation, reward, done, _ = env.step(action)
        env.render()
        # print(f"Position: {observation}, Reward: {reward}, Done: {done}")

    pygame.quit()
    # 等待关闭窗口事件
    pygame.event.wait()
