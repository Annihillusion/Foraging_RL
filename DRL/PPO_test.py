import pygame
import sys

# 初始化Pygame
pygame.init()

# 获取显示器的尺寸信息
info = pygame.display.Info()
width, height = info.current_w, info.current_h

# 设置窗口尺寸和位置
win_width = int(width / 2)
win_height = height
win_pos_x_1 = 0
win_pos_y = 0

# 创建第一个窗口
screen1 = pygame.display.set_mode((win_width, win_height), pygame.RESIZABLE)
screen1.fill((255, 0, 0))  # 填充红色
pygame.display.set_caption("Window 1")

# 设置第二个窗口的位置，使其与第一个窗口并列
win_pos_x_2 = win_width
screen2 = pygame.display.set_mode((win_width, win_height), pygame.RESIZABLE)
screen2.fill((0, 255, 0))  # 填充绿色
pygame.display.set_caption("Window 2")

# 设置图标（可选）
try:
    pygame.display.set_icon(pygame.image.load('game_icon.png'))
except:
    pass

# 游戏主循环
running = True
while running:
    for try_again in range(2):
        all_events = pygame.event.get()
        for event in all_events:
            if event.type == pygame.QUIT:
                running = False

    # 绘制内容到第一个窗口
    pygame.display.flip()

    # 绘制内容到第二个窗口
    pygame.display.flip()

    # 控制循环更新速度
    pygame.time.Clock().tick(30)

# 退出游戏
pygame.quit()
sys.exit()