import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
data1 = np.random.rand(10, 10)
data2 = np.random.rand(10, 10)

# 创建颜色映射对象
cmap = plt.get_cmap('viridis')

# 创建一个图形和两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 绘制第一幅子图
cax1 = ax1.imshow(data1, cmap=cmap)
ax1.set_title('Subplot 1')

# 绘制第二幅子图
cax2 = ax2.imshow(data2, cmap=cmap)
ax2.set_title('Subplot 2')

# 在第一幅子图上添加颜色条
fig.colorbar(cax1, ax=[ax1, ax2])

# 显示图形
plt.show()