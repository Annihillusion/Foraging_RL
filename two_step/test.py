import matplotlib.pyplot as plt
import numpy as np

# 示例数据
time_labels = ['基准值', '0.5~1.5', '1.5~2.5', '2.5~3.5', '3.5~4.0']
x = np.arange(len(time_labels))

# 基准值数据
y1 = [3, np.nan, np.nan, np.nan, np.nan]
y1_err = [0.5, np.nan, np.nan, np.nan, np.nan]

# 小菌斑->大菌斑数据
y2 = [np.nan, 15, 10, 7, 5]
y2_err = [np.nan, 2, 1.5, 1, 0.8]

# 小菌斑->小菌斑数据
y3 = [np.nan, 6, 5, 4, 3]
y3_err = [np.nan, 1, 0.8, 0.6, 0.5]

# 创建图表
fig, ax = plt.subplots()

# 绘制数据点和误差条
ax.errorbar(x, y1, yerr=y1_err, fmt='^', color='grey', linestyle=':', label='基准值（小菌斑）')
ax.errorbar(x, y2, yerr=y2_err, fmt='^', color='blue', linestyle='-', label='小菌斑->大菌斑')
ax.errorbar(x, y3, yerr=y3_err, fmt='^', color='lightblue', linestyle='-', label='小菌斑->小菌斑')

# 添加显著性标记
significance_x = [1, 2, 3]  # 显著性标记的位置
significance_y = [15, 10, 5]  # 对应的y值
for i in range(len(significance_x)):
    ax.text(significance_x[i], significance_y[i] + 1, '**', ha='center')

# 设置x轴标签
ax.set_xticks(x)
ax.set_xticklabels(time_labels)

# 添加图例
ax.legend()

# 设置y轴范围
ax.set_ylim(0, 20)

# 设置x轴标签和y轴标签
ax.set_xlabel('时间（h）')
ax.set_ylabel('')

# 显示图表
plt.show()
