import matplotlib.pyplot as plt
import numpy as np

# 创建示例数据
labels = ['Group 1', 'Group 2', 'Group 3']
data1 = [10, 15, 20]
data2 = [12, 17, 15]

# 设置每组柱状图的宽度
bar_width = 0.35

# 计算每组柱状图的 x 坐标
x = np.arange(len(labels))

# 绘制柱状图
plt.bar(x - bar_width/2, data1, width=bar_width, label='Data 1')
plt.bar(x + bar_width/2, data2, width=bar_width, label='Data 2')

# 添加标签和图例
plt.xlabel('Groups')
plt.ylabel('Values')
plt.title('Grouped Bar Chart')
plt.xticks(x, labels)
plt.legend()

# 显示图形
plt.show()
