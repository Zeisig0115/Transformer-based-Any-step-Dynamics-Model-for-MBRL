import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 从 JSON 文件读取
with open(r"C:\PyCharm\test\ADMPO\result\mujoco\Hopper-v5\admpo\Failed Att lr1e-3\record\GRU+Att.txt", 'r') as f:
    data1 = json.load(f)

with open(r"C:\PyCharm\test\ADMPO\result\mujoco\Hopper-v5\admpo\May 2\record\TADM.txt", 'r') as f:
    data2 = json.load(f)

with open(r"C:\PyCharm\test\ADMPO\result\mujoco\Hopper-v5\admpo\Fail TE lr3e-4\record\Original ADM.txt", 'r') as f:
    data3 = json.load(f)

# 2. 转成 numpy 数组并指定 float 类型
steps1 = np.array(data1['step'], dtype=float)
steps2 = np.array(data2['step'], dtype=float)
steps3 = np.array(data1['step'], dtype=float)
means1 = np.array(data1['loss']['model'], dtype=float)
means2 = np.array(data2['loss']['model'], dtype=float)
means3 = np.array(data3['loss']['model'], dtype=float)

# means1 = np.array(data1['reward_mean'], dtype=float)
# means2 = np.array(data2['reward_mean'], dtype=float)
# means3 = np.array(data3['reward_mean'], dtype=float)



from scipy.signal import savgol_filter

means_smooth1 = savgol_filter(means1, window_length=11, polyorder=2)
means_smooth2 = savgol_filter(means2, window_length=11, polyorder=2)
means_smooth3 = savgol_filter(means3, window_length=11, polyorder=2)



# 4. 绘图
plt.figure(figsize=(8,5))
plt.plot(steps1, means1, label='GRU+Att', alpha=0.3)  # 原始曲线，半透明
plt.plot(steps2, means2, label='TADM', alpha=0.3)  # 原始曲线，半透明
plt.plot(steps1, means3, label='Original ADM', alpha=0.3)  # 原始曲线，半透明
plt.plot(steps1, means_smooth1, label='GRU+Att Average Error', linewidth=2)
plt.plot(steps2, means_smooth2, label='TADM Average Error', linewidth=2)
plt.plot(steps3, means_smooth3, label='Original ADM Average Error', linewidth=2)
plt.xlabel('Steps', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.title('Error vs Steps', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
