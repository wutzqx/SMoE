import os

import numpy as np
import matplotlib.pyplot as plt
lb = 15000
ub = 23000
# 读取多元时间序列数据 (假设文件名为 'data.npy')
dataset = 'SMD'
loss_path = f'{dataset}/TransSMD_loss.npy'
file_path = f'{dataset}/machine-1-1_test.npy'
train_path = f'{dataset}/machine-1-1_train.npy'
label_file_path = f'{dataset}/machine-1-1_labels.npy'
# loss_path = f'{dataset}/SWaT_loss.npy'
# file_path = f'{dataset}/SWaT_test.npy'
# train_path = f'{dataset}/train.npy'
# label_file_path = f'{dataset}/SWaT_labels.npy'
data = np.load(file_path) # shape: [time_steps, num_features]
train = np.load(train_path)
loss_data = np.load(loss_path)
# 读取标签数据 (假设文件名为 'labels.npy')
labels = np.load(label_file_path)  # shape: [time_steps]
data = data[lb:ub, :]
train = train[lb:ub,:]
labels = labels[lb:ub,:]
loss_data = loss_data[lb:ub,:]
# 检查数据形状
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# 创建时间轴
time_steps = data.shape[0]
time = np.arange(time_steps)

# 创建保存图像的文件夹
save_dir = 'plot/smdTransLocal/'
os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，则创建

# 绘制每个特征的时间序列并保存
num_features = data.shape[1]
for i in range(num_features):
    plt.figure(figsize=(10, 1),dpi=200)

    # 绘制时间序列
    plt.plot(time, data[:, i], label=f'Feature {i + 1}')

    # 覆盖对应特征的异常区域
    plt.fill_between(time, y1=0, y2=1,
                     where=labels[:, i] == 1, color='red', alpha=0.3, label='Anomaly')

    # 添加图例和标签
    #plt.legend(loc='upper left')
    plt.legend().set_visible(False)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Time Series for Feature {i + 1} with Anomaly Labels')
    plt.grid(False)
    plt.ylim(0, 1)

    # 保存图像
    save_path = os.path.join(save_dir, f'feature_{i + 1}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存


    plt.figure(figsize=(10, 1),dpi=200)
    # 绘制时间序列
    plt.plot(time, train[:, i], label=f'Feature {i + 1}')
    # 添加图例和标签
    #plt.legend(loc='upper left')
    plt.legend().set_visible(False)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(f'Time Series for Feature {i + 1} with Anomaly Labels')
    plt.grid(False)
    plt.ylim(0, 1)

    # 保存图像
    save_path = os.path.join(save_dir, f'feature_train_{i + 1}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭当前图像，释放内存


    plt.figure(figsize=(10, 1),dpi=200)
    plt.plot(time, loss_data[:, i], label=f'Feature {i + 1}', color='red')

    # 添加图例和标签
    #plt.legend(loc='upper left')
    plt.legend().set_visible(False)
    plt.xlabel('Time Steps')
    plt.grid(False)
    plt.ylim(0, 1)

    # 保存图像
    save_path = os.path.join(save_dir, f'loss_{i + 1}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  #

print(f"所有图像已保存到 {save_dir}")