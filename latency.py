import torch
import time

from torch.utils.data import TensorDataset, DataLoader

from src.cluster import cluster_channels
from src.models import MoETransAD, TranAD
from main import load_dataset
import numpy as np

def convert_to_windows(data, model, tp='MoE'):
    windows = [];
    if 'Tran' in tp or tp == 'MoE' or tp == 'MoE2':
        w_size = model.n_window + 1
    else:
        w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w if 'TranAD' in tp or 'Attention' in tp or 'MoETransAD' in tp or tp == 'MoE' or tp == 'MoE2' else w.reshape(-1))
    return torch.stack(windows)


i = 0
bs = 1
wd = 50
feats = [16 , 7,  4 , 4 , 8]
feat = 39
feats = np.array(feats)
model = TranAD(feats=feat, batch=bs, wd=wd)
# model = MoETransAD(feats=feats, batch=bs, wd=wd)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
train_loader, test_loader, labels = load_dataset('SWaT')
trainD, testD = next(iter(train_loader)), next(iter(test_loader))
cl, trainD, testD, label_count = cluster_channels(trainD, testD, 5)
trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
data = torch.DoubleTensor(testD)# 假设 model 和 device 已经定义好
dataset = TensorDataset(data, data)
dataloader = DataLoader(dataset, batch_size=bs)
model.to(device)
model = model.double()

model.eval()  # 设置为评估模式
timings = []
num_runs = 50
for d, _ in dataloader:
    i+=1
    window = d.permute(1, 0, 2)
    bcs = window.shape[1]
    elem = window[-1:, :, :].view(1, bcs, feat)
    rec = window[-2, :, :].view(1, bcs, feat)
    window = window.to(device)
    elem = elem.to(device)
    if i > 20 and i < 70:
        start_time = time.perf_counter()
    z = model(window[:-1, :, :], elem)
    if i== 100:
        torch.cuda.synchronize()
    if i > 20 and i < 70:
        torch.cuda.synchronize() # 等待本次推理完成
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        timings.append(elapsed_time)
    if i ==70:
        break

avg_latency = sum(timings) / num_runs * 1000  # 转换为毫秒
p99_latency = np.percentile(timings, 99) * 1000
print(f"Average latency: {avg_latency:.2f} ms")
print(f"P99 latency: {p99_latency:.2f} ms")

