import pandas as pd
import numpy as np
import torch
from torch.utils import data


class StockDataSet(data.Dataset):
    # 构造函数
    def __init__(self, x_tensor, y_tensor):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor

    # 数据集大小
    def __len__(self):
        return self.x_tensor.size(0)

    # 生成迭代器
    def __getitem__(self, item):
        return self.x_tensor[item], self.y_tensor[item]


def split_dataset(dataset):
    # X.shape == (-1, seq_length, input_size), Y.shape == (-1, 1)
    y = dataset[1:, -1, 0]  # 获取收盘价y, -1表示stacked中的最后一组数据， 0表示"close"
    return dataset[0:-1], torch.unsqueeze(y, dim=1)


def load_data(batch_size):
    train_X, train_y = split_dataset(train_stacked)
    test_X, test_y = split_dataset(test_stacked)
    train_dataset = StockDataSet(train_X, train_y)
    test_dataset = StockDataSet(test_X, test_y)
    return (data.DataLoader(train_dataset, batch_size, shuffle=False, num_workers=0),
            data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0))


stocks: pd.DataFrame | None = pd.read_csv('Database/PetroChina.csv')
stocks = stocks.loc[:, ["Close", "Open", "High", "Low", "Volume"]]

# 前80%用于训练集，后20%用于测试集
train_size = int(len(stocks) * 0.8)
test_size = len(stocks) - train_size
train = stocks.iloc[0:train_size]
test = stocks.iloc[train_size:len(stocks)].reset_index(drop=True)

# 归一化
train_scaled = (train - train.min(axis=0)) / (train.max(axis=0) - train.min(axis=0))
test_scaled = (test - test.min(axis=0)) / (test.max(axis=0) - test.min(axis=0))

# 转换为tensor
train_scaled = torch.tensor(np.array(train_scaled), dtype=torch.float32)
test_scaled = torch.tensor(np.array(test_scaled), dtype=torch.float32)

# 定义超参数
seq_length = 10

input_size = train_scaled.shape[1]
train_stacked = torch.stack([train_scaled[i:i+seq_length] for i in range(len(train_scaled)-seq_length+1)])
test_stacked = torch.stack([test_scaled[i:i+seq_length] for i in range(len(test_scaled)-seq_length+1)])
