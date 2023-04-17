import pandas as pd
import numpy as np
import torch
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler


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


# 在堆叠的数据中将X和y区分开
def split_dataset(dataset, binary=False):
    # X.shape == (-1, seq_length, input_size), Y.shape == (-1, output_size)
    if binary:
        y = dataset[1:, -1, -1].unsqueeze(-1)
    else:
        y = dataset[1:, -1, [0, 1]]  # 获取收盘价y, -1表示stacked中的最后一组数据， 0--"close"/1--'open'
    return dataset[0:-1], y


def load_data(batch_size, seq_length, select_stock='PetroChina', binary=False):
    stocks: pd.DataFrame | None = pd.read_csv('Database/'+select_stock+'.csv')
    stocks = stocks.loc[:, ["Close", "Open", "High", "Low", "Volume",
                            "D/E", "Profit Margin", "EPS(diluted)", "P/B ratio"]]

    # 添加盈利标签
    stocks['Label'] = ((stocks['Close'] - stocks['Open']) > 0).astype(int)

    # 归一化
    scaler = MinMaxScaler()
    stocks_scaled = pd.DataFrame(scaler.fit_transform(stocks), columns=stocks.columns)

    # 前80%用于训练集，后20%用于测试集
    train_size = int(len(stocks) * 0.85)
    test_size = len(stocks) - train_size
    train = stocks_scaled.iloc[0:train_size]
    test = stocks_scaled.iloc[train_size:len(stocks)].reset_index(drop=True)

    # 转换为tensor
    train_scaled = torch.tensor(np.array(train), dtype=torch.float32)
    test_scaled = torch.tensor(np.array(test), dtype=torch.float32)

    # 滑动堆叠每个时序的数据
    train_stacked = torch.stack([train_scaled[i:i + seq_length] for i in range(len(train_scaled) - seq_length + 1)])
    test_stacked = torch.stack([test_scaled[i:i + seq_length] for i in range(len(test_scaled) - seq_length + 1)])

    train_X, train_y = split_dataset(train_stacked, binary)
    test_X, test_y = split_dataset(test_stacked, binary)

    train_dataset = StockDataSet(train_X, train_y)
    test_dataset = StockDataSet(test_X, test_y)

    return (data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0),
            data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0))


def load_backtest_data(select_stock='PetroChina'):
    stocks: pd.DataFrame | None = pd.read_csv('Database/' + select_stock + '.csv')
    stocks = stocks.loc[:, ["Close", "Open"]]
    train_size = int(len(stocks) * 0.85)
    test = stocks.iloc[train_size:len(stocks)].reset_index(drop=True)
    return test


load_data(10, 10, "ShanXiMeiYe", binary=True)
