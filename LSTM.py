import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from Data_Preprocess import load_data, load_backtest_data
from EarlyStop import EarlyStopping
from BackTest import BackTest


# 定义LSTM类
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, binary=False):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.binary = binary
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_size))  # 将h0,c0作为模型参数去训练
        self.c0 = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x):
        out, _ = self.lstm(x, (self.h0.expand(1, x.size(0), hidden_size),
                               self.c0.expand(1, x.size(0), hidden_size)))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        if self.binary:
            out = self.sigmoid(out)
        return out


# 获取数据库中已有的股票，返回股票列表
def get_stocks(path: str) -> list:
    files = os.listdir(path)
    stocks = []
    i = 0
    for file in files:
        if file.endswith('.csv'):
            stocks.append(file[:-4])
            print(i, file[:-4])
            i = i + 1
    return stocks


# 每一个epoch的训练细节
def train_epoch(net, train_iter, loss, updater, epoch, train_losses, early_stop):
    running_loss = 0.0

    for X, y in train_iter:
        # print(X.size())
        updater.zero_grad()
        y_hat = net(X)
        ls = loss(y_hat, y)
        ls.backward()
        updater.step()
        running_loss += ls.item() * X.size(0)

    epoch_loss = running_loss / len(train_iter.dataset)
    train_losses.append(epoch_loss)
    early_stop(epoch_loss)

    if (epoch+1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, epoch_loss))


# 训练LSTM
def train_lstm(net, train_iter, test_iter, loss, lr):
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    early_stopping = EarlyStopping()
    train_loss = []

    net.train()

    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, optimizer, epoch, train_loss, early_stopping)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # 绘制Loss曲线
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)


def eval_model(model, test_iter):
    y_close_list = []
    y_closehat_list = []
    y_open_list = []
    y_openhat_list = []

    # 验证
    model.eval()
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = model(X)
            y_close_list += y[:, 0].tolist()
            y_open_list += y[:, 1].tolist()
            y_closehat_list += y_hat[:, 0].tolist()
            y_openhat_list += y_hat[:, 1].tolist()
            rmse = torch.norm(y_hat - y)
            print('Test RMSE: {:.5f}'.format(rmse))

    # 绘制折线图
    plt.plot(y_close_list, label='Actual Close')
    plt.plot(y_closehat_list, label='Predicted Close', linestyle='--')
    plt.plot(y_open_list, label='Actual Open')
    plt.plot(y_openhat_list, label='Predicted Open', linestyle='--')
    plt.legend()

    # 计算准确率
    win_list = [True if i > j else False for i, j in zip(y_close_list, y_open_list)]
    win_hat_list = [True if i > j else False for i, j in zip(y_closehat_list, y_openhat_list)]
    count = sum([1 for i, j in zip(win_list, win_hat_list) if i == j])
    accuracy = float(count / len(win_list))
    print('准确率为', accuracy*100, '%')
    with open('buy_signal.txt', 'w') as f:
        for x, y in zip(win_list, win_hat_list):
            f.write(f'{x} {y}\n')


def eval_binary_model(model, test_iter):
    model.eval()
    win_list = []
    win_hat_list = []

    with torch.no_grad():
        for X, y in test_iter:
            y_hat = model(X)
            binary_pred = (y_hat > 0.5).int().squeeze(-1).tolist()
            win_list += y.squeeze(-1).tolist()
            win_hat_list += binary_pred
            back_test.update(binary_pred)

    count = sum([1 for i, j in zip(win_list, win_hat_list) if i == j])
    count_winhat_1 = sum([1 for i in win_hat_list if i == 1.0])
    count_win = sum([1 for i, j in zip(win_list, win_hat_list) if (i==1 and i==j)])
    accuracy = float(count / len(win_list))
    # precision = float(count_win / count_winhat_1)
    print('准确率为', accuracy * 100, '%')
    # print('精确率为', precision*100, '%')
    with open('win_signal.txt', 'w') as f:
        for x, y in zip(win_list, win_hat_list):
            f.write(f'{x} {y}\n')


def main(s_stock: str, load: bool = False, binary: bool = False):
    """
    主函数入口

    :param s_stock: 选取的一支股票
    :param load: 若为真，则加载训练好的模型
    :param binary: 若为真，则选用二分类预测
    """
    global input_size, output_size
    train_iter, test_iter = load_data(batch_size, seq_length, s_stock, binary)

    # 获取输入的维度
    for X, y in train_iter:
        input_size = X.shape[2]
        output_size = y.shape[1]
        print("\ninput_dim = ", input_size, "\noutput_dim = ", output_size)
        break

    model = LSTM(input_size, hidden_size, output_size, binary=True)
    if not load:
        criterion = nn.BCELoss() if binary else nn.MSELoss()
        # 训练模型
        train_lstm(model, train_iter, test_iter, criterion, learning_rate)
        # 验证模型
        if binary:
            eval_binary_model(model, test_iter)
        else:
            eval_model(model, test_iter)

        # 保存训练好的模型
        save = bool(int(input("Save model? 1 for Yes, 0 for No")))
        if save:
            torch.save(model, 'Params/'+s_stock+'.pt')

    else:
        # 直接调用已存储的模型
        clone = torch.load('Params/'+s_stock+'.pt')
        clone.eval()
        if binary:
            eval_binary_model(clone, test_iter)
        else:
            eval_model(clone, test_iter)
            # 添加图标题——当前股票
            plt.title(s_stock)
            plt.show()

    backtest_data = load_backtest_data(s_stock)
    back_test.back_test(backtest_data, seq_length)
    back_test.show(s_stock)


# 超参数
hidden_size = 30
num_epochs = 160
learning_rate = 0.007
batch_size = 16
seq_length = 7
back_test = BackTest(10000, 0, 1000)

if __name__ == '__main__':
    stocks = get_stocks('Database')
    i = eval(input('输入股票序号:'))
    main(stocks[i], load=True, binary=True)
