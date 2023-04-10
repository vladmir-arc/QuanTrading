import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from Data_Preprocess import load_data
from EarlyStop import EarlyStopping


# 定义LSTM类
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(hidden_size, output_size)
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_size))  # 将h0,c0作为模型参数去训练
        self.c0 = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x):
        # h0 = torch.randn(1, x.size(0), self.hidden_size)*0.01  # x.size(1)即seq_length
        # c0 = torch.randn(1, x.size(0), self.hidden_size)*0.01
        out, _ = self.lstm(x, (self.h0.expand(1, x.size(0), hidden_size),
                               self.c0.expand(1, x.size(0), hidden_size)))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
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
def train_epoch(net, train_iter, loss, updater, epoch, train_losses):
    global ls
    for X, y in train_iter:
        # print(X.size())
        y_hat = net(X)
        ls = loss(y_hat, y)
        updater.zero_grad()
        ls.backward()
        updater.step()
        train_losses.append(ls.item())
    if (epoch+1) % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, ls.item()))


# 训练LSTM
def train_lstm(net, train_iter, test_iter, loss, lr):
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    early_stopping = EarlyStopping()
    net.train()

    for epoch in range(num_epochs):
        train_losses = []
        train_epoch(net, train_iter, loss, optimizer, epoch, train_losses)
        # outputs = net(trainX)
        # loss = loss(outputs, trainY)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        loss_for_stop = np.average(train_losses)
        early_stopping(loss_for_stop)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    eval_model(net, test_iter)


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
    buy_list = [True if i > j else False for i, j in zip(y_close_list, y_open_list)]
    buy_hat_list = [True if i > j else False for i, j in zip(y_closehat_list, y_openhat_list)]
    count = sum([1 for i, j in zip(buy_list, buy_hat_list) if i == j])
    accuracy = float(count / len(buy_list))
    print('准确率为', accuracy*100, '%')
    with open('buy_signal.txt', 'w') as f:
        for x, y in zip(buy_list, buy_hat_list):
            f.write(f'{x} {y}\n')


def main(s_stock: str, load: bool = False):
    """
    主函数入口

    :param s_stock: 选取的一支股票
    :param load: 若为真，则加载训练好的模型
    """
    global input_size, output_size
    train_iter, test_iter = load_data(batch_size, seq_length, s_stock)

    # 获取输入的维度
    for X, y in train_iter:
        input_size = X.shape[2]
        output_size = y.shape[1]
        print("\ninput_dim = ", input_size, "\noutput_dim = ", output_size)
        break

    model = LSTM(input_size, hidden_size, output_size)
    if not load:
        train_lstm(model, train_iter, test_iter, nn.MSELoss(), learning_rate)
        # 保存训练好的模型
        save = bool(int(input("Save model? 1 for Yes, 0 for No")))
        if save:
            torch.save(model, 'Params/'+s_stock+'.pt')
    else:
        clone = torch.load('Params/'+s_stock+'.pt')
        clone.eval()
        eval_model(clone, test_iter)

    # 添加图标题——当前股票
    plt.title(s_stock)
    plt.show()


# 超参数
hidden_size = 200
num_epochs = 200
learning_rate = 0.01
batch_size = 16
seq_length = 10


if __name__ == '__main__':
    stocks = get_stocks('Database')
    i = eval(input('输入股票序号:'))
    main(stocks[i], load=False)
