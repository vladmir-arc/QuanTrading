import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Data_Preprocess import load_data


# 定义LSTM类
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_size)  # x.size(1)即seq_length
        c0 = torch.zeros(1, x.size(1), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


output_size = 1
# 超参数
hidden_size = 64
num_epochs = 100
learning_rate = 0.01
batch_size = 15


def main():
    global input_size
    train_iter, test_iter = load_data(batch_size)
    for X, y in train_iter:
        input_size = X.shape[2]
        print("input_size = ", input_size)
        break
    model = LSTM(input_size, hidden_size, output_size)
    train_lstm(model, train_iter, test_iter, nn.MSELoss(), learning_rate)


# 每一个epoch的训练细节
def train_epoch(net, train_iter, loss, updater, epoch):
    global ls
    for X, y in train_iter:
        # print(X.size())
        y_hat = net(X)
        ls = loss(y_hat, y)
        updater.zero_grad()
        ls.backward()
        updater.step()
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, ls.item()))


# 训练LSTM
def train_lstm(net, train_iter, test_iter, loss, lr):
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, optimizer, epoch)
        # outputs = net(trainX)
        # loss = loss(outputs, trainY)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    y_list = []
    y_hat_list = []

    # 验证
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            y_list += y.tolist()
            y_hat_list += y_hat.tolist()
            rmse = torch.norm(y_hat-y)
            print('Test RMSE: {:.5f}'.format(rmse))

    # 绘制折线图
    plt.plot(y_list, label='y')
    plt.plot(y_hat_list, label='y_hat')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
