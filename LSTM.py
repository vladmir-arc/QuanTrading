import torch
import torch.nn as nn
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


input_size = 2
output_size = 1
# 超参数
hidden_size = 8
num_epochs = 100
learning_rate = 0.01
batch_size = 15


def main():
    train_iter, test_iter = load_data(batch_size)
    model = LSTM(input_size, hidden_size, output_size)
    train_lstm(model, train_iter, test_iter, nn.MSELoss(), learning_rate)


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

    # 验证
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            # test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
            # y = scaler.inverse_transform(testY.reshape(-1, 1))
            # rmse = np.sqrt(np.mean(((y_hat - y) ** 2)))
            rmse = torch.norm(y_hat-y)
            print('Test RMSE: {:.5f}'.format(rmse))


if __name__ == '__main__':
    main()
