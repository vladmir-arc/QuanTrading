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
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.randn(1, x.size(1), self.hidden_size)*0.01  # x.size(1)即seq_length
        c0 = torch.randn(1, x.size(1), self.hidden_size)*0.01
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


output_size = 1
# 超参数
hidden_size = 128
num_epochs = 100
learning_rate = 0.01
batch_size = 15
seq_length = 10


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

    eval_model(net, test_iter)
    # y_list = []
    # y_hat_list = []
    #
    # # 验证
    # net.eval()
    # with torch.no_grad():
    #     for X, y in test_iter:
    #         y_hat = net(X)
    #         y_list += y.tolist()
    #         y_hat_list += y_hat.tolist()
    #         rmse = torch.norm(y_hat-y)
    #         print('Test RMSE: {:.5f}'.format(rmse))
    #
    # # 绘制折线图
    # plt.plot(y_list, label='y')
    # plt.plot(y_hat_list, label='y_hat')
    # plt.legend()
    # plt.show()


def eval_model(model, test_iter):
    y_list = []
    y_hat_list = []

    # 验证
    model.eval()
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = model(X)
            y_list += y.tolist()
            y_hat_list += y_hat.tolist()
            rmse = torch.norm(y_hat - y)
            print('Test RMSE: {:.5f}'.format(rmse))

    # 绘制折线图
    plt.plot(y_list, label='y')
    plt.plot(y_hat_list, label='y_hat')
    plt.legend()
    plt.show()


def main(s_stock: str, load: bool = False):
    global input_size
    train_iter, test_iter = load_data(batch_size, seq_length, s_stock)

    # 获取输入的维度
    for X, y in train_iter:
        input_size = X.shape[2]
        print("input_size = ", input_size)
        break

    model = LSTM(input_size, hidden_size, output_size)
    if not load:
        train_lstm(model, train_iter, test_iter, nn.MSELoss(), learning_rate)
        torch.save(model.state_dict(), 'Params/'+s_stock+'.pt')
    else:
        clone = LSTM(input_size, hidden_size, output_size)
        clone.load_state_dict(torch.load('Params/'+s_stock+'.pt'))
        clone.eval()
        eval_model(clone, test_iter)


if __name__ == '__main__':
    select_stock = 'ZhongMeiNengYuan'  # TODO:高级选取股票
    main(select_stock, load=True)
