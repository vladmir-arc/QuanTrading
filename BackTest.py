import matplotlib.pyplot as plt


class BackTest:
    def __init__(self, start_cash, commission, trans_amount):
        self.signals = []
        self.start_cash = start_cash
        self.commission = commission
        self.trans_amount = trans_amount
        self.balance = [self.start_cash]
        # 获取股票数据

    def update(self, signal):
        self.signals += signal

    def back_test(self, data, seq_length):
        # 计算持有多少股
        for index, row in data.iterrows():
            if index < seq_length:
                continue
            if self.signals[index - seq_length]:
                close_price = row['Close']
                open_price = row['Open']
                benefit = (close_price - open_price) / open_price
                # 计算当天收益，即(close - open) * 股数
                today_benefit = self.trans_amount * benefit
                self.start_cash += today_benefit
                self.balance.append(self.start_cash)
            else:
                self.balance.append(self.start_cash)

    def show(self, stock: str):
        plt.plot(self.balance, label='Balance')
        plt.legend()
        plt.title(stock + ' - Profit Chart')
        plt.show()
