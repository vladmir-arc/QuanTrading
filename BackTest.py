import matplotlib.pyplot as plt
from datetime import datetime
from Data_Preprocess import load_backtest_data


class BackTest:
    def __init__(self, start_cash, commission, trans_amount):
        self.signals = []
        self.current_cash = start_cash
        self.commission = commission
        self.trans_amount = trans_amount
        self.balance = [start_cash]
        # 存储回测股票的数据
        self.stock_dic = {}

    def update(self, signal):
        self.signals += signal

    def parse_data(self, data, seq_length):
        # 解析回测股票数据并保存
        columns = data.columns
        stock_info = [data[col_name].tolist() for col_name in columns]
        for index, col_name in enumerate(columns):
            self.stock_dic[col_name] = stock_info[index][seq_length:]

    def back_test(self, data, seq_length):
        """
            回测函数主体

            :param data: 传入测试数据集
            :param seq_length: 训练时的时序长度
        """
        self.parse_data(data, seq_length)
        for index, signal in enumerate(self.signals):
            if signal:
                close = self.stock_dic['Close'][index]
                open = self.stock_dic['Open'][index]
                # 计算当天收益率
                benefit = (close - open) / open
                today_benefit = self.trans_amount * open * benefit
                self.current_cash += today_benefit
                self.balance.append(self.current_cash)
            else:
                self.balance.append(self.current_cash)

        # for index, row in data.iterrows():
        #     if index < seq_length:
        #         continue
        #     if self.signals[index - seq_length]:
        #         close_price = row['Close']
        #         open_price = row['Open']
        #         benefit = (close_price - open_price) / open_price
        #         # 计算当天收益，即(close - open) * 股数
        #         today_benefit = self.trans_amount * benefit
        #         self.current_cash += today_benefit
        #         self.balance.append(self.current_cash)
        #     else:
        #         self.balance.append(self.current_cash)

    def show(self, stock: str):
        date_format = '%Y/%m/%d'
        date_object = [datetime.strptime(date_string, date_format) for date_string in self.stock_dic['Date']]
        self.stock_dic['Date'] = date_object

        plt.plot(self.balance, label='Balance')
        plt.legend()
        plt.title(stock + ' - Profit Chart')
        plt.show()
        print(self.balance[-1])


bt = BackTest(10000, 0, 100)
stock = 'ShanXiMeiYe'
stock_data = load_backtest_data(stock)
bt.parse_data(stock_data, 7)
