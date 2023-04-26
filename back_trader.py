import backtrader as bt
import backtrader.analyzers as btanalyzers
from datetime import datetime
from Data_Preprocess import load_backtrader_data
import matplotlib.pyplot as plt


class MyStrategy(bt.Strategy):
    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.index = 0
        self.dataclose = self.datas[0].close
        self.bar_executed = 0

    def log(self, txt, dt=None):
        # 根据需求输出一条日志
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        # 每次交易执行时输出日志
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price, dt=self.datas[0].datetime.date(-1))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        # 输出此次交易的盈亏
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next_open(self):
        # 在开盘时交易
        for stock in self.datas:
            s_name = stock._name
            if signals[s_name][self.index]:
                self.buy(data=stock, size=180, coc=False)
                # print(self.index, 'buy!!')

    def next(self):
        # 根据每条Line，做出交易
        self.log('Close, %.2f' % self.dataclose[0])
        for stock in self.datas:
            pos = self.getposition(stock).size
            if pos:
                self.sell(data=stock, size=180)
                # print(self.index, 'sell!!')
        self.index += 1

    def start(self):
        # 第一天开盘交易
        for stock in self.datas:
            s_name = stock._name
            if signals[s_name][self.index]:
                self.buy(data=stock, size=180, coc=False)
                # print(self.index, 'buy!')


def cerebro_run(select_stock='ShanXiMeiYe', multi_strand=False):
    cerebro = bt.Cerebro(cheat_on_open=True)

    cerebro.broker.setcash(10000.0)

    if multi_strand:
        # 获取股票列表
        for stock in stocks:
            data_frame = load_backtrader_data(stock, seq_length=7)

            data = bt.feeds.PandasData(
                name=stock,
                dataname=data_frame,
                open=0,
                high=1,
                low=2,
                close=3,
                volume=4,
                openinterest=-1
            )

            cerebro.adddata(data)

    else:
        data_frame = load_backtrader_data(select_stock, seq_length=7)

        # Create a DataFeed
        data = bt.feeds.PandasData(
            name=select_stock,
            dataname=data_frame,
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )

        cerebro.adddata(data)

    cerebro.addstrategy(MyStrategy)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='AnnualReturn')

    cerebro.broker.set_coc(True)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    thestrats = cerebro.run()
    thestrat = thestrats[0]

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('Sharpe Ratio:', thestrat.analyzers.mysharpe.get_analysis())
    print('Annual Return:', thestrat.analyzers.AnnualReturn.get_analysis())

    cerebro.plot(style='candlestick')


def load_signals() -> dict:
    """
    在保存的预测信号文件(.txt)中解析日间交易信号

    :return: 日间交易信号
    """
    sig_dict = dict()
    for stock in stocks:
        with open('Signals/' + stock + '_win_signal.txt', 'r') as file:
            data = [line.strip() for line in file]
            result = [int(x.split()[1]) for x in data]  # 去空格后半部分即第二列
            sig_dict[stock] = result
            # print(result)
    return sig_dict


#stocks = ['PetroChemical', 'PetroChina', 'ShanMeiGuoJi', 'ShanXiMeiYe', 'ZhongGuoShenHua']
stocks = ['ShanXiMeiYe', 'ZhongGuoShenHua']

if __name__ == '__main__':
    signals = load_signals()
    cerebro_run(multi_strand=True)
