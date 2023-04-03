import pandas as pd

# 从csv文件中读取数据
data = pd.read_csv('Database/test.csv')


# 生成信号
def generate_signals(data, keep_previous=True):
    data["signal"] = 0
    # 按每行循环
    for i in range(50, len(data)):
        # 如果10-day_SMA超过50-day_SMA，设置信号为1，即买入
        if data["SMA_10"][i] > data["SMA_50"][i] and data["SMA_10"][i - 1] < data["SMA_50"][i - 1]:
            data.loc[i, 'signal'] = 1  # loc[m, n]先行后列
        # 如果10-day_SMA跌落50-day_SMA，设置信号为0，即卖出
        elif data["SMA_10"][i] < data["SMA_50"][i] and data["SMA_10"][i - 1] > data["SMA_50"][i - 1]:
            data.loc[i, "signal"] = -1
        # 若以上都不是，则保持前一天的信号
        elif keep_previous:
            data.loc[i, "signal"] = data.loc[i - 1, "signal"]
        else:
            data.loc[i, 'signal'] = 0
    return data


data = generate_signals(data)
print(data.tail(10))
