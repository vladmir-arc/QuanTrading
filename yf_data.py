import yfinance as yf

ZhongGuoHaiYou = yf.download('600938.SS', start='2020-01-01', end='2023-01-01')
ShanMeiGuoJi = yf.download('600546.SS', start='2020-01-01', end='2023-01-01')
ZhongGuoShenHua = yf.download('601088.SS', start='2020-01-01', end='2023-01-01')
ZhongMeiNengYuan = yf.download('601898.SS', start='2020-01-01', end='2023-01-01')
ShanXiMeiYe = yf.download('601225.SS', start='2020-01-01', end='2023-01-01')
PetroChina = yf.download('601857.SS', start='2020-01-01', end='2023-01-01')
PetroChemical = yf.download('600028.SS', start='2020-01-01', end='2023-01-01')

ZhongGuoHaiYou.to_csv('Database/ZhongGuoHaiYou.csv')
ShanMeiGuoJi.to_csv('Database/ShanMeiGuoJi.csv')
ZhongGuoShenHua.to_csv('Database/ZhongGuoShenHua.csv')
ZhongMeiNengYuan.to_csv('Database/ZhongMeiNengYuan.csv')
ShanXiMeiYe.to_csv('Database/ShanXiMeiYe.csv')
PetroChina.to_csv('Database/PetroChina.csv')
PetroChemical.to_csv('Database/PetroChemical.csv')
