import yfinance as yf

PetroChina = yf.download('601857.SS', start='2020-01-01', end='2022-01-01')
# PetroChemical = yf.download('600028.SS', start='2020-01-01', end='2022-01-01')

PetroChina.to_csv('Database/PetroChina.csv')
# PetroChemical.to_csv('Database/PetroChemical.csv')
