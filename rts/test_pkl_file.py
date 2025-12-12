import pandas as pd

# PKL_FILE = r"RTS_futures_minute_2015_vectors.pkl"
PKL_FILE = r"RTS_futures_daily_vectors.pkl"

# Загрузка DataFrame из pkl
df = pd.read_pickle(PKL_FILE)

# Вывод первых 5 строк
print("Первые 5 строк:")
print(df.head())

# Вывод последних 5 строк
print("\nПоследние 5 строк:")
print(df.tail())

# Проверка типа данных колонки VECTORS
print("\nТип данных колонки VECTORS:")
print(df["VECTORS"].dtype)
print("\nПример вектора (первая строка):")
print(df["VECTORS"].iloc[0])
