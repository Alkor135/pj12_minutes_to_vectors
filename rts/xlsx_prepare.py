import pandas as pd
import yaml
from pathlib import Path

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
PKL_DAILY = fr"{ticker}_futures_daily_vectors.pkl"
max_prev_days = settings['max_prev_days']

df = pd.read_pickle(PKL_DAILY)  #

# Преобразование колонки TRADEDATE в тип datetime и сортировка по ней
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)

# Проверка типа колонки TRADEDATE
print(f"Тип колонки TRADEDATE: {df['TRADEDATE'].dtype}")

print(df)
