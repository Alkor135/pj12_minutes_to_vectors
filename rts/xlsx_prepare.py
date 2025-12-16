import pandas as pd
import yaml
from pathlib import Path

from rts.rts_download_minutes_to_db import start_date

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
PKL_DAILY = fr"{ticker}_futures_daily_vectors.pkl"
max_prev_days = (3, 30)  # Минимальное и максимальное количество предыдущих дней для сравнения векторов
start_date = '2015-06-01'  # Дата начала анализа

df = pd.read_pickle(PKL_DAILY)  # Чтение DataFrame из pkl файла

# Преобразование колонки TRADEDATE в тип datetime и сортировка по ней
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)

df_rez = pd.DataFrame()  # Создание пустого DataFrame для результатов



print(df_rez)
