import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from tslearn.metrics import dtw

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
PKL_DAILY = fr"{ticker}_futures_daily_vectors.pkl"
start_date = '2015-02-24'  # Дата начала анализа

# === Загрузка дневного датафрейма ===
df = pd.read_pickle(PKL_DAILY)
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)
df.dropna(inplace=True)  # Удаление строк с NaN

# === Поиск индекса строки с нужной датой ===
start_date_ts = pd.to_datetime(start_date)
mask = df['TRADEDATE'].dt.date == start_date_ts.date()
if not mask.any():
    raise ValueError(f"Дата {start_date} не найдена в TRADEDATE")

idx_bar = df.index[mask][0]

# === Текущий дневной вектор (список numpy-векторов) ===
day_vec = df.at[idx_bar, 'VECTORS']
day_vec = np.asarray(day_vec, dtype=float)

# === Словарь для результатов MAX_N ===
max_results = {}

# === Цикл по диапазону от 3 до 30 ===
for n in range(3, 31):
    # Проверка, что есть хотя бы n предыдущих дней
    if idx_bar < n:
        max_results[f"MAX_{n}"] = 0.0
        continue

    best_dist = None
    idx_bar_similar = None

    # Поиск наиболее похожего дня среди n предыдущих
    for shift in range(1, n + 1):
        idx_prev = idx_bar - shift
        prev_vec = df.at[idx_prev, 'VECTORS']
        prev_vec = np.asarray(prev_vec, dtype=float)

        dist = dtw(day_vec, prev_vec)
        if (best_dist is None) or (dist < best_dist):
            best_dist = dist
            idx_bar_similar = idx_prev

    # Сравнение NEXT_BODY
    next_body_curr = df.at[idx_bar, 'NEXT_BODY']
    next_body_sim = df.at[idx_bar_similar, 'NEXT_BODY']

    sign_curr = np.sign(next_body_curr)
    sign_sim = np.sign(next_body_sim)

    value = abs(next_body_curr)
    if sign_curr == 0 or sign_sim == 0:
        max_results[f"MAX_{n}"] = 0.0
    elif sign_curr == sign_sim:
        max_results[f"MAX_{n}"] = value
    else:
        max_results[f"MAX_{n}"] = -value

# === Формирование df_rez ===
df_rez = pd.DataFrame(
    [{
        "TRADEDATE": df.at[idx_bar, "TRADEDATE"],
        "IDX_BAR": idx_bar,
        **max_results  # распаковка словаря в колонки MAX_3, MAX_4, ..., MAX_30
    }]
)

with pd.option_context(
        "display.width", 1000,
        "display.max_columns", 30,
        "display.max_colwidth", 100
):
    print("Датафрейм с результатом:")
    print(df_rez)
