import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from tslearn.metrics import dtw
from tqdm import tqdm

# Путь к settings.yaml
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
PKL_DAILY = fr"{ticker}_futures_daily_vectors.pkl"
# start_date = '2015-01-01'

# === Загрузка дневного датафрейма ===
df = pd.read_pickle(PKL_DAILY)
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)
df.dropna(inplace=True)  # Удаление строк с NaN
# df = df.head(35)  # Ограничение для тестирования

# # === Фильтрация строк после start_date ===
# start_date_ts = pd.to_datetime(start_date)
# df = df[df['TRADEDATE'] >= start_date_ts].reset_index(drop=True)

# === Инициализация df_rez ===
df_rez = pd.DataFrame()

# === Обработка каждой строки с прогресс-баром ===
for idx_bar in tqdm(df.index, desc="Processing rows"):
    day_vec = df.at[idx_bar, 'VECTORS']
    day_vec = np.asarray(day_vec, dtype=float)

    max_results = {}

    for n in range(3, 31):
        if idx_bar < n:
            max_results[f"MAX_{n}"] = 0.0
            continue

        best_dist = None
        idx_bar_similar = None

        for shift in range(1, n + 1):
            idx_prev = idx_bar - shift
            prev_vec = df.at[idx_prev, 'VECTORS']
            prev_vec = np.asarray(prev_vec, dtype=float)

            dist = dtw(day_vec, prev_vec)
            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                idx_bar_similar = idx_prev

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

    # Добавление строки в df_rez
    df_rez = pd.concat([
        df_rez,
        pd.DataFrame([{
            "TRADEDATE": df.at[idx_bar, "TRADEDATE"],
            # "IDX_BAR": idx_bar,
            **max_results
        }])
    ], ignore_index=True)

with pd.option_context(  # Печать широкого и длинного датафрейма
        "display.width", 1000,
        "display.max_columns", 30,
        "display.max_colwidth", 100
):
    print("Датафрейм с результатом:")
    print(df_rez)

# Сохранение df_rez в pkl файл
df_rez.to_pickle(f"{ticker}df_tmp.pkl")
print(f"df_rez saved to {ticker}df_tmp.pkl")
