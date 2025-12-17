"""
Скрипт для анализа дневных векторов фьючерсов с помощью DTW.
Загружает данные из pickle-файла, включая временные ряды (VECTORS).
Выбирает дату начала анализа и находит индекс соответствующего бара.
Сравнивает вектор текущего дня с векторами трёх предыдущих дней.
Использует DTW для расчёта расстояний между многомерными временными рядами.
Определяет наиболее похожий день по минимальному DTW-расстоянию.
Формирует результат: совпадение/несовпадение знака приращения с похожим днём.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from tslearn.metrics import dtw  # pip install tslearn

# Путь к settings.yaml в той же директории, что и скрипт
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

# ==== Параметры ====
ticker = settings['ticker']
PKL_DAILY = fr"{ticker}_futures_daily_vectors.pkl"

max_prev_days = (3, 30)  # сейчас используем только 3, второй элемент пригодится позже
start_date = '2015-02-24'  # Дата начала анализа

# === Загрузка дневного датафрейма ===
df = pd.read_pickle(PKL_DAILY)  # Ожидаются колонки: TRADEDATE, VECTORS, BODY, NEXT_BODY

# Преобразование TRADEDATE в datetime и сортировка
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)

# === Поиск индекса строки с нужной датой ===
start_date_ts = pd.to_datetime(start_date)
mask = df['TRADEDATE'].dt.date == start_date_ts.date()
if not mask.any():
    raise ValueError(f"Дата {start_date} не найдена в TRADEDATE")

idx_bar = df.index[mask][0]

# Проверка, что есть хотя бы 3 предыдущих дня
if idx_bar < 3:
    raise ValueError(f"Для {start_date} недостаточно предыдущих дней (idx_bar={idx_bar})")

# === Текущий дневной вектор (список numpy-векторов) ===
day_vec = df.at[idx_bar, 'VECTORS']  # ожидаем массив shape (N_t, dim)

# Приведём к np.ndarray (на случай, если это список списков)
day_vec = np.asarray(day_vec, dtype=float)

# === Сравнение с тремя предыдущими днями по DTW ===
best_dist = None
idx_bar_similar = None

for shift in range(1, 4):  # 1, 2, 3 дня назад
    idx_prev = idx_bar - shift
    prev_vec = df.at[idx_prev, 'VECTORS']
    prev_vec = np.asarray(prev_vec, dtype=float)

    # DTW для многомерных рядов: считаем расстояние между последовательностями векторов
    # (N_t1, dim) и (N_t2, dim); длины могут отличаться
    dist = dtw(day_vec, prev_vec)

    if (best_dist is None) or (dist < best_dist):
        best_dist = dist
        idx_bar_similar = idx_prev

# === Сравнение NEXT_BODY и запись в df_rez ===
next_body_curr = df.at[idx_bar, 'NEXT_BODY']
next_body_sim = df.at[idx_bar_similar, 'NEXT_BODY']

# знак: np.sign возвращает -1, 0, 1
sign_curr = np.sign(next_body_curr)
sign_sim = np.sign(next_body_sim)

value = abs(next_body_curr)
if sign_curr == 0 or sign_sim == 0:
    # если один из них 0 — можно считать как "нет сигнала"
    max_3 = 0.0
elif sign_curr == sign_sim:
    max_3 = value
else:
    max_3 = -value

# === Формирование df_rez ===
df_rez = pd.DataFrame(
    [{
        "TRADEDATE": df.at[idx_bar, "TRADEDATE"],
        "IDX_BAR": idx_bar,
        "IDX_BAR_SIMILAR": idx_bar_similar,
        "NEXT_BODY": next_body_curr,
        "NEXT_BODY_SIMILAR": next_body_sim,
        "MAX_3": max_3,
    }]
)

print(df_rez)

