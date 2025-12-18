import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import re


# Путь к settings.yaml
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
PKL_SIMILARITY = fr"{ticker}_dtw_similarity_weights.pkl"

# === Загрузка дневного датафрейма ===
df = pd.read_pickle(PKL_SIMILARITY)
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)
df.dropna(inplace=True)  # Удаление строк с NaN

# создаём PL_колонки и считаем rolling-сумму по 22 предыдущим строкам
for n in range(3, 31):
    max_col = f"MAX_{n}"
    pl_col = f"PL_{n}"

    if max_col not in df.columns:
        continue

    # 1) сначала PL_n = MAX_n (как копия значения текущей строки)
    df[pl_col] = df[max_col]

    # 2) затем в PL_n записываем сумму 22 предыдущих значений MAX_n, без текущей
    # shift(1) сдвигает столбец вниз, чтобы исключить текущую строку,
    # rolling(22).sum() берёт сумму по окну из 22 строк над текущей
    df[pl_col] = df[max_col].shift(1).rolling(window=22, min_periods=1).sum()

with pd.option_context(  # Печать широкого и длинного датафрейма
        "display.width", 1000,
        "display.max_columns", 70,
        "display.max_colwidth", 100,
        "display.min_rows", 50,
):
    print("Датафрейм с результатом:")
    print(df)

# Список колонок PL_ и MAX_
pl_cols  = [c for c in df.columns if c.startswith("PL_")]
max_cols = [c for c in df.columns if c.startswith("MAX_")]

# Функция обработки одной строки
def process_row(row):
    # максимум по PL_3..PL_30
    max_pl_val = row[pl_cols].max()
    if max_pl_val <= 0.0 or pd.isna(max_pl_val):
        return pd.Series({"TRADEDATE": row["TRADEDATE"], "P/L": 0.0})

    # имя колонки, где этот максимум
    col_pl = row[pl_cols].idxmax()          # например, 'PL_7'
    n = int(re.findall(r"\d+", col_pl)[0])  # 7

    max_col_name = f"MAX_{n}"
    pl_value = row[max_col_name]

    return pd.Series({"TRADEDATE": row["TRADEDATE"], "P/L": pl_value})

# Формирование df_rez
df_rez = df.apply(process_row, axis=1)

print(df_rez)

# Кумулятивная сумма P/L
df_rez["Cum_P/L"] = df_rez["P/L"].cumsum()

# График кумулятивной суммы
plt.figure(figsize=(10, 5))
plt.plot(df_rez["TRADEDATE"], df_rez["Cum_P/L"])
plt.xlabel("TRADEDATE")
plt.ylabel("Cumulative P/L")
plt.title("Кумулятивная сумма P/L")
plt.grid(True)
plt.tight_layout()
# plt.show()

# Сохранение графика
output_plot = Path(__file__).parent / f"{ticker}_cumsum_plot_max.png"
plt.savefig(output_plot)
plt.close()