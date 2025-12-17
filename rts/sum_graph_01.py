import pandas as pd
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

# Путь к settings.yaml
SETTINGS_FILE = Path(__file__).parent / "settings.yaml"

# Чтение настроек
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)

ticker = settings['ticker']
PKL_DTW = fr"{ticker}_dtw_similarity_weights.pkl"

# === Загрузка дневного датафрейма ===
df = pd.read_pickle(PKL_DTW)
df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
df = df.sort_values('TRADEDATE').reset_index(drop=True)
df.dropna(inplace=True)

# === Кумулятивные суммы для MAX_ колонок ===
columns_to_plot = [col for col in df.columns if col.startswith('MAX_')]
cumsum_data = df[columns_to_plot].cumsum()

# === Выбор топ-5 колонок с наибольшим значением на последней дате ===
final_values = cumsum_data.iloc[-1]  # Значения кумсум на последнюю дату
top5_columns = final_values.nlargest(5).index.tolist()

# === Построение графиков только для топ-5 ===
plt.figure(figsize=(24, 12))
for col in top5_columns:
    plt.plot(df['TRADEDATE'], cumsum_data[col], label=f"{col} (финал: {final_values[col]:.2f})")

plt.title(f'Топ-5 кумулятивных сумм по доходности на конец периода — {ticker}')
plt.xlabel('Дата')
plt.ylabel('Кумулятивная сумма')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Сохранение графика
output_plot = Path(__file__).parent / f"{ticker}_cumsum_top5_plot.png"
plt.savefig(output_plot)
plt.close()

print(f"График сохранён: {output_plot}")