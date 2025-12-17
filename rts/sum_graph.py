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
df.dropna(inplace=True)  # Удаление строк с NaN

# === Построение графиков кумулятивной суммы ===
columns_to_plot = [col for col in df.columns if col.startswith('MAX_')]

plt.figure(figsize=(24 , 12))
for col in columns_to_plot:
    plt.plot(df['TRADEDATE'], df[col].cumsum(), label=col)

plt.title(f'Кумулятивные суммы для {ticker}')
plt.xlabel('Дата')
plt.ylabel('Кумулятивная сумма')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# Сохранение графика
output_plot = Path(__file__).parent / f"{ticker}_cumsum_plot.png"
plt.savefig(output_plot)
plt.close()

print(f"График сохранён: {output_plot}")