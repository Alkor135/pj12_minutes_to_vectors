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

PKL_FILE_LST = [
    fr"{ticker}_futures_minute_2015_vectors.pkl",
    fr"{ticker}_futures_daily_vectors.pkl",
]

for file in PKL_FILE_LST:
    # Загрузка DataFrame из pkl
    df = pd.read_pickle(file)

    print(f"\nФайл: {file}")
    with pd.option_context(
            "display.width", 1000,
            "display.max_columns", 30,
            "display.max_colwidth", 100
    ):
        print("Первые 5 строк:")
        print(df.head())
        print("\nПоследние 5 строк:")
        print(df.tail())

    # Проверка типа данных колонки VECTORS
    print("\nТип данных колонки VECTORS:")
    print(df["VECTORS"].dtype)
    print("\nПример вектора (первая строка):")
    print(df["VECTORS"].iloc[0])
