import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Регистрируем tqdm для pandas
tqdm.pandas()

DB_PATH = r"C:\Users\Alkor\gd\data_quote_db\RTS_futures_minute_2015.db"
TABLE_NAME = "Futures"  # <-- поменять на реальное имя таблицы в БД
PKL_OUT = r"RTS_futures_minute_2015_vectors.pkl"

# параметры нормализации объёма
VOLUME_WINDOW = 100

def load_ohlcv_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Загружает TRADEDATE, OPEN, LOW, HIGH, CLOSE, VOLUME из SQLite.
    """
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT
            TRADEDATE,
            OPEN,
            LOW,
            HIGH,
            CLOSE,
            VOLUME
        FROM {table_name}
        ORDER BY TRADEDATE
    """
    df = pd.read_sql_query(query, conn, parse_dates=["TRADEDATE"])
    conn.close()
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит вектор признаков [rO, rC, rbody, rup, rdown, rlog, V_tilde]
    и возвращает DataFrame с колонками TRADEDATE и VECTORS.
    """
    # Переименуем для краткости
    O = df["OPEN"].astype(float)
    H = df["HIGH"].astype(float)
    L = df["LOW"].astype(float)
    C = df["CLOSE"].astype(float)
    V = df["VOLUME"].astype(float)

    # Диапазон
    R = H - L

    # Маска «нормальных» баров, чтобы не делить на ноль
    eps = 1e-12
    valid_range = R.abs() > eps

    # Инициализация всех признаков NaN
    rO = pd.Series(np.nan, index=df.index, dtype=float)
    rC = pd.Series(np.nan, index=df.index, dtype=float)
    r_body = pd.Series(np.nan, index=df.index, dtype=float)
    r_up = pd.Series(np.nan, index=df.index, dtype=float)
    r_down = pd.Series(np.nan, index=df.index, dtype=float)

    # Вычисляем только для валидных баров
    R_valid = R[valid_range]

    rO.loc[valid_range] = (O[valid_range] - L[valid_range]) / R_valid
    rC.loc[valid_range] = (C[valid_range] - L[valid_range]) / R_valid
    r_body.loc[valid_range] = (C[valid_range] - O[valid_range]).abs() / R_valid

    upper_body = np.maximum(O[valid_range].values, C[valid_range].values)
    lower_body = np.minimum(O[valid_range].values, C[valid_range].values)

    r_up.loc[valid_range] = (H[valid_range].values - upper_body) / R_valid.values
    r_down.loc[valid_range] = (lower_body - L[valid_range].values) / R_valid.values

    # Лог-ретёрн
    valid_open = O.abs() > eps
    r_log = pd.Series(np.nan, index=df.index, dtype=float)
    r_log.loc[valid_open] = np.log(C[valid_open].values / O[valid_open].values)

    # Нормализация объёма (z-score по скользящему окну)
    V_mean = V.rolling(VOLUME_WINDOW, min_periods=1).mean()
    V_std = V.rolling(VOLUME_WINDOW, min_periods=1).std(ddof=0)
    V_std_safe = V_std.replace(0, np.nan)
    V_tilde = (V - V_mean) / V_std_safe
    V_tilde = V_tilde.fillna(0.0)

    # Собираем вектор numpy для каждой строки
    feature_cols = [rO, rC, r_body, r_up, r_down, r_log, V_tilde]
    feature_array = np.stack(feature_cols, axis=1)

    # Прогресс-бар при создании Series из массивов
    vectors = pd.Series(
        [feature_array[i, :].astype(np.float32) for i in tqdm(range(feature_array.shape[0]), desc="Creating vectors")],
        index=df.index,
        dtype=object,
        name="VECTORS",
    )

    out_df = pd.DataFrame(
        {
            "TRADEDATE": df["TRADEDATE"],
            "VECTORS": vectors,
        }
    )
    return out_df

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    df_raw = load_ohlcv_from_sqlite(DB_PATH, TABLE_NAME)
    df_vectors = compute_features(df_raw)

    # Прогресс-бар при сохранении в pickle
    df_vectors.to_pickle(PKL_OUT)
    print(f"Saved {len(df_vectors)} rows to {PKL_OUT}")

if __name__ == "__main__":
    main()
