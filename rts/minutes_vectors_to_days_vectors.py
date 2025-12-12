import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Пути к файлам
PKL_MINUTE = r"RTS_futures_minute_2015_vectors.pkl"
DB_PATH = r"C:\Users\Alkor\gd\data_quote_db\RTS_futures_minute_2015.db"
TABLE_NAME = "Futures"  # <-- замените на реальное имя таблицы
PKL_DAILY = r"RTS_futures_daily_vectors.pkl"

tqdm.pandas()


def load_minute_vectors(pkl_path: str) -> pd.DataFrame:
    """
    Загружаем df с колонками:
    TRADEDATE (datetime), VECTORS (np.array)
    """
    df = pd.read_pickle(pkl_path)  # TRADEDATE, VECTORS
    # гарантируем datetime и сортировку
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.sort_values("TRADEDATE").reset_index(drop=True)
    return df


def load_ohlc_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Загружаем исходные минутные OHLC из SQLite для вычисления BODY.
    """
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT
            TRADEDATE,
            OPEN,
            CLOSE
        FROM {table_name}
        ORDER BY TRADEDATE
    """
    df = pd.read_sql_query(query, conn, parse_dates=["TRADEDATE"])
    conn.close()
    return df


def compute_daily_body(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    BODY = CLOSE(last bar of day) - OPEN(first bar of day)
    Возвращает DataFrame с колонками:
        TRADEDATE (date), BODY (float)
    """
    df_ohlc = df_ohlc.copy()
    df_ohlc["DATE"] = df_ohlc["TRADEDATE"].dt.date

    # сортируем один раз
    df_ohlc = df_ohlc.sort_values("TRADEDATE")

    # первый и последний бар дня
    first = df_ohlc.groupby("DATE").first()
    last = df_ohlc.groupby("DATE").last()

    # first: колонки OPEN, CLOSE, TRADEDATE
    # last:  колонки OPEN, CLOSE, TRADEDATE

    body = last["CLOSE"] - first["OPEN"]

    df_body = body.reset_index()
    df_body = df_body.rename(columns={"DATE": "TRADEDATE", 0: "BODY"})
    # после reset_index() колонка с разностью будет называться так же, как Series (обычно 0),
    # поэтому выше переименовали её в BODY.

    # TRADEDATE как date
    df_body["TRADEDATE"] = pd.to_datetime(df_body["TRADEDATE"]).dt.date

    return df_body


def build_daily_vectors(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Группируем минутные вектора по дате и склеиваем в большие массивы.
    На входе:
        TRADEDATE (datetime), VECTORS (np.array)
    На выходе:
        TRADEDATE (date), VECTORS (np.array shape [N_day, dim])
    """
    df = df_minute.copy()
    df["DATE"] = df["TRADEDATE"].dt.date

    # группируем по дате
    groups = df.groupby("DATE")

    daily_records = []
    for date, g in tqdm(groups, desc="Building daily vectors"):
        # g["VECTORS"] — это Series из np.array одинаковой длины (dim=7)
        vectors_list = g["VECTORS"].tolist()
        # склеиваем по оси 0 -> (N_day, dim)
        day_matrix = np.stack(vectors_list, axis=0).astype(np.float32)
        daily_records.append((date, day_matrix))

    df_daily = pd.DataFrame(daily_records, columns=["TRADEDATE", "VECTORS"])
    return df_daily


def main():
    # Проверки
    if not Path(PKL_MINUTE).exists():
        raise FileNotFoundError(f"Minute vectors pkl not found: {PKL_MINUTE}")
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    # 1. Загружаем минутные вектора
    df_minute = load_minute_vectors(PKL_MINUTE)

    # 2. Строим дневные массивы VECTORS
    df_daily_vectors = build_daily_vectors(df_minute)

    # 3. Загружаем OHLC для вычисления дневного BODY
    df_ohlc = load_ohlc_from_sqlite(DB_PATH, TABLE_NAME)
    df_body = compute_daily_body(df_ohlc)

    # 4. Мёрджим дневные VECTORS и BODY по дате
    # df_daily_vectors.TRADEDATE и df_body.TRADEDATE — тип date
    df_daily = pd.merge(
        df_daily_vectors,
        df_body,
        on="TRADEDATE",
        how="inner",
        validate="one_to_one",
    )

    # 5. Добавляем NEXT_BODY (BODY со сдвигом вверх на 1 день)
    df_daily = df_daily.sort_values("TRADEDATE").reset_index(drop=True)
    df_daily["NEXT_BODY"] = df_daily["BODY"].shift(-1)

    # 6. Сохраняем результат
    df_daily.to_pickle(PKL_DAILY)
    print(f"Saved daily dataframe with {len(df_daily)} rows to {PKL_DAILY}")


if __name__ == "__main__":
    main()
