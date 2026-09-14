import polars as pl
import pandas as pd
from tqdm import tqdm
from datetime import timedelta


class NewsProcessor:
    """
    Raw news alignment/merging utilities.

    Note: Embedding generation (LLM few-shot S-R-O triple extraction ->
    schema-constrained triples -> quality filtering -> text embedding ->
    confidence-weighted daily aggregation per ticker, report Section 4.2
    "Financial news data") now runs in a separate preprocessing project.
    This module only handles raw article alignment; the resulting per-date/
    per-ticker embedding vectors are consumed directly via the JSON file
    ingested in main_test.py / data_pipeline/builder.py.
    """

    def merge_reuters_and_fix_nulls(self, alpaca_path, reuters_path, output_path):
        """
        1. Load Alpaca & Reuters
        2. Align Reuters schema
        3. Concat
        4. Fix null dates caused by end-of-month rollover
        """
        import os
        if not os.path.exists(alpaca_path):
            return None
        news_df = pl.read_parquet(alpaca_path)

        pdf = pd.read_excel(reuters_path)
        reuters_df = pl.from_pandas(pdf)

        reuters_df = reuters_df.rename({
            'Title': 'title', 'datetime_utc': 'datetime', 'Content': 'content', 'Stock_Type': 'equity'
        })
        reuters_df = reuters_df.with_columns([
            pl.lit(None, dtype=pl.String).alias("author"),
            pl.lit(None, dtype=pl.String).alias("source"),
            pl.lit(None, dtype=pl.String).alias("summary"),
            pl.lit(None, dtype=pl.String).alias("url")
        ])

        reuters_df = reuters_df.with_columns(
            pl.col("datetime").cast(pl.Datetime(time_unit='us')).alias("datetime"),
            pl.col("datetime").cast(pl.Date).alias("date")
        )
        reuters_df = reuters_df.select(news_df.columns)

        combined = pl.concat([news_df, reuters_df]).sort("datetime")

        # Recalculate 'date' to fix nulls for end-of-month rollovers.
        # Report Section 4.2: articles published after 16:00 ET are assigned
        # to the subsequent valid trading session.
        combined = combined.with_columns(
            pl.when(
                (pl.col("datetime").dt.hour() >= 16) & ((pl.col("datetime").dt.minute() > 0) | (pl.col("datetime").dt.second() > 0))
            )
            .then(pl.col("datetime").dt.offset_by("1d").cast(pl.Date))
            .otherwise(pl.col("datetime").cast(pl.Date))
            .alias("date")
        )

        combined.write_parquet(output_path)
        return combined.to_pandas()

    def align_to_trading_days(self, news_pd_df, trading_dates):
        """Snap article dates to the next valid trading session."""
        def build_map(tr_dates, start, end):
            tr_dates = pd.to_datetime(sorted(pd.to_datetime(tr_dates).normalize()))
            cal_dates = pd.date_range(start=start, end=end, freq='D')
            mapping = {}
            idx = 0
            for d in tqdm(cal_dates, desc="Building map"):
                while idx < len(tr_dates) and tr_dates[idx] < d:
                    idx += 1
                if idx < len(tr_dates):
                    mapping[d] = tr_dates[idx]
                else:
                    mapping[d] = tr_dates[-1]
            return mapping

        news_pd_df['date'] = pd.to_datetime(news_pd_df['date']).dt.normalize()
        tr_dates_dt = pd.to_datetime(trading_dates).normalize()

        processed = news_pd_df[news_pd_df['date'].isin(tr_dates_dt)]
        raw = news_pd_df[~news_pd_df['date'].isin(tr_dates_dt)]

        if not raw.empty:
            start_d = news_pd_df['date'].min()
            end_d = news_pd_df['date'].max()
            date_map = build_map(trading_dates, start_d, end_d)

            tqdm.pandas(desc="Adjusting dates")
            raw['date'] = raw['date'].progress_map(lambda x: date_map.get(x, x))

        final = pd.concat([processed, raw]).sort_values(by='date').reset_index(drop=True)
        return final
