import pandas as pd
import pickle
from configs.config import GlobalConfig as Config
import os
import json


class DatasetBuilder:
    """
    Synchronizes price, macro, and precomputed news-embedding streams into
    one unified, date-keyed dataset consumed by src/data_loader.py.
    """

    def create_synchronized_data(self, price_macro_dict, news_df, embedding_path):
        """
        Final union logic. Stores the full news object (title, content,
        summary) alongside the precomputed embedding for each (date, ticker).
        """
        embedding_data = {}
        if embedding_path and os.path.exists(embedding_path):
            print(f"Loading embeddings from {embedding_path}...")
            with open(embedding_path, 'r') as f:
                raw_embed_data = json.load(f)
            # Standardize key to YYYY-MM-DD
            for k, v in raw_embed_data.items():
                clean_key = str(k)[:10]
                embedding_data[clean_key] = v

        synchronized_data = {}
        mapping = Config.TICKER_MAPPING

        for date_obj, data in price_macro_dict.items():
            date_dt = pd.to_datetime(date_obj).normalize()
            date_str = str(date_obj)
            synchronized_data[date_obj] = {}

            # 1. Price (align ticker names)
            synchronized_data[date_obj]['price'] = {}
            for t, v in data.items():
                if t != 'macro' and t in mapping:
                    synchronized_data[date_obj]['price'][mapping[t]] = v

            # 2. Macro
            synchronized_data[date_obj]['macro'] = data.get('macro', {})

            # 3. News (raw articles, kept for reference/auditing)
            date_news = news_df[news_df['date'].dt.normalize() == date_dt]
            synchronized_data[date_obj]['news'] = {}

            for ticker in date_news['equity'].unique():
                if ticker in mapping:
                    clean_ticker = mapping[ticker]
                    news_records = date_news[date_news['equity'] == ticker][
                        ['title', 'content', 'summary', 'source', 'url']
                    ].to_dict(orient='records')

                    if clean_ticker not in synchronized_data[date_obj]['news']:
                        synchronized_data[date_obj]['news'][clean_ticker] = []

                    synchronized_data[date_obj]['news'][clean_ticker].extend(news_records)

            # 4. News embeddings (precomputed upstream; see main_test.py)
            synchronized_data[date_obj]['news_embedding'] = {}

            if date_str in embedding_data:
                for rec in embedding_data[date_str]:
                    raw_ticker = rec['equity']
                    if raw_ticker in mapping:
                        clean_ticker = mapping[raw_ticker]
                        synchronized_data[date_obj]['news_embedding'][clean_ticker] = rec['embedding']

        return synchronized_data

    def save(self, data, filename='unified_dataset.pkl'):
        os.makedirs(Config.PROCESSED_PATH, exist_ok=True)
        path = os.path.join(Config.PROCESSED_PATH, filename)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {path}")
