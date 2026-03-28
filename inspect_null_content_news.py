import pandas as pd
import numpy as np

# Define the path to the data file
file_path = "data/raw/raw_news.parquet"

def inspect_null_content_news():
    """
    Loads news data, identifies rows with null or whitespace content,
    and prints headlines for inspection.
    """
    print(f"Loading data from {file_path}...")
    try:
        # It's good practice to specify columns if we only need a few, to save memory.
        df = pd.read_parquet(file_path, columns=['headline', 'content', 'symbols'])
    except Exception as e:
        print(f"Error loading data: {e}")
        print("I couldn't find or read the file at 'data/raw/raw_news.parquet'.")
        print("This file seems to be the starting point for the news data based on my analysis of 'configs/config.py'.")
        print("Please ensure the file exists and is accessible.")
        return

    print(f"Successfully loaded {len(df)} total articles.")

    # Identify rows where 'content' is null, empty, or just whitespace
    is_null_or_empty = df['content'].isnull() | (df['content'].str.strip() == '')
    null_content_df = df[is_null_or_empty]

    num_null_content = len(null_content_df)
    total_articles = len(df)
    percentage_null = (num_null_content / total_articles) * 100 if total_articles > 0 else 0

    print(f"
Found {num_null_content} articles with null or whitespace content ({percentage_null:.2f}% of total).")

    if num_null_content > 0:
        print("
--- Inspecting headlines for the first 50 articles with null content ---")
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_colwidth', 150) # Widen display for long headlines
        
        # We want to show the symbol and the headline
        headlines_to_show = null_content_df[['symbols', 'headline']].head(50)
        
        # Use to_string() to make sure it prints well in the console
        print(headlines_to_show.to_string())
        
        print("
" + "-"*60)
        print("The table above shows the first 50 headlines from articles where")
        print("the 'content' field was missing. Please review them to assess their value.")

if __name__ == "__main__":
    inspect_null_content_news()
