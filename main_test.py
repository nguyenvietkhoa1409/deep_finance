# import os
# import pandas as pd
# from configs.config import GlobalConfig
# from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
# from data_pipeline.processors.price_processor import PriceProcessor
# from data_pipeline.processors.macro_processor import MacroProcessor
# from data_pipeline.processors.news_processor import NewsProcessor, NewsEmbedder
# from data_pipeline.builder import DatasetBuilder

# def run_test_pipeline_skipping_news_fetch():
#     print("🚀 STARTING TEST PIPELINE (Skipping News Fetching)...")

#     # --- SETUP PATHS ---
#     # Đường dẫn đến file kết quả cũ bạn đã có
#     # LƯU Ý: Đảm bảo file này nằm đúng vị trí hoặc bạn sửa đường dẫn tại đây
#     EXISTING_NEWS_PATH = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    
#     if not os.path.exists(EXISTING_NEWS_PATH):
#         print(f"❌ ERROR: Không tìm thấy file tại {EXISTING_NEWS_PATH}")
#         print("   Vui lòng copy file 'concatenated_news_filtered.parquet' vào thư mục 'data/interim/' hoặc cập nhật đường dẫn.")
#         return

#     # ==============================================================================
#     # 1. Fetching Phase (Chỉ lấy Price & Macro, BỎ QUA News Fetcher)
#     # ==============================================================================
#     print("\n--- Phase A: Fetching (Price & Macro only) ---")
#     yahoo = YahooFetcher()
    
#     # Tạo thư mục nếu chưa có
#     os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
#     os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
#     os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

#     # 1.1 Lấy dữ liệu giá (Làm xương sống cho trục thời gian)
#     print(f"   Downloading Price Data ({GlobalConfig.START_DATE} to {GlobalConfig.END_DATE})...")
#     raw_price_list = yahoo.download_data(GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.TICKERS)
    
#     # 1.2 Lấy dữ liệu Vĩ mô
#     print("   Downloading Macro Indicators...")
#     raw_macro = yahoo.fetch_macro_indicators(GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.MACRO_SYMBOLS)

#     # ==============================================================================
#     # 2. Processing Phase (Load News có sẵn -> Align -> Embed)
#     # ==============================================================================
#     print("\n--- Phase B: Processing ---")
#     price_proc = PriceProcessor()
#     macro_proc = MacroProcessor()
#     news_proc = NewsProcessor()

#     # 2.1 Xử lý Price & Macro
#     print("   Processing Price & Macro...")
#     price_dict = price_proc.combine_to_nested_dict(raw_price_list, GlobalConfig.TICKERS)
#     processed_price_macro = macro_proc.process_and_enrich(price_dict, raw_macro)
    
#     # Lấy danh sách ngày giao dịch chuẩn (Trading Days Backbone)
#     trading_dates = list(processed_price_macro.keys())
#     print(f"   Detected {len(trading_dates)} trading days.")

#     # 2.2 Load Existing News
#     print(f"   📥 Loading existing news from: {EXISTING_NEWS_PATH}")
#     try:
#         processed_news = pd.read_parquet(EXISTING_NEWS_PATH)
#         print(f"   Loaded {len(processed_news)} news records.")
        
#         # [ANNOTATION 1] Kiểm tra Schema
#         required_cols = ['date', 'equity', 'title'] # Các cột bắt buộc cho bước sau
#         missing_cols = [c for c in required_cols if c not in processed_news.columns]
#         if missing_cols:
#             print(f"   ⚠️ WARNING: File parquet thiếu các cột: {missing_cols}")
#             print("   Logic cũ có thể dùng tên khác (ví dụ: 'headline' thay vì 'title'). Đang thử tự động sửa...")
#             if 'headline' in processed_news.columns and 'title' not in processed_news.columns:
#                 processed_news = processed_news.rename(columns={'headline': 'title'})
#                 print("   ✅ Renamed 'headline' -> 'title'.")
            
#             # Kiểm tra lại cột date phải là datetime
#             if not pd.api.types.is_datetime64_any_dtype(processed_news['date']):
#                  processed_news['date'] = pd.to_datetime(processed_news['date']).dt.date
    
#     except Exception as e:
#         print(f"❌ ERROR reading parquet file: {e}")
#         return

#     # 2.3 Align News to Trading Days
#     # Bước này vẫn CẦN THIẾT vì Price Data mới tải về có thể có ngày nghỉ lễ khác hoặc range khác
#     print("   Aligning news to current Trading Days...")
#     aligned_news = news_proc.align_to_trading_days(processed_news, trading_dates)
#     print(f"   News after alignment: {len(aligned_news)} records.")

#     # ==============================================================================
#     # 3. Embedding Phase (Chạy Embedding trên dữ liệu đã load)
#     # ==============================================================================
#     print("\n--- Phase B.1: Embedding News ---")
    
#     # [ANNOTATION 2] Kiểm tra file embedding cũ
#     embedder = NewsEmbedder()
#     embedding_output_file = os.path.join(GlobalConfig.NEWS_EMBEDDING_OUTPUT_PATH, "embedded_news.json")
    
#     if os.path.exists(embedding_output_file):
#         print(f"   ⚠️ File embedding đã tồn tại: {embedding_output_file}")
#         user_input = input("   Bạn có muốn chạy lại Embedding (tốn tiền/thời gian) không? (y/n): ")
#         if user_input.lower() == 'y':
#             embedding_json_path = embedder.process_and_save(aligned_news)
#         else:
#             print("   Skipping Embedding calculation. Using existing file.")
#             embedding_json_path = embedding_output_file
#     else:
#         # Nếu chưa có file thì chạy mới
#         embedding_json_path = embedder.process_and_save(aligned_news)

#     # ==============================================================================
#     # 4. Building Phase (Tạo file Union cuối cùng)
#     # ==============================================================================
#     print("\n--- Phase C: Building Union File ---")
#     builder = DatasetBuilder()
    
#     # Giả định file Filings đã có (hoặc bỏ qua nếu chưa cần test filings)
#     filing_path = os.path.join(GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet")
#     if not os.path.exists(filing_path):
#         print(f"   ⚠️ Warning: Filing file not found at {filing_path}. Creating dataset without filings.")
#         # Tạo dummy empty dataframe để code không lỗi
#         pd.DataFrame(columns=['filedAt', 'ticker', 'formType', 'content_summary']).to_parquet("dummy_filings.parquet")
#         filing_path = "dummy_filings.parquet"

#     dataset = builder.create_synchronized_data(
#         processed_price_macro, 
#         aligned_news, 
#         filing_path,
#         embedding_path=embedding_json_path
#     )
    
#     builder.save(dataset, filename='unified_dataset_test.pkl')
    
#     # Xóa file dummy nếu có
#     if os.path.exists("dummy_filings.parquet"): os.remove("dummy_filings.parquet")
    
#     print("\n✅ TEST PIPELINE COMPLETED SUCCESSFULLY!")

# if __name__ == "__main__":
#     run_test_pipeline_skipping_news_fetch()

"""
data_quality_explorer.py

Comprehensive data quality analysis for Deep Finance project.
Explores ALL aspects of data before making architectural decisions.

Usage:
    python data_quality_explorer.py --output ./data_analysis_report/
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try import PyG
try:
    from torch_geometric.data import Data, HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ PyG not found - KG analysis will be limited")

from configs.config import GlobalConfig, TrainConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

class ExplorerConfig:
    """Configuration for data exploration"""
    # Paths
    PICKLE_PATH = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    KG_PATH = os.path.join(GlobalConfig.KG_CACHE_DIR, 'hetero_kg_graphs.pkl')
    NEWS_PATH = os.path.join(GlobalConfig.RAW_NEWS_PATH, "03_primary/news.parquet")
    
    # Output
    OUTPUT_DIR = "./data_analysis_report"
    
    # Analysis params
    TICKERS = GlobalConfig.TICKERS
    WINDOW_SIZE = TrainConfig.window_size


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path):
    """Create directory if not exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_figure(fig, name, output_dir):
    """Save matplotlib figure"""
    fig.savefig(os.path.join(output_dir, f"{name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)


# ============================================================================
# 1. TEMPORAL COVERAGE ANALYSIS
# ============================================================================

class TemporalCoverageAnalyzer:
    """Analyze date ranges and alignment across data sources"""
    
    def __init__(self, pickle_path, kg_path, news_path):
        self.pickle_path = pickle_path
        self.kg_path = kg_path
        self.news_path = news_path
        self.results = {}
    
    def analyze(self):
        """Run full temporal analysis"""
        print_section("1. TEMPORAL COVERAGE ANALYSIS")
        
        # Load data
        print("📂 Loading data sources...")
        
        # Main pickle
        try:
            with open(self.pickle_path, 'rb') as f:
                main_data = pickle.load(f)
            print(f"   ✓ Main pickle: {len(main_data)} dates")
            self.results['main_dates'] = sorted(main_data.keys())
        except Exception as e:
            print(f"   ❌ Main pickle failed: {e}")
            return None
        
        # KG graphs
        if os.path.exists(self.kg_path) and HAS_PYG:
            try:
                with open(self.kg_path, 'rb') as f:
                    kg_data = pickle.load(f)
                print(f"   ✓ KG graphs: {len(kg_data)} dates")
                self.results['kg_dates'] = sorted(kg_data.keys())
            except Exception as e:
                print(f"   ⚠️  KG graphs failed: {e}")
                self.results['kg_dates'] = []
        else:
            print(f"   ⚠️  KG graphs not found")
            self.results['kg_dates'] = []
        
        # News
        if os.path.exists(self.news_path):
            try:
                news_df = pd.read_parquet(self.news_path)
                news_df['date'] = pd.to_datetime(news_df['date']).dt.date
                unique_dates = sorted(news_df['date'].unique())
                print(f"   ✓ News: {len(unique_dates)} dates, {len(news_df)} articles")
                self.results['news_dates'] = unique_dates
            except Exception as e:
                print(f"   ⚠️  News failed: {e}")
                self.results['news_dates'] = []
        else:
            print(f"   ⚠️  News not found")
            self.results['news_dates'] = []
        
        # Analyze coverage
        self._analyze_coverage()
        self._analyze_gaps()
        self._analyze_alignment()
        
        return self.results
    
    def _analyze_coverage(self):
        """Analyze date range coverage"""
        print("\n📅 Date Range Coverage:")
        
        for source in ['main', 'kg', 'news']:
            dates_key = f'{source}_dates'
            if not self.results.get(dates_key):
                continue
            
            dates = self.results[dates_key]
            start = min(dates)
            end = max(dates)
            span_days = (end - start).days if hasattr(start, 'days') else (pd.to_datetime(end) - pd.to_datetime(start)).days
            
            print(f"\n   {source.upper()}:")
            print(f"      Start: {start}")
            print(f"      End:   {end}")
            print(f"      Span:  {span_days} days")
            print(f"      Count: {len(dates)} dates")
            
            # Expected trading days (rough estimate: 252 trading days/year)
            expected_days = span_days * (252 / 365)
            coverage_pct = len(dates) / expected_days * 100 if expected_days > 0 else 0
            print(f"      Coverage: {coverage_pct:.1f}%")
    
    def _analyze_gaps(self):
        """Find gaps in date sequences"""
        print("\n🕳️  Gap Analysis:")
        
        for source in ['main', 'kg', 'news']:
            dates_key = f'{source}_dates'
            if not self.results.get(dates_key):
                continue
            
            dates = [pd.to_datetime(d) for d in self.results[dates_key]]
            dates = sorted(dates)
            
            gaps = []
            for i in range(len(dates) - 1):
                diff = (dates[i+1] - dates[i]).days
                if diff > 7:  # Gap > 1 week
                    gaps.append({
                        'start': dates[i],
                        'end': dates[i+1],
                        'days': diff
                    })
            
            if gaps:
                print(f"\n   {source.upper()}: {len(gaps)} significant gaps")
                for gap in gaps[:5]:  # Show first 5
                    print(f"      {gap['start'].date()} → {gap['end'].date()} ({gap['days']} days)")
                if len(gaps) > 5:
                    print(f"      ... and {len(gaps) - 5} more gaps")
            else:
                print(f"\n   {source.upper()}: No significant gaps ✓")
    
    def _analyze_alignment(self):
        """Check alignment between data sources"""
        print("\n🔗 Data Source Alignment:")
        
        main_dates = set(self.results.get('main_dates', []))
        kg_dates = set(self.results.get('kg_dates', []))
        news_dates = set(self.results.get('news_dates', []))
        
        # Normalize to same type (datetime.date)
        main_dates = {pd.to_datetime(d).date() for d in main_dates}
        kg_dates = {pd.to_datetime(d).date() if not isinstance(d, datetime) else d.date() for d in kg_dates}
        
        # Intersection
        main_kg = main_dates & kg_dates
        main_news = main_dates & news_dates
        all_three = main_dates & kg_dates & news_dates
        
        print(f"\n   Main ∩ KG:   {len(main_kg)}/{len(main_dates)} ({len(main_kg)/len(main_dates)*100:.1f}%)")
        print(f"   Main ∩ News: {len(main_news)}/{len(main_dates)} ({len(main_news)/len(main_dates)*100:.1f}%)")
        print(f"   All Three:   {len(all_three)}/{len(main_dates)} ({len(all_three)/len(main_dates)*100:.1f}%)")
        
        # Missing dates
        main_no_kg = main_dates - kg_dates
        main_no_news = main_dates - news_dates
        
        if main_no_kg:
            print(f"\n   ⚠️  {len(main_no_kg)} dates in Main but NOT in KG")
            print(f"      Sample: {sorted(main_no_kg)[:5]}")
        
        if main_no_news:
            print(f"\n   ⚠️  {len(main_no_news)} dates in Main but NOT in News")
            print(f"      Sample: {sorted(main_no_news)[:5]}")
        
        self.results['alignment'] = {
            'main_kg_overlap': len(main_kg) / len(main_dates) if main_dates else 0,
            'main_news_overlap': len(main_news) / len(main_dates) if main_dates else 0,
            'all_three_overlap': len(all_three) / len(main_dates) if main_dates else 0
        }
    
    def visualize(self, output_dir):
        """Create visualization"""
        ensure_dir(output_dir)
        
        # Timeline plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sources = []
        if self.results.get('main_dates'):
            main_dates = [pd.to_datetime(d) for d in self.results['main_dates']]
            ax.scatter(main_dates, [0]*len(main_dates), alpha=0.3, s=10, label='Main Data')
            sources.append('Main')
        
        if self.results.get('kg_dates'):
            kg_dates = [pd.to_datetime(d) for d in self.results['kg_dates']]
            ax.scatter(kg_dates, [1]*len(kg_dates), alpha=0.3, s=10, label='KG Graphs')
            sources.append('KG')
        
        if self.results.get('news_dates'):
            news_dates = [pd.to_datetime(d) for d in self.results['news_dates']]
            ax.scatter(news_dates, [2]*len(news_dates), alpha=0.3, s=10, label='News')
            sources.append('News')
        
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel('Date')
        ax.set_title('Temporal Coverage Across Data Sources')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_figure(fig, '01_temporal_coverage', output_dir)


# ============================================================================
# 2. KG QUALITY ANALYSIS
# ============================================================================

class KGQualityAnalyzer:
    """Analyze Knowledge Graph quality and statistics"""
    
    def __init__(self, kg_path, tickers):
        self.kg_path = kg_path
        self.tickers = tickers
        self.results = {}
    
    def analyze(self):
        """Run KG analysis"""
        print_section("2. KNOWLEDGE GRAPH QUALITY ANALYSIS")
        
        if not os.path.exists(self.kg_path) or not HAS_PYG:
            print("⚠️  KG data not available for analysis")
            return None
        
        # Load KG data
        print("📂 Loading KG graphs...")
        try:
            with open(self.kg_path, 'rb') as f:
                kg_data = pickle.load(f)
            print(f"   ✓ Loaded {len(kg_data)} dates")
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            return None
        
        # Analyze structure
        self._analyze_graph_structure(kg_data)
        self._analyze_event_distribution(kg_data)
        self._analyze_embeddings(kg_data)
        self._analyze_empty_patterns(kg_data)
        
        return self.results
    
    def _analyze_graph_structure(self, kg_data):
        """Analyze graph structure statistics"""
        print("\n🕸️  Graph Structure:")
        
        stats = {
            'total_graphs': 0,
            'empty_graphs': 0,
            'num_events': [],
            'num_edges': [],
            'ticker_dim': None,
            'event_dim': None
        }
        
        for date, ticker_graphs in kg_data.items():
            for ticker, graph in ticker_graphs.items():
                if ticker not in self.tickers:
                    continue
                
                stats['total_graphs'] += 1
                
                if isinstance(graph, HeteroData):
                    # Check if empty
                    num_events = graph['event'].x.size(0)
                    num_edges = graph['event', 'affects', 'ticker'].edge_index.size(1)
                    
                    if num_events == 0:
                        stats['empty_graphs'] += 1
                    
                    stats['num_events'].append(num_events)
                    stats['num_edges'].append(num_edges)
                    
                    # Dimensions
                    if stats['ticker_dim'] is None:
                        stats['ticker_dim'] = graph['ticker'].x.size(1)
                        stats['event_dim'] = graph['event'].x.size(1) if num_events > 0 else 2061
        
        # Report
        print(f"\n   Total graphs: {stats['total_graphs']}")
        print(f"   Empty graphs: {stats['empty_graphs']} ({stats['empty_graphs']/stats['total_graphs']*100:.1f}%)")
        print(f"   Non-empty:    {stats['total_graphs'] - stats['empty_graphs']}")
        
        if stats['num_events']:
            events_arr = np.array(stats['num_events'])
            print(f"\n   Events per graph:")
            print(f"      Mean:   {events_arr.mean():.2f}")
            print(f"      Median: {np.median(events_arr):.2f}")
            print(f"      Max:    {events_arr.max()}")
            print(f"      Min:    {events_arr.min()}")
            print(f"      Std:    {events_arr.std():.2f}")
        
        print(f"\n   Dimensions:")
        print(f"      Ticker node: {stats['ticker_dim']}")
        print(f"      Event node:  {stats['event_dim']}")
        
        self.results['structure'] = stats
    
    def _analyze_event_distribution(self, kg_data):
        """Analyze event distribution across tickers and time"""
        print("\n📊 Event Distribution:")
        
        events_by_ticker = defaultdict(list)
        events_by_date = defaultdict(int)
        
        for date, ticker_graphs in kg_data.items():
            for ticker, graph in ticker_graphs.items():
                if ticker not in self.tickers:
                    continue
                
                if isinstance(graph, HeteroData):
                    num_events = graph['event'].x.size(0)
                    events_by_ticker[ticker].append(num_events)
                    events_by_date[date] += num_events
        
        # By ticker
        print("\n   By Ticker:")
        for ticker in self.tickers:
            if ticker in events_by_ticker:
                events = np.array(events_by_ticker[ticker])
                print(f"      {ticker}: Mean={events.mean():.2f}, "
                      f"Median={np.median(events):.0f}, "
                      f"Max={events.max()}, "
                      f"Empty={np.sum(events == 0)}/{len(events)}")
        
        # Temporal trends
        if events_by_date:
            dates_sorted = sorted(events_by_date.keys())
            events_ts = [events_by_date[d] for d in dates_sorted]
            
            print(f"\n   Temporal Trend:")
            print(f"      First 10 days avg: {np.mean(events_ts[:10]):.1f} events/day")
            print(f"      Last 10 days avg:  {np.mean(events_ts[-10:]):.1f} events/day")
            
            # Check if declining
            if np.mean(events_ts[-10:]) < np.mean(events_ts[:10]) * 0.5:
                print("      ⚠️  WARNING: Event count declining over time!")
        
        self.results['events_by_ticker'] = events_by_ticker
        self.results['events_by_date'] = events_by_date
    
    def _analyze_embeddings(self, kg_data):
        """Analyze embedding quality"""
        print("\n🎯 Embedding Quality:")
        
        ticker_embeddings = []
        event_embeddings = []
        
        sample_count = 0
        max_samples = 100  # Sample limit
        
        for date, ticker_graphs in kg_data.items():
            if sample_count >= max_samples:
                break
            
            for ticker, graph in ticker_graphs.items():
                if ticker not in self.tickers or sample_count >= max_samples:
                    break
                
                if isinstance(graph, HeteroData):
                    # Ticker embedding
                    ticker_emb = graph['ticker'].x[0].cpu().numpy()
                    ticker_embeddings.append(ticker_emb)
                    
                    # Event embeddings (if any)
                    if graph['event'].x.size(0) > 0:
                        event_emb = graph['event'].x.cpu().numpy()
                        event_embeddings.append(event_emb)
                    
                    sample_count += 1
        
        if ticker_embeddings:
            ticker_embeddings = np.stack(ticker_embeddings)
            
            print(f"\n   Ticker Embeddings (n={len(ticker_embeddings)}):")
            print(f"      Mean:  {ticker_embeddings.mean():.4f}")
            print(f"      Std:   {ticker_embeddings.std():.4f}")
            print(f"      Min:   {ticker_embeddings.min():.4f}")
            print(f"      Max:   {ticker_embeddings.max():.4f}")
            
            # Check for degenerate embeddings
            if ticker_embeddings.std() < 0.01:
                print("      ⚠️  WARNING: Very low std - embeddings may be degenerate!")
            
            # Check for zeros
            zero_ratio = np.mean(ticker_embeddings == 0)
            print(f"      Zero ratio: {zero_ratio*100:.1f}%")
            if zero_ratio > 0.5:
                print("      ⚠️  WARNING: >50% zeros - embeddings may not be properly initialized!")
        
        if event_embeddings:
            event_embeddings = np.concatenate(event_embeddings, axis=0)
            
            print(f"\n   Event Embeddings (n={len(event_embeddings)}):")
            print(f"      Mean:  {event_embeddings.mean():.4f}")
            print(f"      Std:   {event_embeddings.std():.4f}")
            print(f"      Min:   {event_embeddings.min():.4f}")
            print(f"      Max:   {event_embeddings.max():.4f}")
            
            zero_ratio = np.mean(event_embeddings == 0)
            print(f"      Zero ratio: {zero_ratio*100:.1f}%")
    
    def _analyze_empty_patterns(self, kg_data):
        """Analyze patterns in empty graphs"""
        print("\n🕳️  Empty Graph Patterns:")
        
        empty_by_ticker = Counter()
        empty_by_date = []
        
        for date, ticker_graphs in kg_data.items():
            date_empty_count = 0
            for ticker, graph in ticker_graphs.items():
                if ticker not in self.tickers:
                    continue
                
                if isinstance(graph, HeteroData):
                    if graph['event'].x.size(0) == 0:
                        empty_by_ticker[ticker] += 1
                        date_empty_count += 1
            
            empty_by_date.append({
                'date': date,
                'empty_count': date_empty_count
            })
        
        print("\n   Empty Graphs by Ticker:")
        total_graphs_per_ticker = len(kg_data)
        for ticker in self.tickers:
            count = empty_by_ticker[ticker]
            pct = count / total_graphs_per_ticker * 100
            print(f"      {ticker}: {count}/{total_graphs_per_ticker} ({pct:.1f}%)")
        
        # Check temporal pattern
        empty_by_date_df = pd.DataFrame(empty_by_date)
        if len(empty_by_date_df) > 0:
            print(f"\n   Temporal Pattern:")
            print(f"      First 10 dates: {empty_by_date_df.head(10)['empty_count'].mean():.1f} empty/date")
            print(f"      Last 10 dates:  {empty_by_date_df.tail(10)['empty_count'].mean():.1f} empty/date")
    
    def visualize(self, output_dir):
        """Create visualizations"""
        ensure_dir(output_dir)
        
        # Events distribution
        if 'events_by_ticker' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histogram per ticker
            for idx, ticker in enumerate(self.tickers):
                ax = axes[idx // 2, idx % 2]
                events = self.results['events_by_ticker'][ticker]
                ax.hist(events, bins=20, edgecolor='black', alpha=0.7)
                ax.set_title(f'{ticker}: Events per Graph')
                ax.set_xlabel('Number of Events')
                ax.set_ylabel('Frequency')
                ax.axvline(np.mean(events), color='red', linestyle='--', label=f'Mean: {np.mean(events):.1f}')
                ax.legend()
            
            plt.tight_layout()
            save_figure(fig, '02_kg_events_distribution', output_dir)
        
        # Temporal trend
        if 'events_by_date' in self.results:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            dates = sorted(self.results['events_by_date'].keys())
            events = [self.results['events_by_date'][d] for d in dates]
            
            dates_dt = [pd.to_datetime(d) for d in dates]
            ax.plot(dates_dt, events, marker='.', alpha=0.6)
            ax.set_xlabel('Date')
            ax.set_ylabel('Total Events Across All Tickers')
            ax.set_title('KG Events Over Time')
            ax.grid(True, alpha=0.3)
            
            save_figure(fig, '02_kg_temporal_trend', output_dir)


# ============================================================================
# 3. NEWS COVERAGE ANALYSIS
# ============================================================================

class NewsCoverageAnalyzer:
    """Analyze news coverage and quality"""
    
    def __init__(self, pickle_path, news_path, tickers):
        self.pickle_path = pickle_path
        self.news_path = news_path
        self.tickers = tickers
        self.results = {}
    
    def analyze(self):
        """Run news analysis"""
        print_section("3. NEWS COVERAGE ANALYSIS")
        
        # Load news
        if not os.path.exists(self.news_path):
            print("⚠️  News file not found")
            return None
        
        print("📂 Loading news data...")
        try:
            news_df = pd.read_parquet(self.news_path)
            print(f"   ✓ Loaded {len(news_df)} articles")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return None
        
        # Load main data for embedding check
        try:
            with open(self.pickle_path, 'rb') as f:
                main_data = pickle.load(f)
        except Exception as e:
            print(f"   ⚠️  Cannot load main data: {e}")
            main_data = None
        
        self._analyze_coverage(news_df)
        self._analyze_content_quality(news_df)
        
        if main_data:
            self._analyze_embeddings(main_data)
        
        return self.results
    
    def _analyze_coverage(self, news_df):
        """Analyze news coverage per ticker"""
        print("\n📰 News Coverage:")
        
        # Normalize date
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        
        # Map equity names
        equity_mapping = {v: k for k, aliases in GlobalConfig.TICKER_MAPPING.items() for v in aliases}
        
        coverage = {}
        for ticker in self.tickers:
            # Get all aliases for this ticker
            aliases = [ticker]  # Add more if needed
            ticker_news = news_df[news_df['equity'].isin(aliases)]
            
            coverage[ticker] = {
                'total_articles': len(ticker_news),
                'unique_dates': ticker_news['date'].nunique(),
                'avg_per_day': len(ticker_news) / max(ticker_news['date'].nunique(), 1)
            }
        
        print("\n   By Ticker:")
        for ticker, stats in coverage.items():
            print(f"      {ticker}:")
            print(f"         Articles: {stats['total_articles']}")
            print(f"         Days:     {stats['unique_dates']}")
            print(f"         Avg/Day:  {stats['avg_per_day']:.2f}")
        
        self.results['coverage'] = coverage
    
    def _analyze_content_quality(self, news_df):
        """Analyze news content quality"""
        print("\n📝 Content Quality:")
        
        # Check for nulls
        null_counts = news_df[['title', 'content', 'summary']].isnull().sum()
        print(f"\n   Missing Values:")
        for col, count in null_counts.items():
            pct = count / len(news_df) * 100
            print(f"      {col}: {count} ({pct:.1f}%)")
        
        # Content length
        if 'content' in news_df.columns:
            news_df['content_len'] = news_df['content'].fillna('').str.len()
            print(f"\n   Content Length:")
            print(f"      Mean:   {news_df['content_len'].mean():.0f} chars")
            print(f"      Median: {news_df['content_len'].median():.0f} chars")
            print(f"      Min:    {news_df['content_len'].min():.0f} chars")
            print(f"      Max:    {news_df['content_len'].max():.0f} chars")
            
            # Check for suspiciously short content
            short_content = (news_df['content_len'] < 100).sum()
            print(f"      <100 chars: {short_content} ({short_content/len(news_df)*100:.1f}%)")
        
        # Title length
        if 'title' in news_df.columns:
            news_df['title_len'] = news_df['title'].fillna('').str.len()
            print(f"\n   Title Length:")
            print(f"      Mean:   {news_df['title_len'].mean():.0f} chars")
            print(f"      Median: {news_df['title_len'].median():.0f} chars")
    
    def _analyze_embeddings(self, main_data):
        """Analyze news embedding quality"""
        print("\n🎯 News Embeddings:")
        
        embeddings = []
        missing_count = 0
        
        for date, content in main_data.items():
            news_emb = content.get('news_embedding', {})
            
            for ticker in self.tickers:
                if ticker in news_emb:
                    emb = news_emb[ticker]
                    if emb is not None:
                        embeddings.append(np.array(emb))
                    else:
                        missing_count += 1
                else:
                    missing_count += 1
        
        if embeddings:
            embeddings = np.stack(embeddings)
            
            print(f"\n   Embedding Statistics (n={len(embeddings)}):")
            print(f"      Shape:     {embeddings.shape}")
            print(f"      Mean:      {embeddings.mean():.4f}")
            print(f"      Std:       {embeddings.std():.4f}")
            print(f"      Min:       {embeddings.min():.4f}")
            print(f"      Max:       {embeddings.max():.4f}")
            
            # Check for issues
            zero_ratio = np.mean(embeddings == 0)
            print(f"      Zero ratio: {zero_ratio*100:.1f}%")
            
            if zero_ratio > 0.3:
                print("      ⚠️  WARNING: High zero ratio - embeddings may be sparse!")
            
            if embeddings.std() < 0.01:
                print("      ⚠️  WARNING: Very low std - embeddings may be degenerate!")
        
        print(f"\n   Missing embeddings: {missing_count}")


# ============================================================================
# 4. LABEL ANALYSIS
# ============================================================================

class LabelAnalyzer:
    """Analyze label distribution and quality"""
    
    def __init__(self, pickle_path, tickers):
        self.pickle_path = pickle_path
        self.tickers = tickers
        self.results = {}
    
    def analyze(self):
        """Run label analysis"""
        print_section("4. LABEL DISTRIBUTION ANALYSIS")
        
        # Import data_prepare to generate labels
        from src.data_loader import data_prepare
        
        dp = data_prepare(self.pickle_path, kg_data_path=None)
        
        label_stats = {}
        
        for ticker in self.tickers:
            try:
                train, valid, test = dp.prepare_data(ticker)
                
                if not train or 'label' not in train:
                    continue
                
                # Combine all labels
                all_labels = np.concatenate([
                    train['label'].numpy(),
                    valid['label'].numpy(),
                    test['label'].numpy()
                ])
                
                label_counts = Counter(all_labels)
                
                label_stats[ticker] = {
                    'distribution': label_counts,
                    'total': len(all_labels),
                    'train_size': len(train['label']),
                    'valid_size': len(valid['label']),
                    'test_size': len(test['label'])
                }
                
            except Exception as e:
                print(f"   ⚠️  {ticker}: {e}")
        
        self._report_distribution(label_stats)
        self.results['label_stats'] = label_stats
        
        return self.results
    
    def _report_distribution(self, label_stats):
        """Report label distribution"""
        print("\n📊 Label Distribution by Ticker:")
        
        for ticker, stats in label_stats.items():
            dist = stats['distribution']
            total = stats['total']
            
            print(f"\n   {ticker} (Total: {total}):")
            print(f"      DOWN (0): {dist.get(0, 0)} ({dist.get(0, 0)/total*100:.1f}%)")
            print(f"      FLAT (1): {dist.get(1, 0)} ({dist.get(1, 0)/total*100:.1f}%)")
            print(f"      UP (2):   {dist.get(2, 0)} ({dist.get(2, 0)/total*100:.1f}%)")
            
            # Check for severe imbalance
            max_class = max(dist.values())
            min_class = min(dist.values())
            imbalance_ratio = max_class / max(min_class, 1)
            
            if imbalance_ratio > 2.0:
                print(f"      ⚠️  WARNING: Class imbalance ratio {imbalance_ratio:.1f}:1")
    
    def visualize(self, output_dir):
        """Create visualizations"""
        ensure_dir(output_dir)
        
        if 'label_stats' not in self.results:
            return
        
        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tickers = list(self.results['label_stats'].keys())
        down = [self.results['label_stats'][t]['distribution'].get(0, 0) for t in tickers]
        flat = [self.results['label_stats'][t]['distribution'].get(1, 0) for t in tickers]
        up = [self.results['label_stats'][t]['distribution'].get(2, 0) for t in tickers]
        
        x = np.arange(len(tickers))
        width = 0.6
        
        ax.bar(x, down, width, label='DOWN', color='#e74c3c')
        ax.bar(x, flat, width, bottom=down, label='FLAT', color='#95a5a6')
        ax.bar(x, up, width, bottom=np.array(down)+np.array(flat), label='UP', color='#27ae60')
        
        ax.set_ylabel('Count')
        ax.set_title('Label Distribution by Ticker')
        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.legend()
        
        save_figure(fig, '04_label_distribution', output_dir)


# ============================================================================
# 5. FEATURE CORRELATION ANALYSIS
# ============================================================================

class FeatureCorrelationAnalyzer:
    """Analyze correlations between features and labels"""
    
    def __init__(self, pickle_path, tickers):
        self.pickle_path = pickle_path
        self.tickers = tickers
        self.results = {}
    
    def analyze(self):
        """Run correlation analysis"""
        print_section("5. FEATURE CORRELATION ANALYSIS")
        
        from src.data_loader import data_prepare
        
        dp = data_prepare(self.pickle_path, kg_data_path=None)
        
        correlations = {}
        
        for ticker in self.tickers[:1]:  # Sample 1 ticker for speed
            try:
                train, _, _ = dp.prepare_data(ticker)
                
                if not train:
                    continue
                
                # Extract features
                price_close = train['s_c'][:, -1, 0].numpy()  # Last close
                macro = train['s_m'][:, -1, :].numpy()  # Last macro
                news = train['s_n'][:, -1, :].numpy()  # Last news
                labels = train['label'].numpy()
                
                # Price vs Label
                price_corr = np.corrcoef(price_close, labels)[0, 1]
                
                # Macro vs Label (per feature)
                macro_corrs = [np.corrcoef(macro[:, i], labels)[0, 1] 
                              for i in range(macro.shape[1])]
                
                # News vs Label (average)
                news_mean = news.mean(axis=1)
                news_corr = np.corrcoef(news_mean, labels)[0, 1]
                
                correlations[ticker] = {
                    'price_label': price_corr,
                    'macro_label': macro_corrs,
                    'news_label': news_corr
                }
                
                print(f"\n📈 {ticker} Correlations with Label:")
                print(f"   Price:     {price_corr:.4f}")
                print(f"   News:      {news_corr:.4f}")
                print(f"   Macro (avg): {np.mean(macro_corrs):.4f}")
                
            except Exception as e:
                print(f"   ⚠️  {ticker}: {e}")
        
        self.results['correlations'] = correlations
        return self.results


# ============================================================================
# 6. MAIN EXPLORER
# ============================================================================

class DataQualityExplorer:
    """Main explorer orchestrating all analyses"""
    
    def __init__(self, config=None):
        self.config = config or ExplorerConfig()
        self.results = {}
    
    def run_all_analyses(self):
        """Run all data quality analyses"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE DATA QUALITY EXPLORATION")
        print("="*80)
        print(f"\nOutput directory: {self.config.OUTPUT_DIR}")
        ensure_dir(self.config.OUTPUT_DIR)
        
        # 1. Temporal Coverage
        temporal_analyzer = TemporalCoverageAnalyzer(
            self.config.PICKLE_PATH,
            self.config.KG_PATH,
            self.config.NEWS_PATH
        )
        self.results['temporal'] = temporal_analyzer.analyze()
        temporal_analyzer.visualize(self.config.OUTPUT_DIR)
        
        # 2. KG Quality
        kg_analyzer = KGQualityAnalyzer(
            self.config.KG_PATH,
            self.config.TICKERS
        )
        self.results['kg'] = kg_analyzer.analyze()
        if self.results['kg']:
            kg_analyzer.visualize(self.config.OUTPUT_DIR)
        
        # 3. News Coverage
        news_analyzer = NewsCoverageAnalyzer(
            self.config.PICKLE_PATH,
            self.config.NEWS_PATH,
            self.config.TICKERS
        )
        self.results['news'] = news_analyzer.analyze()
        
        # 4. Labels
        label_analyzer = LabelAnalyzer(
            self.config.PICKLE_PATH,
            self.config.TICKERS
        )
        self.results['labels'] = label_analyzer.analyze()
        if self.results['labels']:
            label_analyzer.visualize(self.config.OUTPUT_DIR)
        
        # 5. Correlations
        corr_analyzer = FeatureCorrelationAnalyzer(
            self.config.PICKLE_PATH,
            self.config.TICKERS
        )
        self.results['correlations'] = corr_analyzer.analyze()
        
        # Generate summary report
        self._generate_summary_report()
        
        print("\n" + "="*80)
        print("✅ EXPLORATION COMPLETE")
        print(f"📁 Results saved to: {self.config.OUTPUT_DIR}")
        print("="*80 + "\n")
    
    def _generate_summary_report(self):
        """Generate summary markdown report"""
        report_path = os.path.join(self.config.OUTPUT_DIR, 'SUMMARY_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write("# Data Quality Exploration Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🚨 Key Findings\n\n")
            
            # Critical issues
            issues = []
            
            # Check temporal alignment
            if self.results.get('temporal'):
                alignment = self.results['temporal'].get('alignment', {})
                if alignment.get('main_kg_overlap', 0) < 0.8:
                    issues.append(f"⚠️  **Low KG coverage**: Only {alignment['main_kg_overlap']*100:.1f}% of dates have KG graphs")
            
            # Check empty graphs
            if self.results.get('kg'):
                structure = self.results['kg'].get('structure', {})
                empty_pct = structure.get('empty_graphs', 0) / max(structure.get('total_graphs', 1), 1) * 100
                if empty_pct > 50:
                    issues.append(f"⚠️  **High empty graph rate**: {empty_pct:.1f}% of KG graphs are empty")
            
            if issues:
                for issue in issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("✅ No critical data quality issues detected\n")
            
            f.write("\n## 📊 Detailed Statistics\n\n")
            f.write(f"See individual visualizations in `{self.config.OUTPUT_DIR}/`\n")
        
        print(f"\n📄 Summary report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Finance Data Quality Explorer')
    parser.add_argument('--output', type=str, default='./data_analysis_report',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Update config
    ExplorerConfig.OUTPUT_DIR = args.output
    
    # Run exploration
    explorer = DataQualityExplorer()
    explorer.run_all_analyses()