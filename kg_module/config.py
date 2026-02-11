"""
Configuration for Knowledge Graph Module
"""

# ============================================
# EMBEDDINGS CONFIGURATION
# ============================================
# CHANGED: Use full Voyage output dimension
TICKER_EMBEDDING_DIM = 1024  # Full Voyage-finance-2 output
RELATION_EMBEDDING_DIM = 1024  # Full Voyage-finance-2 output

# Node feature composition:
# - Center node: [ticker_embedding (1024,) + padding (4,)] = 1028-dim
# - Event node: [relation_embedding (1024,) + magnitude + polarity + certainty + decay] = 1028-dim
NODE_FEATURE_DIM = TICKER_EMBEDDING_DIM + 4  # 1028

# ============================================
# GCN ENCODER SETTINGS (UPDATED)
# ============================================
GCN_INPUT_DIM = NODE_FEATURE_DIM  # 1028 (was 772)
GCN_HIDDEN_DIM = 512  # Scaled up (was 256)
GCN_OUTPUT_DIM = 128  # Keep same (must match TrainConfig.dim)
GCN_DROPOUT = 0.1
GCN_NUM_LAYERS = 2




# ============================================
# RELATION TYPES (No clustering needed)
# ============================================
RELATION_TYPES = {
    'revenue_change': {
        'id': 0,
        'keywords': ['revenue', 'sales', 'income', 'quarterly', 'earnings'],
        'half_life_days': 2
    },
    'guidance_update': {
        'id': 1,
        'keywords': ['guidance', 'outlook', 'forecast', 'expects', 'projects'],
        'half_life_days': 5
    },
    'acquisition': {
        'id': 2,
        'keywords': ['acquire', 'acquisition', 'purchase', 'buy', 'merger', 'M&A'],
        'half_life_days': 14
    },
    'partnership': {
        'id': 3,
        'keywords': ['partner', 'partnership', 'collaboration', 'deal', 'agreement'],
        'half_life_days': 7
    },
    'product_launch': {
        'id': 4,
        'keywords': ['launch', 'release', 'unveil', 'introduce', 'debut'],
        'half_life_days': 5
    },
    'cost_change': {
        'id': 5,
        'keywords': ['cost', 'expense', 'margin', 'efficiency', 'layoff'],
        'half_life_days': 3
    },
    'market_share': {
        'id': 6,
        'keywords': ['market share', 'competition', 'competitive', 'outperform'],
        'half_life_days': 10
    },
    'regulation': {
        'id': 7,
        'keywords': ['regulation', 'regulatory', 'lawsuit', 'legal', 'policy', 'fine'],
        'half_life_days': 14
    }
}

# ============================================
# ENTITY ALIASES (Hardcoded - Expand as needed)
# ============================================
ENTITY_ALIASES = {
    'TSLA': [
        'Tesla', 'TSLA', 'Tesla Inc', 'Tesla Motors',
        'Elon Musk company', 'Tesla Energy'
    ],
    'AMZN': [
        'Amazon', 'Amazon.com', 'AMZN', 'Amazon Inc',
        'AWS', 'Amazon Web Services', 'Amazon Prime',
        'Bezos company'
    ],
    'MSFT': [
        'Microsoft', 'MSFT', 'MS', 'Microsoft Corporation',
        'Azure', 'Windows', 'Office 365', 'Nadella company'
    ],
    'NFLX': [
        'Netflix', 'NFLX', 'Netflix Inc',
        'Reed Hastings company'
    ],
    # ADD MORE TICKERS AS NEEDED
}

# ============================================
# MAGNITUDE EXTRACTION PATTERNS
# ============================================
MAGNITUDE_PATTERNS = [
    # Percentage patterns
    r'(\d+\.?\d*)\s*%\s*(increase|decrease|growth|decline|up|down)',
    r'(increase|decrease|growth|decline|up|down)\s*(\d+\.?\d*)\s*%',
    r'(increased|decreased|grew|declined)\s*(\d+\.?\d*)\s*percent',
    
    # Dollar amounts
    r'\$(\d+\.?\d*)\s*(billion|million|B|M)',
    r'(\d+\.?\d*)\s*(billion|million)\s*dollars',
    
    # Multipliers
    r'(\d+\.?\d*)x',
    r'(\d+\.?\d*)\s*times',
]

# ============================================
# POLARITY KEYWORDS
# ============================================
POLARITY_KEYWORDS = {
    'positive': [
        'increase', 'growth', 'surge', 'soar', 'jump', 'rise', 'gain',
        'beat', 'exceed', 'outperform', 'strong', 'robust', 'record',
        'up', 'higher', 'improve', 'boost', 'accelerate'
    ],
    'negative': [
        'decrease', 'decline', 'drop', 'fall', 'plunge', 'tumble',
        'miss', 'underperform', 'weak', 'disappointing', 'loss',
        'down', 'lower', 'worsen', 'slash', 'cut', 'slow'
    ]
}

# ============================================
# CERTAINTY KEYWORDS
# ============================================
CERTAINTY_KEYWORDS = {
    1.0: ['reported', 'announced', 'confirmed', 'achieved', 'posted'],
    0.8: ['expects', 'anticipates', 'projects', 'forecasts', 'plans'],
    0.5: ['may', 'could', 'might', 'possibly', 'potential'],
    0.3: ['rumored', 'sources say', 'reportedly', 'allegedly']
}

# ============================================
# GRAPH CONSTRUCTION SETTINGS
# ============================================
MAX_EVENTS_PER_GRAPH = 20  # Limit to prevent huge graphs
MIN_CERTAINTY_THRESHOLD = 0.3  # Filter low-confidence events

# ============================================
# GCN ENCODER SETTINGS
# ============================================
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 128  # Must match TrainConfig.dim
GCN_DROPOUT = 0.1
GCN_NUM_LAYERS = 2

# ============================================
# LLM SETTINGS (for fallback extraction)
# ============================================
USE_LLM_FALLBACK = True
LLM_MODEL = "gpt-3.5-turbo"  # Or "gpt-4" if budget allows
LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 10  # seconds

# LLM Prompt Template
LLM_PROMPT_TEMPLATE = """You are a financial analyst. Extract structured information from this news headline.

News: "{headline}"
Ticker: {ticker}

Output ONLY valid JSON (no markdown, no explanation):
{{
  "relation": "one of [revenue_change, guidance_update, acquisition, partnership, product_launch, cost_change, market_share, regulation]",
  "object": "brief description (max 15 words)",
  "magnitude": <float or null>,
  "polarity": <-1, 0, or 1>,
  "certainty": <0.3, 0.5, 0.8, or 1.0>
}}

Rules:
- If no clear relation, use null for relation
- magnitude: extract percentage or dollar amount as decimal (e.g., "23%" → 0.23)
- polarity: +1 (positive), -1 (negative), 0 (neutral)
- certainty: 1.0 (confirmed fact), 0.8 (expected), 0.5 (possible), 0.3 (rumor)
"""