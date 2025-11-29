"""
GDELT Historical News Fetcher for MarketBrain

GDELT (Global Database of Events, Language, and Tone) provides:
- Free access to news data from 1979 to present
- Coverage of major global news sources
- Pre-computed tone/sentiment scores
- Entity extraction (companies, people, locations)

This module fetches GDELT data and processes it through Ollama for
stock-specific sentiment analysis.

Data sources:
1. GDELT 2.0 GKG (Global Knowledge Graph) - richest data, 2015+
2. GDELT 1.0 Events - 1979-2015
3. GDELT DOC API - Full-text search

We'll use a combination approach:
- 2015+: Use GKG for detailed sentiment with entity extraction
- 1979-2015: Use Events database + price-derived sentiment proxy
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import zipfile
import io
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GDELTArticle:
    """Represents a GDELT news article"""
    date: datetime
    url: str
    title: str
    source: str
    tone: float              # GDELT pre-computed tone (-100 to +100)
    themes: List[str]        # GDELT themes (e.g., ECON_BANKRUPTCY)
    organizations: List[str] # Extracted organizations
    persons: List[str]       # Extracted persons
    locations: List[str]     # Extracted locations
    word_count: int

    # Ollama-computed sentiment (filled in later)
    ollama_sentiment: Optional[float] = None
    ollama_confidence: Optional[float] = None


class GDELTFetcher:
    """
    Fetches historical news from GDELT for stock sentiment analysis

    GDELT Data Overview:
    - GKG files are released every 15 minutes
    - Each file contains ~20,000-50,000 records
    - Daily master files available for bulk download

    Rate limits:
    - DOC API: ~1 request/second recommended
    - Raw files: No limit (public S3 bucket)
    """

    # GDELT base URLs
    GDELT_GKG_BASE = "http://data.gdeltproject.org/gdeltv2"
    GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
    GDELT_GEO_API = "https://api.gdeltproject.org/api/v2/geo/geo"

    # Stock-related GDELT themes
    STOCK_THEMES = [
        'ECON_', 'BUS_', 'TAX_', 'TRADE_', 'MANMADE_DISASTER',
        'BANKRUPTCY', 'MERGER', 'ACQUISITION', 'IPO', 'EARNINGS',
        'SEC_', 'FRAUD', 'LAWSUIT', 'REGULATION', 'TARIFF'
    ]

    def __init__(
        self,
        output_dir: str = "data/gdelt",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Cache for company name -> ticker mapping
        self.company_ticker_map = self._load_company_mappings()

    def _load_company_mappings(self) -> Dict[str, str]:
        """Load company name to ticker symbol mappings"""
        # Common variations of company names
        mappings = {
            # Tech giants
            'apple': 'AAPL', 'apple inc': 'AAPL', 'apple computer': 'AAPL',
            'microsoft': 'MSFT', 'microsoft corp': 'MSFT', 'microsoft corporation': 'MSFT',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'alphabet inc': 'GOOGL',
            'amazon': 'AMZN', 'amazon.com': 'AMZN', 'amazon inc': 'AMZN',
            'meta': 'META', 'facebook': 'META', 'meta platforms': 'META',
            'nvidia': 'NVDA', 'nvidia corp': 'NVDA', 'nvidia corporation': 'NVDA',
            'tesla': 'TSLA', 'tesla inc': 'TSLA', 'tesla motors': 'TSLA',
            'netflix': 'NFLX', 'netflix inc': 'NFLX',
            'intel': 'INTC', 'intel corp': 'INTC', 'intel corporation': 'INTC',
            'amd': 'AMD', 'advanced micro devices': 'AMD',
            # Add more mappings as needed...
        }

        # Try to load extended mappings from file
        mapping_file = self.output_dir / "company_mappings.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                mappings.update(json.load(f))

        return mappings

    def fetch_gkg_file(self, date: datetime) -> pd.DataFrame:
        """
        Fetch GDELT GKG (Global Knowledge Graph) data for a specific date

        GKG contains:
        - Extracted themes and topics
        - Tone/sentiment scores
        - Named entities (orgs, people, locations)
        - Social media amplification data

        Note: GDELT files are released every 15 minutes with timestamps.
        We fetch multiple files to cover a full day.
        """
        date_str = date.strftime("%Y%m%d")

        # Check cache first
        cache_file = self.output_dir / f"gkg_{date_str}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        # GKG files are released every 15 minutes
        # Timestamps: 000000, 001500, 003000, 004500, etc.
        # We'll sample a few key times: market open, midday, close
        sample_times = ['140000', '160000', '180000', '200000', '220000']  # UTC times covering US market

        all_data = []

        for time_str in sample_times:
            url = f"{self.GDELT_GKG_BASE}/{date_str}{time_str}.gkg.csv.zip"

            try:
                logger.debug(f"Fetching GKG: {url}")
                response = requests.get(url, timeout=60)

                if response.status_code == 404:
                    continue

                response.raise_for_status()

                # Extract and parse CSV
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = self._parse_gkg_csv(f)
                        if not df.empty:
                            all_data.append(df)

            except Exception as e:
                logger.debug(f"Error fetching GKG {date_str}{time_str}: {e}")
                continue

        if not all_data:
            logger.warning(f"GKG data not available for {date_str}")
            return pd.DataFrame()

        # Combine all files for the day
        combined = pd.concat(all_data, ignore_index=True)

        # Deduplicate by URL (same article may appear in multiple 15-min windows)
        combined = combined.drop_duplicates(subset=['url'], keep='first')

        logger.info(f"Fetched {len(combined)} unique GKG records for {date_str}")

        # Cache the result
        if not combined.empty:
            combined.to_parquet(cache_file)

        return combined

    def _parse_gkg_csv(self, file_handle) -> pd.DataFrame:
        """Parse GDELT GKG CSV format"""
        # GKG column names (v2.1 schema)
        columns = [
            'GKGRECORDID', 'V2.1DATE', 'V2SOURCECOLLECTIONIDENTIFIER',
            'V2SOURCECOMMONNAME', 'V2DOCUMENTIDENTIFIER', 'V1COUNTS',
            'V2.1COUNTS', 'V1THEMES', 'V2ENHANCEDTHEMES', 'V1LOCATIONS',
            'V2ENHANCEDLOCATIONS', 'V1PERSONS', 'V2ENHANCEDPERSONS',
            'V1ORGANIZATIONS', 'V2ENHANCEDORGANIZATIONS', 'V1.5TONE',
            'V2.1ENHANCEDDATES', 'V2GCAM', 'V2.1SHARINGIMAGE',
            'V2.1RELATEDIMAGES', 'V2.1SOCIALIMAGEEMBEDS',
            'V2.1SOCIALVIDEOEMBEDS', 'V2.1QUOTATIONS', 'V2.1ALLNAMES',
            'V2.1AMOUNTS', 'V2.1TRANSLATIONINFO', 'V2EXTRASXML'
        ]

        try:
            df = pd.read_csv(
                file_handle,
                sep='\t',
                names=columns,
                encoding='utf-8',
                on_bad_lines='skip'
            )
        except Exception as e:
            logger.warning(f"Error parsing GKG CSV: {e}")
            return pd.DataFrame()

        # Parse key fields
        records = []
        for _, row in df.iterrows():
            try:
                # Parse tone (V1.5TONE format: tone,pos,neg,polarity,wordcount,...)
                tone_parts = str(row['V1.5TONE']).split(',')
                tone = float(tone_parts[0]) if tone_parts else 0
                word_count = int(float(tone_parts[4])) if len(tone_parts) > 4 else 0

                # Parse themes
                themes = str(row['V1THEMES']).split(';') if pd.notna(row['V1THEMES']) else []

                # Parse organizations
                orgs = str(row['V1ORGANIZATIONS']).split(';') if pd.notna(row['V1ORGANIZATIONS']) else []

                # Parse persons
                persons = str(row['V1PERSONS']).split(';') if pd.notna(row['V1PERSONS']) else []

                # Parse locations
                locations = str(row['V1LOCATIONS']).split(';') if pd.notna(row['V1LOCATIONS']) else []

                # Parse date
                date_str = str(row['V2.1DATE'])[:8]
                date = datetime.strptime(date_str, '%Y%m%d')

                records.append({
                    'date': date,
                    'url': row['V2DOCUMENTIDENTIFIER'],
                    'source': row['V2SOURCECOMMONNAME'],
                    'tone': tone,
                    'themes': themes,
                    'organizations': [o.lower() for o in orgs if o],
                    'persons': [p.lower() for p in persons if p],
                    'locations': locations,
                    'word_count': word_count
                })

            except Exception as e:
                continue

        return pd.DataFrame(records)

    def search_stock_news(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_records: int = 250
    ) -> List[Dict]:
        """
        Search GDELT DOC API for stock-related news

        This is useful for targeted searches (specific company, event, etc.)
        Limited to 250 records per query.
        """
        params = {
            'query': query,
            'mode': 'ArtList',
            'maxrecords': min(max_records, 250),
            'format': 'json',
            'startdatetime': start_date.strftime('%Y%m%d%H%M%S'),
            'enddatetime': end_date.strftime('%Y%m%d%H%M%S'),
            'sort': 'DateDesc'
        }

        try:
            response = requests.get(self.GDELT_DOC_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', [])
        except Exception as e:
            logger.error(f"DOC API search error: {e}")
            return []

    def filter_stock_relevant(
        self,
        df: pd.DataFrame,
        symbols: List[str] = None
    ) -> pd.DataFrame:
        """Filter GDELT records to stock-relevant news"""
        if df.empty:
            return df

        # Filter by stock-related themes
        def has_stock_theme(themes):
            if not themes:
                return False
            for theme in themes:
                for stock_theme in self.STOCK_THEMES:
                    if stock_theme in str(theme).upper():
                        return True
            return False

        # Filter by company mentions
        def mentions_company(orgs, symbols):
            if not orgs:
                return False
            for org in orgs:
                org_lower = org.lower().strip()
                # Check if org matches any company name
                if org_lower in self.company_ticker_map:
                    ticker = self.company_ticker_map[org_lower]
                    if symbols is None or ticker in symbols:
                        return True
            return False

        # Apply filters
        mask = df['themes'].apply(has_stock_theme)
        if symbols:
            mask = mask | df['organizations'].apply(lambda x: mentions_company(x, symbols))

        return df[mask].copy()

    def extract_stock_mentions(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract specific stock mentions from GDELT records"""
        records = []

        for _, row in df.iterrows():
            # Find which stocks are mentioned
            mentioned_tickers = set()

            for org in row['organizations']:
                org_lower = org.lower().strip()
                if org_lower in self.company_ticker_map:
                    mentioned_tickers.add(self.company_ticker_map[org_lower])

            # Create a record for each mentioned stock
            for ticker in mentioned_tickers:
                records.append({
                    'date': row['date'],
                    'ticker': ticker,
                    'url': row['url'],
                    'source': row['source'],
                    'gdelt_tone': row['tone'],
                    'themes': row['themes'],
                    'word_count': row['word_count']
                })

        return pd.DataFrame(records)

    def analyze_sentiment_ollama(
        self,
        articles: List[Dict],
        ticker: str,
        batch_size: int = 10
    ) -> List[Dict]:
        """
        Use Ollama to analyze sentiment for stock-specific articles

        This provides more accurate stock sentiment than GDELT's generic tone.
        """
        results = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]

            for article in batch:
                try:
                    # Create prompt for sentiment analysis
                    prompt = f"""Analyze this news headline's sentiment for {ticker} stock.

Headline: {article.get('title', article.get('url', 'N/A'))}
Source: {article.get('source', 'Unknown')}
GDELT Themes: {', '.join(article.get('themes', [])[:5])}

Rate the sentiment from -1.0 (very bearish) to +1.0 (very bullish).
Also rate your confidence from 0.0 to 1.0.

Respond in JSON format only:
{{"sentiment": 0.0, "confidence": 0.0, "reasoning": "brief explanation"}}"""

                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.ollama_model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json"
                        },
                        timeout=30
                    )

                    if response.ok:
                        result = response.json()
                        output = json.loads(result.get('response', '{}'))
                        article['ollama_sentiment'] = float(output.get('sentiment', 0))
                        article['ollama_confidence'] = float(output.get('confidence', 0.5))
                    else:
                        # Fall back to GDELT tone
                        article['ollama_sentiment'] = article.get('gdelt_tone', 0) / 100
                        article['ollama_confidence'] = 0.3

                except Exception as e:
                    logger.warning(f"Ollama analysis error: {e}")
                    article['ollama_sentiment'] = article.get('gdelt_tone', 0) / 100
                    article['ollama_confidence'] = 0.3

                results.append(article)

            # Rate limiting
            time.sleep(0.1)

        return results

    def fetch_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str] = None,
        workers: int = 4
    ) -> pd.DataFrame:
        """
        Fetch GDELT data for a date range

        For large ranges, this downloads daily GKG files in parallel.
        """
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        all_data = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.fetch_gkg_file, date): date
                for date in dates
            }

            for future in as_completed(futures):
                date = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        # Filter to stock-relevant news
                        df = self.filter_stock_relevant(df, symbols)
                        if not df.empty:
                            all_data.append(df)
                            logger.info(f"Found {len(df)} stock-relevant articles for {date.date()}")
                except Exception as e:
                    logger.error(f"Error processing {date}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def build_sentiment_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        use_ollama: bool = True
    ) -> pd.DataFrame:
        """
        Build daily sentiment timeseries for specified stocks

        Returns DataFrame with columns:
        - date
        - ticker
        - article_count
        - avg_gdelt_tone
        - avg_ollama_sentiment (if use_ollama=True)
        - sentiment_std
        - bullish_ratio
        - bearish_ratio
        """
        # Fetch GDELT data
        logger.info(f"Fetching GDELT data from {start_date.date()} to {end_date.date()}...")
        raw_data = self.fetch_date_range(start_date, end_date, symbols)

        if raw_data.empty:
            logger.warning("No GDELT data found for the specified range")
            return pd.DataFrame()

        # Extract stock mentions
        stock_data = self.extract_stock_mentions(raw_data)

        if stock_data.empty:
            logger.warning("No stock mentions found in GDELT data")
            return pd.DataFrame()

        # Optionally enhance with Ollama sentiment
        if use_ollama:
            logger.info("Analyzing sentiment with Ollama...")
            for ticker in stock_data['ticker'].unique():
                ticker_articles = stock_data[stock_data['ticker'] == ticker].to_dict('records')
                enhanced = self.analyze_sentiment_ollama(ticker_articles, ticker)

                # Update DataFrame
                for i, article in enumerate(enhanced):
                    idx = stock_data[(stock_data['ticker'] == ticker)].index[i]
                    stock_data.loc[idx, 'ollama_sentiment'] = article.get('ollama_sentiment', 0)
                    stock_data.loc[idx, 'ollama_confidence'] = article.get('ollama_confidence', 0.5)

        # Aggregate to daily timeseries
        stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

        sentiment_col = 'ollama_sentiment' if use_ollama and 'ollama_sentiment' in stock_data.columns else 'gdelt_tone'

        daily_sentiment = stock_data.groupby(['date', 'ticker']).agg({
            'url': 'count',  # Article count
            'gdelt_tone': ['mean', 'std'],
            sentiment_col: ['mean', 'std'] if sentiment_col in stock_data.columns else [],
        }).reset_index()

        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip('_') for col in daily_sentiment.columns]
        daily_sentiment = daily_sentiment.rename(columns={
            'url_count': 'article_count',
            'gdelt_tone_mean': 'avg_gdelt_tone',
            'gdelt_tone_std': 'gdelt_tone_std'
        })

        # Calculate bullish/bearish ratios
        def calc_ratios(group):
            sentiment = group[sentiment_col] if sentiment_col in group.columns else group['gdelt_tone'] / 100
            bullish = (sentiment > 0.1).mean()
            bearish = (sentiment < -0.1).mean()
            return pd.Series({'bullish_ratio': bullish, 'bearish_ratio': bearish})

        ratios = stock_data.groupby(['date', 'ticker']).apply(calc_ratios).reset_index()
        daily_sentiment = daily_sentiment.merge(ratios, on=['date', 'ticker'], how='left')

        return daily_sentiment


class PriceDerivedSentiment:
    """
    Creates sentiment proxy from price/volume action for historical periods
    where news data is not available or incomplete.

    This is crucial for 1979-2015 data where GDELT GKG is not available.

    Key insight: Price and volume patterns contain "fossilized sentiment":
    - Gap ups/downs reflect overnight news sentiment
    - Volume spikes indicate high attention/activity
    - Price volatility reflects uncertainty
    - Relative performance vs market indicates stock-specific news
    """

    def __init__(self):
        pass

    def calculate_sentiment_proxy(
        self,
        df: pd.DataFrame,
        market_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate sentiment proxy features from price data

        Args:
            df: Stock OHLCV data
            market_df: Market index data (e.g., SPY) for relative calculations

        Returns:
            DataFrame with sentiment proxy features
        """
        features = pd.DataFrame(index=df.index)

        # 1. Gap Analysis (overnight sentiment)
        features['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['gap_direction'] = np.sign(features['overnight_gap'])
        features['gap_magnitude'] = np.abs(features['overnight_gap'])

        # Large gaps indicate significant news
        features['large_gap_up'] = (features['overnight_gap'] > 0.02).astype(float)
        features['large_gap_down'] = (features['overnight_gap'] < -0.02).astype(float)

        # 2. Intraday Reversal (sentiment shift during trading)
        intraday_move = (df['close'] - df['open']) / df['open']
        features['gap_reversal'] = -features['overnight_gap'] * intraday_move

        # 3. Volume Sentiment
        avg_volume = df['volume'].rolling(20).mean()
        features['volume_surprise'] = (df['volume'] - avg_volume) / (avg_volume + 1e-8)

        # High volume on up days = bullish sentiment
        features['bullish_volume'] = features['volume_surprise'] * (df['close'] > df['open']).astype(float)
        features['bearish_volume'] = features['volume_surprise'] * (df['close'] < df['open']).astype(float)

        # 4. Range Analysis (uncertainty/volatility)
        features['daily_range'] = (df['high'] - df['low']) / df['open']
        features['range_expansion'] = features['daily_range'] / features['daily_range'].rolling(20).mean()

        # High range on down days = fear
        features['fear_signal'] = features['range_expansion'] * (df['close'] < df['open']).astype(float)

        # 5. Close Position (buyer/seller control)
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # 6. Relative Performance (stock-specific sentiment)
        if market_df is not None and 'close' in market_df.columns:
            stock_return = df['close'].pct_change()
            market_return = market_df['close'].pct_change()

            # Align indices
            common_idx = stock_return.index.intersection(market_return.index)
            if len(common_idx) > 0:
                features['excess_return'] = stock_return.loc[common_idx] - market_return.loc[common_idx]
                features['relative_strength'] = (
                    (1 + stock_return.loc[common_idx]).rolling(20).apply(np.prod) /
                    (1 + market_return.loc[common_idx]).rolling(20).apply(np.prod) - 1
                )

        # 7. Momentum Divergence (sentiment exhaustion)
        price_momentum = df['close'].pct_change(5)
        volume_momentum = df['volume'].pct_change(5)
        features['momentum_divergence'] = price_momentum / (np.abs(volume_momentum) + 0.01)

        # 8. Composite Sentiment Score (-1 to +1)
        excess_return_contrib = 0
        if 'excess_return' in features.columns:
            excess_return_contrib = features['excess_return'].clip(-0.05, 0.05) * 20

        features['sentiment_proxy'] = (
            0.2 * features['overnight_gap'].clip(-0.1, 0.1) * 10 +  # Gap contribution
            0.2 * (features['close_position'] * 2 - 1) +            # Close position
            0.2 * np.tanh(features['bullish_volume'] - features['bearish_volume']) +  # Volume sentiment
            0.2 * -np.tanh(features['fear_signal']) +               # Fear (negative)
            0.2 * excess_return_contrib                             # Relative performance
        )

        # Smooth the composite score
        features['sentiment_proxy_smooth'] = features['sentiment_proxy'].rolling(5).mean()

        # Fill NaN
        features = features.fillna(0)

        return features

    def estimate_news_intensity(
        self,
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Estimate relative news intensity from price/volume patterns

        High news intensity days typically show:
        - Unusual volume
        - Large gaps
        - High intraday range
        """
        avg_volume = df['volume'].rolling(20).mean()
        volume_surprise = np.abs((df['volume'] - avg_volume) / (avg_volume + 1e-8))

        gap = np.abs((df['open'] - df['close'].shift(1)) / df['close'].shift(1))

        range_pct = (df['high'] - df['low']) / df['open']
        avg_range = range_pct.rolling(20).mean()
        range_surprise = range_pct / (avg_range + 1e-8)

        # Combine signals
        intensity = (
            0.4 * volume_surprise.clip(0, 5) / 5 +
            0.3 * gap.clip(0, 0.1) / 0.1 +
            0.3 * (range_surprise - 1).clip(0, 3) / 3
        )

        return intensity.clip(0, 1)


def build_historical_sentiment_database(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: str = "data/sentiment_history",
    use_gdelt_2015_plus: bool = True,
    use_price_proxy_pre_2015: bool = True,
    price_data_dir: str = "data/historical"
) -> pd.DataFrame:
    """
    Build comprehensive historical sentiment database

    Strategy:
    - 2015+: Use GDELT GKG with Ollama enhancement
    - Pre-2015: Use price-derived sentiment proxy

    This gives us 25+ years of sentiment data for training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_sentiment = []

    # Split by GDELT availability
    gdelt_start = datetime(2015, 2, 18)  # GDELT GKG started

    if use_gdelt_2015_plus and end_date >= gdelt_start:
        gdelt_from = max(start_date, gdelt_start)
        logger.info(f"Fetching GDELT sentiment from {gdelt_from.date()} to {end_date.date()}")

        fetcher = GDELTFetcher(output_dir=str(output_path / "gdelt"))
        gdelt_sentiment = fetcher.build_sentiment_timeseries(
            gdelt_from, end_date, symbols, use_ollama=True
        )

        if not gdelt_sentiment.empty:
            gdelt_sentiment['source'] = 'gdelt'
            all_sentiment.append(gdelt_sentiment)

    if use_price_proxy_pre_2015 and start_date < gdelt_start:
        price_end = min(end_date, gdelt_start - timedelta(days=1))
        logger.info(f"Building price-derived sentiment from {start_date.date()} to {price_end.date()}")

        price_sentiment = PriceDerivedSentiment()
        price_data_path = Path(price_data_dir)

        for symbol in symbols:
            # Try to load price data
            price_file = price_data_path / f"{symbol}_max.parquet"
            if not price_file.exists():
                continue

            df = pd.read_parquet(price_file)

            # Filter to date range
            df = df[(df.index >= start_date) & (df.index <= price_end)]

            if df.empty:
                continue

            # Calculate sentiment proxy
            sentiment_features = price_sentiment.calculate_sentiment_proxy(df)

            # Create daily records
            daily_records = pd.DataFrame({
                'date': df.index.date,
                'ticker': symbol,
                'article_count': 0,  # No actual articles
                'avg_gdelt_tone': 0,  # Not available
                'sentiment_proxy': sentiment_features['sentiment_proxy_smooth'].values,
                'news_intensity': price_sentiment.estimate_news_intensity(df).values,
                'source': 'price_proxy'
            })

            all_sentiment.append(daily_records)

    if all_sentiment:
        combined = pd.concat(all_sentiment, ignore_index=True)

        # Save to disk
        output_file = output_path / "sentiment_database.parquet"
        combined.to_parquet(output_file)
        logger.info(f"Saved {len(combined)} sentiment records to {output_file}")

        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    # Test price-derived sentiment first (doesn't need network)
    print("=" * 60)
    print("Price-Derived Sentiment Test")
    print("=" * 60)

    price_sentiment = PriceDerivedSentiment()

    # Create sample data with more realistic values
    np.random.seed(42)
    n_days = 50
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.cumprod(1 + returns)

    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.random.uniform(0.005, 0.03, n_days)),
        'low': prices * (1 - np.random.uniform(0.005, 0.03, n_days)),
        'close': prices,
        'volume': np.random.randint(500000, 2000000, n_days)
    })

    features = price_sentiment.calculate_sentiment_proxy(sample_data)
    print("\nPrice-derived sentiment features (sample):")
    cols = ['overnight_gap', 'close_position', 'bullish_volume', 'fear_signal', 'sentiment_proxy', 'sentiment_proxy_smooth']
    print(features[cols].tail(10).round(3))

    print(f"\nSentiment proxy statistics:")
    print(f"  Mean: {features['sentiment_proxy'].mean():.3f}")
    print(f"  Std:  {features['sentiment_proxy'].std():.3f}")
    print(f"  Min:  {features['sentiment_proxy'].min():.3f}")
    print(f"  Max:  {features['sentiment_proxy'].max():.3f}")

    # News intensity estimation
    intensity = price_sentiment.estimate_news_intensity(sample_data)
    print(f"\nNews intensity estimation:")
    print(f"  Mean: {intensity.mean():.3f}")
    print(f"  High-news days (>0.5): {(intensity > 0.5).sum()}")

    # Test GDELT fetching with historical dates
    print("\n" + "=" * 60)
    print("GDELT Historical News Fetcher Test")
    print("=" * 60)

    fetcher = GDELTFetcher()

    # Use a known historical date range (Nov 2024 - should have data)
    end_date = datetime(2024, 11, 15)
    start_date = datetime(2024, 11, 10)

    print(f"\nFetching GDELT data from {start_date.date()} to {end_date.date()}...")
    print("(This tests the GKG file parsing - actual download may take time)")

    # Test with a few major symbols
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL']

    try:
        sentiment = fetcher.build_sentiment_timeseries(
            start_date, end_date, test_symbols, use_ollama=False
        )

        if not sentiment.empty:
            print(f"\nFetched {len(sentiment)} daily sentiment records")
            print(f"\nSample data:")
            print(sentiment.head(20))

            print(f"\nSentiment by ticker:")
            print(sentiment.groupby('ticker').agg({
                'article_count': 'sum',
                'avg_gdelt_tone': 'mean'
            }))
        else:
            print("\nNo GDELT data found (this is OK - GKG files are large and may not be cached)")
            print("The price-derived sentiment proxy works as a fallback.")

    except Exception as e:
        print(f"\nGDELT fetch error (expected for testing): {e}")
        print("The price-derived sentiment proxy works as a fallback for historical data.")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Historical sentiment strategy:
1. 2015+: GDELT GKG with Ollama sentiment enhancement
2. Pre-2015: Price-derived sentiment proxy (works offline)

The price-derived proxy captures 'fossilized' sentiment from:
- Overnight gaps (news events)
- Volume patterns (market attention)
- Intraday reversals (sentiment shifts)
- Relative performance vs market (stock-specific news)
""")
