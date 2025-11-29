"""
News Sentiment Pipeline - Free Data Sources + Local Ollama LLM

Scrapes news from free sources:
- Google News RSS
- Yahoo Finance
- Finviz
- Reddit (wallstreetbets, stocks, investing)
- SEC EDGAR filings (8-K, 10-K sentiment)

Processes through local Ollama LLM for sentiment scoring (-1 to +1)
Aggregates daily sentiment per stock
"""

import asyncio
import aiohttp
import feedparser
import requests
import json
import re
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    source: str
    symbol: str
    published_date: datetime
    url: str
    content_snippet: str = ""
    sentiment_score: float = None  # -1 to +1
    sentiment_confidence: float = None
    sentiment_reasoning: str = ""
    article_hash: str = ""

    def __post_init__(self):
        if not self.article_hash:
            self.article_hash = hashlib.md5(
                f"{self.title}{self.url}".encode()
            ).hexdigest()[:16]


class NewsDatabase:
    """SQLite database for caching news and sentiment"""

    def __init__(self, db_path: str = "data/news_sentiment.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    hash TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    symbol TEXT,
                    published_date TEXT,
                    url TEXT,
                    content_snippet TEXT,
                    sentiment_score REAL,
                    sentiment_confidence REAL,
                    sentiment_reasoning TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_date
                ON articles(symbol, published_date)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_sentiment (
                    symbol TEXT,
                    date TEXT,
                    avg_sentiment REAL,
                    sentiment_std REAL,
                    article_count INTEGER,
                    source_breakdown TEXT,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.commit()

    def article_exists(self, article_hash: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT 1 FROM articles WHERE hash = ?", (article_hash,)
            ).fetchone()
            return result is not None

    def save_article(self, article: NewsArticle):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO articles
                (hash, title, source, symbol, published_date, url,
                 content_snippet, sentiment_score, sentiment_confidence, sentiment_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.article_hash,
                article.title,
                article.source,
                article.symbol,
                article.published_date.isoformat(),
                article.url,
                article.content_snippet,
                article.sentiment_score,
                article.sentiment_confidence,
                article.sentiment_reasoning
            ))
            conn.commit()

    def get_articles_for_symbol(
        self,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[NewsArticle]:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM articles WHERE symbol = ?"
            params = [symbol]

            if start_date:
                query += " AND published_date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND published_date <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY published_date DESC"

            rows = conn.execute(query, params).fetchall()

            articles = []
            for row in rows:
                articles.append(NewsArticle(
                    article_hash=row[0],
                    title=row[1],
                    source=row[2],
                    symbol=row[3],
                    published_date=datetime.fromisoformat(row[4]),
                    url=row[5],
                    content_snippet=row[6],
                    sentiment_score=row[7],
                    sentiment_confidence=row[8],
                    sentiment_reasoning=row[9]
                ))
            return articles

    def save_daily_sentiment(
        self,
        symbol: str,
        date: datetime,
        avg_sentiment: float,
        sentiment_std: float,
        article_count: int,
        source_breakdown: Dict[str, float]
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_sentiment
                (symbol, date, avg_sentiment, sentiment_std, article_count, source_breakdown)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                date.strftime("%Y-%m-%d"),
                avg_sentiment,
                sentiment_std,
                article_count,
                json.dumps(source_breakdown)
            ))
            conn.commit()

    def get_daily_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT date, avg_sentiment, sentiment_std, article_count, source_breakdown
                FROM daily_sentiment
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, conn, params=(
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            ))
            return df


class GoogleNewsScraper:
    """Scrapes Google News RSS feeds"""

    BASE_URL = "https://news.google.com/rss/search"

    def __init__(self):
        self.session = None

    async def get_news(
        self,
        symbol: str,
        company_name: str = None,
        days_back: int = 7
    ) -> List[NewsArticle]:
        articles = []

        # Search queries
        queries = [f"{symbol} stock"]
        if company_name:
            queries.append(f'"{company_name}" stock')

        for query in queries:
            url = f"{self.BASE_URL}?q={query}&hl=en-US&gl=US&ceid=US:en"

            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:20]:  # Limit per query
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])

                        # Skip old articles
                        if pub_date < datetime.now() - timedelta(days=days_back):
                            continue

                        article = NewsArticle(
                            title=entry.title,
                            source="google_news",
                            symbol=symbol,
                            published_date=pub_date,
                            url=entry.link,
                            content_snippet=BeautifulSoup(
                                entry.get('summary', ''), 'html.parser'
                            ).get_text()[:500]
                        )
                        articles.append(article)
                    except Exception as e:
                        logger.debug(f"Error parsing entry: {e}")
                        continue

            except Exception as e:
                logger.error(f"Google News error for {symbol}: {e}")

        return articles


class YahooFinanceScraper:
    """Scrapes Yahoo Finance news"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        articles = []
        url = f"https://finance.yahoo.com/quote/{symbol}/news"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return articles

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Find news items (Yahoo's structure changes frequently)
                    news_items = soup.find_all('li', class_=re.compile('stream-item'))

                    for item in news_items[:15]:
                        try:
                            link = item.find('a')
                            if not link:
                                continue

                            title = link.get_text(strip=True)
                            href = link.get('href', '')

                            if not title or len(title) < 10:
                                continue

                            article = NewsArticle(
                                title=title,
                                source="yahoo_finance",
                                symbol=symbol,
                                published_date=datetime.now(),  # Yahoo doesn't always show dates
                                url=f"https://finance.yahoo.com{href}" if href.startswith('/') else href,
                                content_snippet=""
                            )
                            articles.append(article)
                        except Exception as e:
                            continue

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")

        return articles


class FinvizScraper:
    """Scrapes Finviz news"""

    BASE_URL = "https://finviz.com/quote.ashx"

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        articles = []
        url = f"{self.BASE_URL}?t={symbol}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return articles

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Find news table
                    news_table = soup.find('table', class_='fullview-news-outer')
                    if not news_table:
                        return articles

                    rows = news_table.find_all('tr')

                    for row in rows[:20]:
                        try:
                            cells = row.find_all('td')
                            if len(cells) < 2:
                                continue

                            date_cell = cells[0].get_text(strip=True)
                            link = cells[1].find('a')

                            if not link:
                                continue

                            title = link.get_text(strip=True)
                            href = link.get('href', '')

                            # Parse date (format: "Dec-28-24 08:30AM" or "Today 08:30AM")
                            try:
                                if 'Today' in date_cell:
                                    pub_date = datetime.now()
                                elif 'Yesterday' in date_cell:
                                    pub_date = datetime.now() - timedelta(days=1)
                                else:
                                    pub_date = datetime.strptime(
                                        date_cell.split()[0], "%b-%d-%y"
                                    )
                            except:
                                pub_date = datetime.now()

                            article = NewsArticle(
                                title=title,
                                source="finviz",
                                symbol=symbol,
                                published_date=pub_date,
                                url=href,
                                content_snippet=""
                            )
                            articles.append(article)
                        except Exception as e:
                            continue

        except Exception as e:
            logger.error(f"Finviz error for {symbol}: {e}")

        return articles


class RedditScraper:
    """Scrapes Reddit for stock discussions"""

    SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 MarketBrain/1.0'
        }

    async def get_news(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        articles = []

        for subreddit in self.SUBREDDITS:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': f"{symbol} OR ${symbol}",
                'restrict_sr': 'on',
                'sort': 'new',
                't': 'week',
                'limit': 25
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        params=params,
                        headers=self.headers
                    ) as response:
                        if response.status != 200:
                            continue

                        data = await response.json()

                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})

                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')[:500]
                            created = post_data.get('created_utc', 0)
                            permalink = post_data.get('permalink', '')
                            score = post_data.get('score', 0)

                            # Skip low quality posts
                            if score < 10:
                                continue

                            pub_date = datetime.fromtimestamp(created)

                            article = NewsArticle(
                                title=title,
                                source=f"reddit_{subreddit}",
                                symbol=symbol,
                                published_date=pub_date,
                                url=f"https://reddit.com{permalink}",
                                content_snippet=selftext
                            )
                            articles.append(article)

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Reddit error for {symbol} in r/{subreddit}: {e}")

        return articles


class OllamaSentimentAnalyzer:
    """
    Uses local Ollama LLM for sentiment analysis

    Supports models: llama3, mistral, phi3, etc.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def _create_prompt(self, article: NewsArticle) -> str:
        return f"""Analyze the sentiment of this stock news article for {article.symbol}.

Title: {article.title}
Content: {article.content_snippet[:1000] if article.content_snippet else 'N/A'}

Respond with ONLY a JSON object (no other text):
{{
    "sentiment_score": <float from -1.0 (very negative) to 1.0 (very positive)>,
    "confidence": <float from 0.0 to 1.0>,
    "reasoning": "<brief 1-2 sentence explanation>"
}}

Consider:
- Direct impact on stock price
- Market sentiment implications
- Is this fear-mongering or genuine concern?
- Is this hype or substantive positive news?

JSON response:"""

    async def analyze(self, article: NewsArticle) -> NewsArticle:
        """Analyze a single article"""
        prompt = self._create_prompt(article)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temp for consistent scoring
                            "num_predict": 200
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Ollama error: {response.status}")
                        return article

                    data = await response.json()
                    response_text = data.get('response', '')

                    # Parse JSON from response
                    try:
                        # Find JSON in response
                        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                            article.sentiment_score = float(result.get('sentiment_score', 0))
                            article.sentiment_confidence = float(result.get('confidence', 0.5))
                            article.sentiment_reasoning = result.get('reasoning', '')

                            # Clamp values
                            article.sentiment_score = max(-1, min(1, article.sentiment_score))
                            article.sentiment_confidence = max(0, min(1, article.sentiment_confidence))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama response: {response_text[:100]}")

        except asyncio.TimeoutError:
            logger.warning(f"Ollama timeout for article: {article.title[:50]}")
        except Exception as e:
            logger.error(f"Ollama error: {e}")

        return article

    async def analyze_batch(
        self,
        articles: List[NewsArticle],
        max_concurrent: int = 3
    ) -> List[NewsArticle]:
        """Analyze multiple articles with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(article):
            async with semaphore:
                return await self.analyze(article)

        tasks = [analyze_with_limit(article) for article in articles]
        return await asyncio.gather(*tasks)

    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                return self.model in model_names
        except:
            pass
        return False


class SentimentPipeline:
    """
    Main pipeline for collecting and analyzing news sentiment
    """

    def __init__(
        self,
        ollama_model: str = "llama3",
        db_path: str = "data/news_sentiment.db"
    ):
        self.db = NewsDatabase(db_path)
        self.analyzer = OllamaSentimentAnalyzer(model=ollama_model)

        # Scrapers
        self.scrapers = [
            GoogleNewsScraper(),
            YahooFinanceScraper(),
            FinvizScraper(),
            RedditScraper()
        ]

        self.executor = ThreadPoolExecutor(max_workers=4)

    async def fetch_news(
        self,
        symbol: str,
        company_name: str = None,
        days_back: int = 7
    ) -> List[NewsArticle]:
        """Fetch news from all sources"""
        all_articles = []

        tasks = []
        for scraper in self.scrapers:
            if isinstance(scraper, GoogleNewsScraper):
                tasks.append(scraper.get_news(symbol, company_name, days_back))
            else:
                tasks.append(scraper.get_news(symbol, days_back))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Scraper error: {result}")

        # Deduplicate by hash
        seen = set()
        unique_articles = []
        for article in all_articles:
            if article.article_hash not in seen:
                seen.add(article.article_hash)
                unique_articles.append(article)

        logger.info(f"Fetched {len(unique_articles)} unique articles for {symbol}")
        return unique_articles

    async def process_symbol(
        self,
        symbol: str,
        company_name: str = None,
        days_back: int = 7,
        skip_existing: bool = True
    ) -> List[NewsArticle]:
        """
        Fetch news, analyze sentiment, and store results
        """
        # Fetch news
        articles = await self.fetch_news(symbol, company_name, days_back)

        # Filter out already processed
        if skip_existing:
            articles = [a for a in articles if not self.db.article_exists(a.article_hash)]
            logger.info(f"Processing {len(articles)} new articles for {symbol}")

        if not articles:
            return []

        # Analyze sentiment
        if self.analyzer.check_availability():
            articles = await self.analyzer.analyze_batch(articles)
        else:
            logger.warning("Ollama not available, skipping sentiment analysis")

        # Save to database
        for article in articles:
            self.db.save_article(article)

        return articles

    def compute_daily_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Aggregate article sentiment into daily scores
        """
        articles = self.db.get_articles_for_symbol(symbol, start_date, end_date)

        if not articles:
            return pd.DataFrame()

        # Group by date
        daily_data = {}
        for article in articles:
            if article.sentiment_score is None:
                continue

            date_key = article.published_date.strftime("%Y-%m-%d")
            if date_key not in daily_data:
                daily_data[date_key] = {
                    'scores': [],
                    'sources': {}
                }

            daily_data[date_key]['scores'].append(article.sentiment_score)

            source = article.source
            if source not in daily_data[date_key]['sources']:
                daily_data[date_key]['sources'][source] = []
            daily_data[date_key]['sources'][source].append(article.sentiment_score)

        # Compute daily aggregates
        results = []
        for date, data in daily_data.items():
            scores = data['scores']
            if not scores:
                continue

            avg_sentiment = sum(scores) / len(scores)
            sentiment_std = (
                (sum((s - avg_sentiment) ** 2 for s in scores) / len(scores)) ** 0.5
                if len(scores) > 1 else 0
            )

            source_breakdown = {
                source: sum(s) / len(s)
                for source, s in data['sources'].items()
            }

            results.append({
                'date': date,
                'avg_sentiment': avg_sentiment,
                'sentiment_std': sentiment_std,
                'article_count': len(scores),
                'source_breakdown': source_breakdown
            })

            # Save to database
            self.db.save_daily_sentiment(
                symbol,
                datetime.strptime(date, "%Y-%m-%d"),
                avg_sentiment,
                sentiment_std,
                len(scores),
                source_breakdown
            )

        return pd.DataFrame(results)

    async def run_daily_update(
        self,
        symbols: List[str],
        symbol_names: Dict[str, str] = None
    ):
        """
        Run daily sentiment update for all symbols
        """
        symbol_names = symbol_names or {}

        logger.info(f"Starting daily sentiment update for {len(symbols)} symbols")

        for i, symbol in enumerate(symbols):
            try:
                company_name = symbol_names.get(symbol)
                await self.process_symbol(symbol, company_name, days_back=3)

                # Compute daily aggregates
                self.compute_daily_sentiment(
                    symbol,
                    datetime.now() - timedelta(days=7),
                    datetime.now()
                )

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(symbols)} symbols")

                # Rate limiting
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info("Daily sentiment update complete")


async def main():
    """Example usage"""
    pipeline = SentimentPipeline(ollama_model="llama3")

    # Check Ollama availability
    if pipeline.analyzer.check_availability():
        print("✓ Ollama is available")
    else:
        print("✗ Ollama not available - install and run: ollama run llama3")
        return

    # Process a single symbol
    articles = await pipeline.process_symbol("NVDA", "NVIDIA", days_back=3)

    print(f"\nProcessed {len(articles)} articles for NVDA:")
    for article in articles[:5]:
        print(f"  [{article.source}] {article.title[:60]}...")
        if article.sentiment_score is not None:
            print(f"    Sentiment: {article.sentiment_score:.2f} ({article.sentiment_reasoning})")

    # Get daily sentiment
    daily = pipeline.compute_daily_sentiment(
        "NVDA",
        datetime.now() - timedelta(days=7),
        datetime.now()
    )
    print(f"\nDaily sentiment for NVDA:")
    print(daily)


if __name__ == "__main__":
    asyncio.run(main())
