"""
GICS Sector and Industry Classification System

Provides standardized sector/industry classification for all stocks.
Uses GICS (Global Industry Classification Standard) with 11 sectors and 69 industries.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GICS Sector definitions (11 sectors)
GICS_SECTORS = {
    0: {"name": "Information Technology", "aliases": ["Technology", "Tech"]},
    1: {"name": "Health Care", "aliases": ["Healthcare"]},
    2: {"name": "Financials", "aliases": ["Financial Services", "Finance"]},
    3: {"name": "Consumer Discretionary", "aliases": ["Consumer Cyclical"]},
    4: {"name": "Consumer Staples", "aliases": ["Consumer Defensive"]},
    5: {"name": "Industrials", "aliases": ["Industrial"]},
    6: {"name": "Energy", "aliases": []},
    7: {"name": "Utilities", "aliases": ["Utility"]},
    8: {"name": "Real Estate", "aliases": ["REIT"]},
    9: {"name": "Materials", "aliases": ["Basic Materials"]},
    10: {"name": "Communication Services", "aliases": ["Telecommunications", "Media"]},
}

# GICS Industry definitions (69 industries mapped to sector)
GICS_INDUSTRIES = {
    # Information Technology (0)
    0: {"name": "Software", "sector_id": 0},
    1: {"name": "Hardware", "sector_id": 0},
    2: {"name": "Semiconductors", "sector_id": 0},
    3: {"name": "IT Services", "sector_id": 0},
    4: {"name": "Electronic Equipment", "sector_id": 0},
    5: {"name": "Technology Distributors", "sector_id": 0},

    # Health Care (1)
    6: {"name": "Biotechnology", "sector_id": 1},
    7: {"name": "Pharmaceuticals", "sector_id": 1},
    8: {"name": "Health Care Equipment", "sector_id": 1},
    9: {"name": "Health Care Services", "sector_id": 1},
    10: {"name": "Life Sciences Tools", "sector_id": 1},
    11: {"name": "Health Care Technology", "sector_id": 1},

    # Financials (2)
    12: {"name": "Banks", "sector_id": 2},
    13: {"name": "Insurance", "sector_id": 2},
    14: {"name": "Capital Markets", "sector_id": 2},
    15: {"name": "Consumer Finance", "sector_id": 2},
    16: {"name": "Mortgage Finance", "sector_id": 2},
    17: {"name": "Financial Exchanges", "sector_id": 2},

    # Consumer Discretionary (3)
    18: {"name": "Automobiles", "sector_id": 3},
    19: {"name": "Auto Components", "sector_id": 3},
    20: {"name": "Household Durables", "sector_id": 3},
    21: {"name": "Leisure Products", "sector_id": 3},
    22: {"name": "Textiles & Apparel", "sector_id": 3},
    23: {"name": "Hotels & Restaurants", "sector_id": 3},
    24: {"name": "Retail - Discretionary", "sector_id": 3},
    25: {"name": "Internet Retail", "sector_id": 3},

    # Consumer Staples (4)
    26: {"name": "Food & Beverage", "sector_id": 4},
    27: {"name": "Tobacco", "sector_id": 4},
    28: {"name": "Household Products", "sector_id": 4},
    29: {"name": "Personal Products", "sector_id": 4},
    30: {"name": "Retail - Staples", "sector_id": 4},

    # Industrials (5)
    31: {"name": "Aerospace & Defense", "sector_id": 5},
    32: {"name": "Building Products", "sector_id": 5},
    33: {"name": "Construction & Engineering", "sector_id": 5},
    34: {"name": "Electrical Equipment", "sector_id": 5},
    35: {"name": "Industrial Conglomerates", "sector_id": 5},
    36: {"name": "Machinery", "sector_id": 5},
    37: {"name": "Commercial Services", "sector_id": 5},
    38: {"name": "Professional Services", "sector_id": 5},
    39: {"name": "Airlines", "sector_id": 5},
    40: {"name": "Freight & Logistics", "sector_id": 5},
    41: {"name": "Transportation Infrastructure", "sector_id": 5},

    # Energy (6)
    42: {"name": "Oil & Gas - Exploration", "sector_id": 6},
    43: {"name": "Oil & Gas - Integrated", "sector_id": 6},
    44: {"name": "Oil & Gas - Refining", "sector_id": 6},
    45: {"name": "Oil & Gas - Equipment", "sector_id": 6},
    46: {"name": "Renewable Energy", "sector_id": 6},

    # Utilities (7)
    47: {"name": "Electric Utilities", "sector_id": 7},
    48: {"name": "Gas Utilities", "sector_id": 7},
    49: {"name": "Multi-Utilities", "sector_id": 7},
    50: {"name": "Water Utilities", "sector_id": 7},
    51: {"name": "Independent Power", "sector_id": 7},

    # Real Estate (8)
    52: {"name": "Equity REITs", "sector_id": 8},
    53: {"name": "Mortgage REITs", "sector_id": 8},
    54: {"name": "Real Estate Services", "sector_id": 8},
    55: {"name": "Real Estate Development", "sector_id": 8},

    # Materials (9)
    56: {"name": "Chemicals", "sector_id": 9},
    57: {"name": "Construction Materials", "sector_id": 9},
    58: {"name": "Containers & Packaging", "sector_id": 9},
    59: {"name": "Metals & Mining", "sector_id": 9},
    60: {"name": "Paper & Forest Products", "sector_id": 9},

    # Communication Services (10)
    61: {"name": "Diversified Telecom", "sector_id": 10},
    62: {"name": "Wireless Telecom", "sector_id": 10},
    63: {"name": "Media", "sector_id": 10},
    64: {"name": "Entertainment", "sector_id": 10},
    65: {"name": "Interactive Media", "sector_id": 10},
    66: {"name": "Advertising", "sector_id": 10},
    67: {"name": "Publishing", "sector_id": 10},
    68: {"name": "Cable & Satellite", "sector_id": 10},
}

# Industry name to ID mapping (for string lookups)
INDUSTRY_NAME_TO_ID = {
    info["name"].lower(): id for id, info in GICS_INDUSTRIES.items()
}

# Common industry aliases
INDUSTRY_ALIASES = {
    "software—infrastructure": 0,
    "software—application": 0,
    "software - infrastructure": 0,
    "software - application": 0,
    "semiconductors": 2,
    "semiconductor equipment & materials": 2,
    "semiconductor memory": 2,
    "computer hardware": 1,
    "consumer electronics": 1,
    "electronic components": 4,
    "scientific & technical instruments": 4,
    "communication equipment": 4,
    "information technology services": 3,
    "internet content & information": 65,
    "internet retail": 25,
    "specialty retail": 24,
    "apparel retail": 24,
    "home improvement retail": 24,
    "department stores": 24,
    "discount stores": 30,
    "grocery stores": 30,
    "drug manufacturers—general": 7,
    "drug manufacturers—specialty & generic": 7,
    "biotechnology": 6,
    "medical devices": 8,
    "medical instruments & supplies": 8,
    "diagnostics & research": 10,
    "health information services": 11,
    "medical care facilities": 9,
    "banks—regional": 12,
    "banks—diversified": 12,
    "credit services": 15,
    "insurance—life": 13,
    "insurance—property & casualty": 13,
    "insurance—diversified": 13,
    "asset management": 14,
    "capital markets": 14,
    "financial data & stock exchanges": 17,
    "auto manufacturers": 18,
    "auto parts": 19,
    "recreational vehicles": 21,
    "gambling": 21,
    "resorts & casinos": 23,
    "restaurants": 23,
    "lodging": 23,
    "travel services": 23,
    "apparel manufacturing": 22,
    "footwear & accessories": 22,
    "luxury goods": 22,
    "household & personal products": 28,
    "packaged foods": 26,
    "beverages—non-alcoholic": 26,
    "beverages—brewers": 26,
    "beverages—wineries & distilleries": 26,
    "tobacco": 27,
    "aerospace & defense": 31,
    "airlines": 39,
    "railroads": 41,
    "trucking": 40,
    "integrated freight & logistics": 40,
    "marine shipping": 40,
    "farm & heavy construction machinery": 36,
    "specialty industrial machinery": 36,
    "metal fabrication": 36,
    "industrial distribution": 37,
    "security & protection services": 37,
    "staffing & employment services": 38,
    "consulting services": 38,
    "conglomerates": 35,
    "electrical equipment & parts": 34,
    "building products & equipment": 32,
    "engineering & construction": 33,
    "oil & gas e&p": 42,
    "oil & gas integrated": 43,
    "oil & gas refining & marketing": 44,
    "oil & gas equipment & services": 45,
    "oil & gas midstream": 43,
    "oil & gas drilling": 45,
    "solar": 46,
    "utilities—regulated electric": 47,
    "utilities—regulated gas": 48,
    "utilities—diversified": 49,
    "utilities—renewable": 51,
    "utilities—independent power producers": 51,
    "reit—residential": 52,
    "reit—retail": 52,
    "reit—office": 52,
    "reit—industrial": 52,
    "reit—healthcare facilities": 52,
    "reit—hotel & motel": 52,
    "reit—diversified": 52,
    "reit—mortgage": 53,
    "reit—specialty": 52,
    "real estate services": 54,
    "real estate—development": 55,
    "real estate—diversified": 54,
    "specialty chemicals": 56,
    "agricultural inputs": 56,
    "chemicals": 56,
    "building materials": 57,
    "packaging & containers": 58,
    "gold": 59,
    "silver": 59,
    "copper": 59,
    "other industrial metals & mining": 59,
    "steel": 59,
    "aluminum": 59,
    "coking coal": 59,
    "lumber & wood production": 60,
    "paper & paper products": 60,
    "telecom services": 61,
    "advertising agencies": 66,
    "broadcasting": 63,
    "entertainment": 64,
    "electronic gaming & multimedia": 64,
    "publishing": 67,
}


class SectorClassifier:
    """Classifies stocks by sector and industry"""

    def __init__(self, cache_path: str = "data/sector_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _normalize_industry(self, industry: str) -> int:
        """Convert industry string to ID"""
        if not industry:
            return -1

        industry_lower = industry.lower().strip()

        # Check aliases first
        if industry_lower in INDUSTRY_ALIASES:
            return INDUSTRY_ALIASES[industry_lower]

        # Check exact match
        if industry_lower in INDUSTRY_NAME_TO_ID:
            return INDUSTRY_NAME_TO_ID[industry_lower]

        # Fuzzy match
        for name, id in INDUSTRY_NAME_TO_ID.items():
            if name in industry_lower or industry_lower in name:
                return id

        return -1

    def _normalize_sector(self, sector: str) -> int:
        """Convert sector string to ID"""
        if not sector:
            return -1

        sector_lower = sector.lower().strip()

        for sector_id, info in GICS_SECTORS.items():
            if sector_lower == info["name"].lower():
                return sector_id
            for alias in info["aliases"]:
                if sector_lower == alias.lower():
                    return sector_id

        return -1

    def classify(self, symbol: str, force_refresh: bool = False) -> Tuple[int, int, str, str]:
        """
        Get sector and industry classification for a symbol

        Returns: (sector_id, industry_id, sector_name, industry_name)
        """
        # Check cache
        if symbol in self.cache and not force_refresh:
            cached = self.cache[symbol]
            return (
                cached["sector_id"],
                cached["industry_id"],
                cached["sector_name"],
                cached["industry_name"]
            )

        # Fetch from Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector_name = info.get('sector', '')
            industry_name = info.get('industry', '')

            sector_id = self._normalize_sector(sector_name)
            industry_id = self._normalize_industry(industry_name)

            # If industry ID found, ensure sector matches
            if industry_id >= 0:
                expected_sector = GICS_INDUSTRIES[industry_id]["sector_id"]
                if sector_id < 0:
                    sector_id = expected_sector
                    sector_name = GICS_SECTORS[sector_id]["name"]

            # Cache result
            self.cache[symbol] = {
                "sector_id": sector_id,
                "industry_id": industry_id,
                "sector_name": sector_name or "Unknown",
                "industry_name": industry_name or "Unknown"
            }
            self._save_cache()

            return sector_id, industry_id, sector_name, industry_name

        except Exception as e:
            logger.warning(f"Could not classify {symbol}: {e}")
            return -1, -1, "Unknown", "Unknown"

    def get_sector_stocks(self, sector_id: int) -> list:
        """Get all cached stocks in a sector"""
        return [
            sym for sym, data in self.cache.items()
            if data.get("sector_id") == sector_id
        ]

    def get_industry_stocks(self, industry_id: int) -> list:
        """Get all cached stocks in an industry"""
        return [
            sym for sym, data in self.cache.items()
            if data.get("industry_id") == industry_id
        ]

    def classify_batch(self, symbols: list, max_workers: int = 5) -> Dict:
        """Classify multiple symbols"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        results = {}
        uncached = [s for s in symbols if s not in self.cache]

        logger.info(f"Classifying {len(uncached)} uncached symbols (cached: {len(symbols) - len(uncached)})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.classify, sym): sym
                for sym in uncached
            }

            for i, future in enumerate(as_completed(futures)):
                sym = futures[future]
                try:
                    sector_id, industry_id, sector_name, industry_name = future.result()
                    results[sym] = {
                        "sector_id": sector_id,
                        "industry_id": industry_id,
                        "sector_name": sector_name,
                        "industry_name": industry_name
                    }
                except Exception as e:
                    logger.error(f"Error classifying {sym}: {e}")

                if (i + 1) % 50 == 0:
                    logger.info(f"Classified {i + 1}/{len(uncached)}")
                    time.sleep(1)  # Rate limiting

        # Add cached results
        for sym in symbols:
            if sym in self.cache:
                results[sym] = self.cache[sym]

        return results


def get_anchor_stocks() -> Dict[int, list]:
    """
    Returns anchor stocks for each sector
    These are large, influential stocks that often lead sector moves
    """
    return {
        0: ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO"],  # IT
        1: ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],  # Healthcare
        2: ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C", "AXP", "SPGI"],  # Financials
        3: ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],  # Consumer Disc
        4: ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KHC"],  # Consumer Staples
        5: ["UNP", "HON", "UPS", "BA", "RTX", "CAT", "DE", "LMT", "GE", "MMM"],  # Industrials
        6: ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],  # Energy
        7: ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED"],  # Utilities
        8: ["PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "AVB"],  # Real Estate
        9: ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DD", "PPG", "VMC"],  # Materials
        10: ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA"],  # Comm Services
    }


def get_correlation_pairs() -> list:
    """
    Returns known highly correlated stock pairs
    Useful for cross-stock attention
    """
    return [
        # Tech semiconductors
        ("NVDA", "AMD"), ("NVDA", "AVGO"), ("NVDA", "INTC"), ("AMD", "INTC"),
        ("AMAT", "LRCX"), ("AMAT", "KLAC"), ("LRCX", "KLAC"),

        # Big tech
        ("AAPL", "MSFT"), ("GOOGL", "META"), ("AMZN", "MSFT"),

        # Banks
        ("JPM", "BAC"), ("JPM", "WFC"), ("BAC", "WFC"), ("GS", "MS"),

        # Oil majors
        ("XOM", "CVX"), ("COP", "EOG"), ("SLB", "HAL"),

        # Pharma
        ("PFE", "MRK"), ("JNJ", "PFE"), ("LLY", "BMY"),

        # Retail
        ("WMT", "TGT"), ("HD", "LOW"), ("COST", "WMT"),

        # Airlines
        ("DAL", "UAL"), ("DAL", "AAL"), ("UAL", "AAL"), ("LUV", "DAL"),

        # Auto
        ("GM", "F"), ("TSLA", "RIVN"), ("TSLA", "LCID"),

        # Cloud/Software
        ("CRM", "NOW"), ("ADBE", "CRM"), ("SNOW", "DDOG"),

        # Streaming
        ("NFLX", "DIS"), ("NFLX", "WBD"), ("DIS", "PARA"),

        # Social
        ("META", "SNAP"), ("META", "PINS"),

        # Crypto-adjacent
        ("COIN", "MARA"), ("COIN", "RIOT"), ("MARA", "RIOT"),
    ]


if __name__ == "__main__":
    # Test classification
    classifier = SectorClassifier()

    test_symbols = ["AAPL", "NVDA", "JPM", "XOM", "AMZN", "JNJ", "META", "TSLA"]

    print("Testing sector classification:\n")
    for symbol in test_symbols:
        sector_id, industry_id, sector_name, industry_name = classifier.classify(symbol)
        print(f"{symbol}:")
        print(f"  Sector: [{sector_id}] {sector_name}")
        print(f"  Industry: [{industry_id}] {industry_name}")
        print()

    print("\nAnchor stocks by sector:")
    for sector_id, stocks in get_anchor_stocks().items():
        sector_name = GICS_SECTORS[sector_id]["name"]
        print(f"  {sector_name}: {', '.join(stocks[:5])}...")
