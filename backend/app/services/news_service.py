"""
News service for fetching real-time market news.
Uses multiple sources: Google News RSS, NewsAPI, and fallback data.
"""
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
from bs4 import BeautifulSoup

from app.core.redis import get_cache, set_cache
import json


# Cache TTL
NEWS_CACHE_TTL = 300  # 5 minutes

# NIFTY 50 related keywords
NIFTY_KEYWORDS = [
    "NIFTY", "Sensex", "NSE", "BSE", "Indian stock", "India market",
    "RBI", "Reserve Bank", "FII", "DII", "rupee", "INR"
]

# Top NIFTY 50 stock names for matching
NIFTY50_STOCKS = {
    "RELIANCE": ["Reliance", "RIL", "Jio", "Mukesh Ambani"],
    "TCS": ["TCS", "Tata Consultancy", "Tata Tech"],
    "HDFCBANK": ["HDFC Bank", "HDFC"],
    "INFY": ["Infosys", "INFY"],
    "ICICIBANK": ["ICICI Bank", "ICICI"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"],
    "BHARTIARTL": ["Bharti Airtel", "Airtel"],
    "SBIN": ["SBI", "State Bank of India"],
    "BAJFINANCE": ["Bajaj Finance", "Bajaj Finserv"],
    "ITC": ["ITC"],
    "KOTAKBANK": ["Kotak Mahindra", "Kotak Bank"],
    "LT": ["Larsen & Toubro", "L&T"],
    "AXISBANK": ["Axis Bank"],
    "ASIANPAINT": ["Asian Paints"],
    "MARUTI": ["Maruti Suzuki", "Maruti"],
    "TATAMOTORS": ["Tata Motors"],
    "SUNPHARMA": ["Sun Pharma", "Sun Pharmaceutical"],
    "WIPRO": ["Wipro"],
    "HCLTECH": ["HCL Tech", "HCL Technologies"],
    "ULTRACEMCO": ["UltraTech Cement", "UltraTech"],
}


async def fetch_google_news_rss(query: str = "NIFTY 50") -> List[Dict[str, Any]]:
    """Fetch news from Google News RSS feed."""
    try:
        url = f"https://news.google.com/rss/search?q={query}+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    print(f"❌ Google News RSS failed: {response.status}")
                    return []
                
                content = await response.text()
        
        feed = feedparser.parse(content)
        
        news_items = []
        for entry in feed.entries[:20]:  # Limit to 20 items
            # Clean HTML from title and description
            title = BeautifulSoup(entry.get('title', ''), 'html.parser').get_text()
            summary = BeautifulSoup(entry.get('summary', ''), 'html.parser').get_text()
            
            # Extract source from title (usually after the last " - ")
            source = "Google News"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                if len(parts) > 1:
                    title = parts[0]
                    source = parts[1]
            
            # Parse published date
            published = entry.get('published', '')
            try:
                pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
            except:
                pub_date = datetime.now()
            
            news_items.append({
                "id": entry.get('id', str(hash(title)))[:16],
                "title": title,
                "summary": summary[:300] if summary else title,
                "source": source,
                "url": entry.get('link', ''),
                "publishedAt": pub_date.isoformat(),
                "sentiment": analyze_sentiment(title + " " + summary),
                "impact": analyze_impact(title + " " + summary),
                "category": categorize_news(title + " " + summary),
                "relatedStocks": find_related_stocks(title + " " + summary),
            })
        
        print(f"✅ Fetched {len(news_items)} news from Google News RSS")
        return news_items
        
    except Exception as e:
        print(f"❌ Google News RSS error: {str(e)}")
        return []


async def fetch_economic_times_rss() -> List[Dict[str, Any]]:
    """Fetch news from Economic Times RSS feed."""
    try:
        url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    print(f"❌ ET RSS failed: {response.status}")
                    return []
                
                content = await response.text()
        
        feed = feedparser.parse(content)
        
        news_items = []
        for entry in feed.entries[:15]:
            title = BeautifulSoup(entry.get('title', ''), 'html.parser').get_text()
            summary = BeautifulSoup(entry.get('summary', ''), 'html.parser').get_text()
            
            published = entry.get('published', '')
            try:
                pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
            except:
                try:
                    pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                except:
                    pub_date = datetime.now()
            
            news_items.append({
                "id": entry.get('id', str(hash(title)))[:16],
                "title": title,
                "summary": summary[:300] if summary else title,
                "source": "Economic Times",
                "url": entry.get('link', ''),
                "publishedAt": pub_date.isoformat(),
                "sentiment": analyze_sentiment(title + " " + summary),
                "impact": analyze_impact(title + " " + summary),
                "category": categorize_news(title + " " + summary),
                "relatedStocks": find_related_stocks(title + " " + summary),
            })
        
        print(f"✅ Fetched {len(news_items)} news from Economic Times RSS")
        return news_items
        
    except Exception as e:
        print(f"❌ ET RSS error: {str(e)}")
        return []


async def fetch_moneycontrol_rss() -> List[Dict[str, Any]]:
    """Fetch news from Moneycontrol RSS feed."""
    try:
        url = "https://www.moneycontrol.com/rss/marketreports.xml"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    print(f"❌ Moneycontrol RSS failed: {response.status}")
                    return []
                
                content = await response.text()
        
        feed = feedparser.parse(content)
        
        news_items = []
        for entry in feed.entries[:15]:
            title = BeautifulSoup(entry.get('title', ''), 'html.parser').get_text()
            summary = BeautifulSoup(entry.get('description', ''), 'html.parser').get_text()
            
            published = entry.get('published', '')
            try:
                pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
            except:
                try:
                    pub_date = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                except:
                    pub_date = datetime.now()
            
            news_items.append({
                "id": entry.get('id', str(hash(title)))[:16],
                "title": title,
                "summary": summary[:300] if summary else title,
                "source": "Moneycontrol",
                "url": entry.get('link', ''),
                "publishedAt": pub_date.isoformat(),
                "sentiment": analyze_sentiment(title + " " + summary),
                "impact": analyze_impact(title + " " + summary),
                "category": categorize_news(title + " " + summary),
                "relatedStocks": find_related_stocks(title + " " + summary),
            })
        
        print(f"✅ Fetched {len(news_items)} news from Moneycontrol RSS")
        return news_items
        
    except Exception as e:
        print(f"❌ Moneycontrol RSS error: {str(e)}")
        return []


def analyze_sentiment(text: str) -> str:
    """Simple keyword-based sentiment analysis."""
    text_lower = text.lower()
    
    positive_words = [
        "surge", "gain", "rise", "rally", "bullish", "growth", "profit", "up",
        "jump", "soar", "boost", "recovery", "positive", "strong", "buy",
        "upgrade", "outperform", "record high", "breakout", "support"
    ]
    
    negative_words = [
        "fall", "drop", "decline", "crash", "bearish", "loss", "down",
        "plunge", "sink", "slump", "negative", "weak", "sell", "cut",
        "downgrade", "underperform", "record low", "breakdown", "resistance",
        "fear", "concern", "risk", "tension", "crisis"
    ]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count + 1:
        return "positive"
    elif neg_count > pos_count + 1:
        return "negative"
    return "neutral"


def analyze_impact(text: str) -> str:
    """Analyze the potential market impact of news."""
    text_lower = text.lower()
    
    high_impact_words = [
        "rbi", "reserve bank", "policy", "rate", "inflation", "gdp",
        "fii", "dii", "billion", "trillion", "major", "significant",
        "breaking", "urgent", "crisis", "global", "fed", "us market"
    ]
    
    medium_impact_words = [
        "quarterly", "results", "earnings", "sector", "industry",
        "merger", "acquisition", "investment", "analyst", "rating"
    ]
    
    high_count = sum(1 for word in high_impact_words if word in text_lower)
    medium_count = sum(1 for word in medium_impact_words if word in text_lower)
    
    if high_count >= 2:
        return "high"
    elif high_count >= 1 or medium_count >= 2:
        return "medium"
    return "low"


def categorize_news(text: str) -> str:
    """Categorize news by topic."""
    text_lower = text.lower()
    
    categories = {
        "Monetary Policy": ["rbi", "reserve bank", "repo rate", "interest rate", "monetary policy", "mpc"],
        "FII Activity": ["fii", "foreign institutional", "dii", "domestic institutional"],
        "Global Markets": ["us market", "fed", "global", "asia", "europe", "dow", "nasdaq", "s&p"],
        "Commodities": ["oil", "gold", "crude", "commodity", "metal"],
        "Economy": ["gdp", "inflation", "growth", "economic", "fiscal", "budget"],
        "Corporate News": ["quarterly", "results", "earnings", "merger", "acquisition", "ipo"],
        "Sector Update": ["it sector", "banking", "pharma", "auto", "fmcg", "realty"],
        "Technical": ["support", "resistance", "breakout", "trend", "chart"],
    }
    
    for category, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            return category
    
    return "Market Update"


def find_related_stocks(text: str) -> List[str]:
    """Find NIFTY 50 stocks mentioned in the text."""
    text_upper = text.upper()
    related = []
    
    for symbol, keywords in NIFTY50_STOCKS.items():
        for keyword in keywords:
            if keyword.upper() in text_upper:
                related.append(symbol)
                break
    
    return list(set(related))


async def get_market_news(watchlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Fetch market news from multiple sources.
    Prioritizes news related to watchlist stocks.
    """
    # Check cache first
    cache_key = "market_news:all"
    cached = await get_cache(cache_key)
    if cached:
        news = json.loads(cached)
        if watchlist:
            news = prioritize_watchlist_news(news, watchlist)
        return news
    
    # Fetch from multiple sources in parallel
    results = await asyncio.gather(
        fetch_google_news_rss("NIFTY 50 stock market India"),
        fetch_economic_times_rss(),
        fetch_moneycontrol_rss(),
        return_exceptions=True
    )
    
    all_news = []
    for result in results:
        if isinstance(result, list):
            all_news.extend(result)
    
    # Remove duplicates based on title similarity
    seen_titles = set()
    unique_news = []
    for item in all_news:
        title_key = item['title'][:50].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(item)
    
    # Sort by published date (most recent first)
    unique_news.sort(key=lambda x: x['publishedAt'], reverse=True)
    
    # Limit to 30 items
    unique_news = unique_news[:30]
    
    # Cache the results
    if unique_news:
        await set_cache(cache_key, json.dumps(unique_news), expire=NEWS_CACHE_TTL)
    
    # Prioritize watchlist stocks if provided
    if watchlist:
        unique_news = prioritize_watchlist_news(unique_news, watchlist)
    
    return unique_news


def prioritize_watchlist_news(news: List[Dict[str, Any]], watchlist: List[str]) -> List[Dict[str, Any]]:
    """Prioritize news related to watchlist stocks."""
    # Clean up watchlist symbols (remove .NS suffix)
    clean_watchlist = [s.replace(".NS", "").replace("^", "") for s in watchlist]
    
    watchlist_news = []
    other_news = []
    
    for item in news:
        related = item.get('relatedStocks', [])
        is_watchlist = any(stock in clean_watchlist for stock in related)
        
        if is_watchlist:
            item['isWatchlist'] = True
            watchlist_news.append(item)
        else:
            item['isWatchlist'] = False
            other_news.append(item)
    
    # Return watchlist news first, then other news
    return watchlist_news + other_news


async def get_news_with_fallback(watchlist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get news - returns empty list if APIs fail (no fake data in production)."""
    news = await get_market_news(watchlist)
    
    if not news:
        print("⚠️ No news available from any source")
        return []
    
    return news
