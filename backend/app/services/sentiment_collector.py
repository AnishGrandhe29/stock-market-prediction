"""
Sentiment collection service.
Fetches and analyzes sentiment from Google News RSS and Reddit.
"""
import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import feedparser
import json
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.stock import SentimentScore
from app.config import settings


# Keywords for NIFTY 50 related news
NIFTY_KEYWORDS = [
    "nifty", "nifty 50", "sensex", "indian stock market",
    "nse", "bse", "indian market", "dalal street",
]

# Sentiment words (simple rule-based for fallback)
POSITIVE_WORDS = [
    "surge", "rally", "gain", "rise", "jump", "soar", "bull", "bullish",
    "profit", "growth", "strong", "positive", "up", "high", "record",
]
NEGATIVE_WORDS = [
    "fall", "drop", "crash", "decline", "loss", "bear", "bearish",
    "weak", "negative", "down", "low", "slump", "tumble", "plunge",
]


async def fetch_google_news_sentiment(symbol: str = "^NSEI") -> Dict:
    """
    Fetch news from Google News RSS and compute sentiment.
    """
    # Google News RSS feed for Indian stock market
    rss_url = "https://news.google.com/rss/search?q=nifty+50+indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
    
    loop = asyncio.get_event_loop()
    feed = await loop.run_in_executor(None, lambda: feedparser.parse(rss_url))
    
    if not feed.entries:
        return {
            "sentiment": 0.0,
            "count": 0,
            "headlines": [],
        }
    
    headlines = []
    sentiments = []
    
    for entry in feed.entries[:20]:  # Top 20 headlines
        title = entry.get('title', '')
        headlines.append(title)
        
        # Simple rule-based sentiment
        title_lower = title.lower()
        score = compute_simple_sentiment(title_lower)
        sentiments.append(score)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    
    return {
        "sentiment": avg_sentiment,
        "count": len(headlines),
        "headlines": headlines[:5],  # Return top 5
    }


async def fetch_reddit_sentiment(symbol: str = "^NSEI") -> Dict:
    """
    Fetch sentiment from Reddit (r/IndiaInvestments, r/IndianStreetBets).
    Uses PRAW if credentials are available, otherwise returns neutral.
    """
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        return {
            "sentiment": 0.0,
            "count": 0,
            "posts": [],
        }
    
    try:
        import praw
        
        loop = asyncio.get_event_loop()
        
        def get_reddit_posts():
            reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
            )
            
            posts = []
            subreddits = ["IndiaInvestments", "IndianStreetBets"]
            
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for post in subreddit.hot(limit=10):
                        posts.append({
                            "title": post.title,
                            "score": post.score,
                            "subreddit": sub_name,
                        })
                except Exception:
                    continue
            
            return posts
        
        posts = await loop.run_in_executor(None, get_reddit_posts)
        
        if not posts:
            return {"sentiment": 0.0, "count": 0, "posts": []}
        
        sentiments = []
        for post in posts:
            score = compute_simple_sentiment(post['title'].lower())
            # Weight by Reddit score
            weight = min(post['score'] / 100, 2.0)  # Cap at 2x
            sentiments.append(score * (1 + weight * 0.1))
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        return {
            "sentiment": max(-1, min(1, avg_sentiment)),  # Clamp to [-1, 1]
            "count": len(posts),
            "posts": [p['title'] for p in posts[:5]],
        }
        
    except ImportError:
        return {"sentiment": 0.0, "count": 0, "posts": []}
    except Exception as e:
        print(f"Reddit error: {e}")
        return {"sentiment": 0.0, "count": 0, "posts": []}


def compute_simple_sentiment(text: str) -> float:
    """
    Simple rule-based sentiment scoring.
    Returns value between -1 (negative) and 1 (positive).
    """
    positive_count = sum(1 for word in POSITIVE_WORDS if word in text)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in text)
    
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    return (positive_count - negative_count) / total


async def collect_daily_sentiment(
    symbol: str,
    db: AsyncSession
) -> SentimentScore:
    """
    Collect and store daily sentiment from all sources.
    """
    today = date.today()
    
    # Check if already collected today
    existing = await db.execute(
        select(SentimentScore).where(
            SentimentScore.symbol == symbol,
            SentimentScore.date == today
        )
    )
    if existing.scalar_one_or_none():
        return existing.scalar_one()
    
    # Fetch from all sources
    news_result = await fetch_google_news_sentiment(symbol)
    reddit_result = await fetch_reddit_sentiment(symbol)
    
    # Combine sentiments (weighted average)
    news_weight = 0.6
    reddit_weight = 0.4
    
    combined = (
        news_result['sentiment'] * news_weight +
        reddit_result['sentiment'] * reddit_weight
    )
    
    # Store in database
    sentiment = SentimentScore(
        symbol=symbol,
        date=today,
        news_sentiment=news_result['sentiment'],
        reddit_sentiment=reddit_result['sentiment'],
        combined_sentiment=combined,
        news_count=news_result['count'],
        reddit_count=reddit_result['count'],
        top_headlines=json.dumps(news_result['headlines']),
    )
    
    db.add(sentiment)
    await db.commit()
    await db.refresh(sentiment)
    
    return sentiment
