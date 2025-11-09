"""
News Feed Module - Polygon Benzinga Integration
Handles real-time news headline ingestion for halted stocks using Polygon's Benzinga partnership.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .polygon_client import PolygonClient, BenzingaNews, PolygonRestClient

logger = logging.getLogger(__name__)


@dataclass
class NewsContext:
    """News context for a halted ticker"""

    ticker: str
    halt_time: datetime
    headlines: List[BenzingaNews]
    pre_halt_news: List[BenzingaNews]
    post_halt_news: List[BenzingaNews]
    catalyst_identified: bool = False
    primary_headline: Optional[BenzingaNews] = None


class NewsFeedProvider:
    """
    News feed provider using Polygon's Benzinga partnership.
    Fetches real-time news headlines for trading decisions.
    """

    def __init__(self, api_key: str):
        """
        Initialize news feed provider.

        Args:
            api_key: Polygon API key
        """
        self.api_key = api_key
        self.client = PolygonClient(api_key)
        self.rest_client = self.client.rest

        # Cache recent news to avoid duplicate API calls
        self.news_cache: Dict[str, List[BenzingaNews]] = {}
        self.cache_ttl_minutes = 5

        logger.info("NewsFeedProvider initialized with Polygon Benzinga integration")

    def get_news_for_ticker(
        self, ticker: str, lookback_hours: int = 24, limit: int = 50
    ) -> List[BenzingaNews]:
        """
        Get recent news headlines for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            lookback_hours: How many hours back to search
            limit: Maximum number of headlines

        Returns:
            List of BenzingaNews objects
        """
        # Check cache first
        cache_key = f"{ticker}_{lookback_hours}"
        if cache_key in self.news_cache:
            cached_news = self.news_cache[cache_key]
            if cached_news and len(cached_news) > 0:
                # Check if cache is still fresh
                latest_cached = cached_news[0].published_utc
                if (
                    datetime.utcnow() - latest_cached
                ).total_seconds() < self.cache_ttl_minutes * 60:
                    logger.debug(f"Returning cached news for {ticker}")
                    return cached_news

        # Fetch from Polygon
        start_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        news = self.rest_client.get_benzinga_news(
            ticker=ticker,
            published_utc_gte=start_time,
            order="desc",
            limit=limit,
            sort="published_utc",
        )

        # Update cache
        if news:
            self.news_cache[cache_key] = news
            logger.info(f"Retrieved {len(news)} news articles for {ticker}")
        else:
            logger.warning(f"No news found for {ticker}")

        return news

    def get_halt_context_news(
        self,
        ticker: str,
        halt_time: datetime,
        lookback_minutes: int = 60,
        lookahead_minutes: int = 5,
    ) -> NewsContext:
        """
        Get news context specifically for a halt event.

        Fetches news before and after the halt to identify catalysts.

        Args:
            ticker: Stock ticker that halted
            halt_time: Time when halt occurred
            lookback_minutes: Minutes to look back before halt
            lookahead_minutes: Minutes to look ahead after halt

        Returns:
            NewsContext object with categorized headlines
        """
        start_time = halt_time - timedelta(minutes=lookback_minutes)
        end_time = halt_time + timedelta(minutes=lookahead_minutes)

        # Fetch all news in the window
        all_news = self.rest_client.get_benzinga_news(
            ticker=ticker,
            published_utc_gte=start_time,
            published_utc_lte=end_time,
            order="desc",
            limit=50,
        )

        # Categorize by timing
        pre_halt = []
        post_halt = []

        for article in all_news:
            if article.published_utc <= halt_time:
                pre_halt.append(article)
            else:
                post_halt.append(article)

        # Sort by proximity to halt time
        all_news_sorted = sorted(
            all_news, key=lambda x: abs((x.published_utc - halt_time).total_seconds())
        )

        # Identify primary headline (closest to halt time)
        primary = all_news_sorted[0] if all_news_sorted else None

        context = NewsContext(
            ticker=ticker,
            halt_time=halt_time,
            headlines=all_news_sorted,
            pre_halt_news=pre_halt,
            post_halt_news=post_halt,
            catalyst_identified=len(all_news_sorted) > 0,
            primary_headline=primary,
        )

        logger.info(
            f"Halt context for {ticker}: {len(pre_halt)} pre-halt, "
            f"{len(post_halt)} post-halt headlines"
        )

        return context

    def get_breaking_news(
        self,
        tickers: Optional[List[str]] = None,
        last_n_minutes: int = 5,
        limit: int = 100,
    ) -> List[BenzingaNews]:
        """
        Get breaking news from the last N minutes.

        Args:
            tickers: Optional list of tickers to filter (None = all tickers)
            last_n_minutes: How many minutes back to look
            limit: Maximum number of articles

        Returns:
            List of recent BenzingaNews objects
        """
        start_time = datetime.utcnow() - timedelta(minutes=last_n_minutes)

        if tickers:
            # Fetch for specific tickers
            all_news = []
            for ticker in tickers:
                news = self.rest_client.get_benzinga_news(
                    ticker=ticker,
                    published_utc_gte=start_time,
                    order="desc",
                    limit=limit // len(tickers) if len(tickers) > 0 else limit,
                )
                all_news.extend(news)

            # Sort by time
            all_news.sort(key=lambda x: x.published_utc, reverse=True)
            return all_news[:limit]
        else:
            # Fetch all breaking news (no ticker filter)
            news = self.rest_client.get_benzinga_news(
                published_utc_gte=start_time, order="desc", limit=limit
            )
            return news

    def get_earnings_news(self, ticker: str, days_back: int = 7) -> List[BenzingaNews]:
        """
        Get earnings-related news for a ticker.

        Args:
            ticker: Stock ticker symbol
            days_back: How many days to look back

        Returns:
            List of earnings-related news
        """
        start_time = datetime.utcnow() - timedelta(days=days_back)

        all_news = self.rest_client.get_benzinga_news(
            ticker=ticker, published_utc_gte=start_time, order="desc", limit=100
        )

        # Filter for earnings-related keywords
        earnings_keywords = [
            "earnings",
            "revenue",
            "quarterly",
            "q1",
            "q2",
            "q3",
            "q4",
            "eps",
            "guidance",
            "forecast",
            "results",
        ]

        earnings_news = []
        for article in all_news:
            title_lower = article.title.lower()
            if any(keyword in title_lower for keyword in earnings_keywords):
                earnings_news.append(article)

        logger.info(
            f"Found {len(earnings_news)} earnings-related articles for {ticker}"
        )
        return earnings_news

    def get_fda_news(self, ticker: str, days_back: int = 7) -> List[BenzingaNews]:
        """
        Get FDA-related news for a ticker.

        Args:
            ticker: Stock ticker symbol
            days_back: How many days to look back

        Returns:
            List of FDA-related news
        """
        start_time = datetime.utcnow() - timedelta(days=days_back)

        all_news = self.rest_client.get_benzinga_news(
            ticker=ticker, published_utc_gte=start_time, order="desc", limit=100
        )

        # Filter for FDA-related keywords
        fda_keywords = [
            "fda",
            "approval",
            "clinical trial",
            "phase 1",
            "phase 2",
            "phase 3",
            "drug",
            "pdufa",
            "complete response letter",
            "crl",
            "orphan drug",
            "breakthrough therapy",
            "fast track",
            "priority review",
        ]

        fda_news = []
        for article in all_news:
            title_lower = article.title.lower()
            if any(keyword in title_lower for keyword in fda_keywords):
                fda_news.append(article)

        logger.info(f"Found {len(fda_news)} FDA-related articles for {ticker}")
        return fda_news

    def search_news_by_keyword(
        self, ticker: str, keywords: List[str], days_back: int = 7
    ) -> List[BenzingaNews]:
        """
        Search news by custom keywords.

        Args:
            ticker: Stock ticker symbol
            keywords: List of keywords to search for
            days_back: How many days to look back

        Returns:
            List of matching news articles
        """
        start_time = datetime.utcnow() - timedelta(days=days_back)

        all_news = self.rest_client.get_benzinga_news(
            ticker=ticker, published_utc_gte=start_time, order="desc", limit=100
        )

        matching_news = []
        keywords_lower = [k.lower() for k in keywords]

        for article in all_news:
            title_lower = article.title.lower()
            description_lower = (article.description or "").lower()

            # Check title and description
            if any(
                keyword in title_lower or keyword in description_lower
                for keyword in keywords_lower
            ):
                matching_news.append(article)

        logger.info(
            f"Found {len(matching_news)} articles matching keywords {keywords} for {ticker}"
        )
        return matching_news

    def has_recent_news(self, ticker: str, minutes: int = 30) -> bool:
        """
        Check if ticker has news in the last N minutes.

        Args:
            ticker: Stock ticker symbol
            minutes: Time window in minutes

        Returns:
            True if news exists, False otherwise
        """
        start_time = datetime.utcnow() - timedelta(minutes=minutes)

        news = self.rest_client.get_benzinga_news(
            ticker=ticker, published_utc_gte=start_time, limit=1
        )

        return len(news) > 0

    def get_news_summary(
        self, ticker: str, halt_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of news activity for a ticker.

        Args:
            ticker: Stock ticker symbol
            halt_time: Optional halt time for context

        Returns:
            Dictionary with news summary statistics
        """
        if halt_time:
            context = self.get_halt_context_news(ticker, halt_time)

            return {
                "ticker": ticker,
                "halt_time": halt_time.isoformat(),
                "total_headlines": len(context.headlines),
                "pre_halt_headlines": len(context.pre_halt_news),
                "post_halt_headlines": len(context.post_halt_news),
                "catalyst_identified": context.catalyst_identified,
                "primary_headline": context.primary_headline.title
                if context.primary_headline
                else None,
                "primary_published": context.primary_headline.published_utc.isoformat()
                if context.primary_headline
                else None,
            }
        else:
            # General summary
            news_24h = self.get_news_for_ticker(ticker, lookback_hours=24, limit=100)
            news_1h = self.get_news_for_ticker(ticker, lookback_hours=1, limit=50)

            return {
                "ticker": ticker,
                "headlines_24h": len(news_24h),
                "headlines_1h": len(news_1h),
                "latest_headline": news_24h[0].title if news_24h else None,
                "latest_published": news_24h[0].published_utc.isoformat()
                if news_24h
                else None,
            }

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear news cache.

        Args:
            ticker: Specific ticker to clear, or None for all
        """
        if ticker:
            # Clear all cache entries for this ticker
            keys_to_remove = [
                k for k in self.news_cache.keys() if k.startswith(f"{ticker}_")
            ]
            for key in keys_to_remove:
                self.news_cache.pop(key, None)
        else:
            self.news_cache.clear()

        logger.info(f"Cleared news cache{' for ' + ticker if ticker else ''}")

    def format_headline_for_ai(self, article: BenzingaNews) -> str:
        """
        Format a news article for AI consumption.

        Args:
            article: BenzingaNews object

        Returns:
            Formatted string for AI processing
        """
        parts = [
            f"Title: {article.title}",
            f"Published: {article.published_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Author: {article.author}",
            f"Publisher: {article.publisher}",
        ]

        if article.description:
            parts.append(f"Description: {article.description}")

        if article.keywords:
            parts.append(f"Keywords: {', '.join(article.keywords)}")

        return "\n".join(parts)

    def get_ai_ready_news_context(self, ticker: str, halt_time: datetime) -> str:
        """
        Get news context formatted for AI catalyst classification.

        Args:
            ticker: Stock ticker symbol
            halt_time: Time of halt

        Returns:
            Formatted string with all relevant news
        """
        context = self.get_halt_context_news(ticker, halt_time)

        if not context.headlines:
            return f"No news found for {ticker} around halt time."

        lines = [
            f"News Context for {ticker} (Halted at {halt_time.strftime('%Y-%m-%d %H:%M:%S UTC')})",
            "=" * 80,
            "",
        ]

        # Add primary headline first
        if context.primary_headline:
            lines.append("PRIMARY HEADLINE (Closest to Halt):")
            lines.append(self.format_headline_for_ai(context.primary_headline))
            lines.append("")

        # Add pre-halt news
        if context.pre_halt_news:
            lines.append(f"PRE-HALT NEWS ({len(context.pre_halt_news)} articles):")
            for i, article in enumerate(context.pre_halt_news[:3], 1):  # Limit to 3
                lines.append(f"\n{i}. {self.format_headline_for_ai(article)}")
            lines.append("")

        # Add post-halt news if any
        if context.post_halt_news:
            lines.append(f"POST-HALT NEWS ({len(context.post_halt_news)} articles):")
            for i, article in enumerate(context.post_halt_news[:2], 1):  # Limit to 2
                lines.append(f"\n{i}. {self.format_headline_for_ai(article)}")

        return "\n".join(lines)


# Factory function
def create_news_feed_provider(api_key: str) -> NewsFeedProvider:
    """
    Factory function to create NewsFeedProvider.

    Args:
        api_key: Polygon API key

    Returns:
        Configured NewsFeedProvider instance
    """
    return NewsFeedProvider(api_key)
