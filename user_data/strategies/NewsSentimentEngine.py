from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import math
import re
from typing import Any, Iterable, Optional, Sequence

import requests


@dataclass
class ScoredArticle:
    title: str
    source: str
    url: str
    published_at: Optional[datetime]
    sentiment: float
    recency_weight: float
    source_weight: float
    event_score: float
    event_tags: list[str] = field(default_factory=list)
    matched_aliases: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsSignal:
    asset: str
    checked_at: datetime
    article_count: int
    sentiment_score: float
    confidence: float
    impact_score: float
    block_entries: bool
    reduce_stake: bool
    recommended_stake_multiplier: float
    backend: str
    reasons: list[str] = field(default_factory=list)
    articles: list[ScoredArticle] = field(default_factory=list)
    fallback_used: bool = False
    backend_available: bool = True
    stale_cache: bool = False


class SentimentBackendBase:
    name = "none"

    def score(self, text: str) -> Optional[float]:
        raise NotImplementedError


class VaderSentimentBackend(SentimentBackendBase):
    name = "vader"

    def __init__(self, auto_download: bool = True) -> None:
        self.auto_download = auto_download
        self._analyzer = None

    def _ensure_analyzer(self):
        if self._analyzer is not None:
            return self._analyzer

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except Exception:
            return None

        try:
            self._analyzer = SentimentIntensityAnalyzer()
            return self._analyzer
        except LookupError:
            if not self.auto_download:
                return None
            try:
                import nltk

                nltk.download("vader_lexicon", quiet=True)
                self._analyzer = SentimentIntensityAnalyzer()
                return self._analyzer
            except Exception:
                return None

    def score(self, text: str) -> Optional[float]:
        analyzer = self._ensure_analyzer()
        if analyzer is None:
            return None
        return float(analyzer.polarity_scores(text or "").get("compound", 0.0))


class RobertaSentimentBackend(SentimentBackendBase):
    name = "roberta"

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ) -> None:
        self.model_name = model_name
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            return self._pipeline
        except Exception:
            return None

    @staticmethod
    def _label_to_score(label: str, score: float) -> float:
        normalized = (label or "").strip().lower()
        if normalized in {"positive", "pos", "label_2"}:
            return float(score)
        if normalized in {"negative", "neg", "label_0"}:
            return -float(score)
        return 0.0

    def score(self, text: str) -> Optional[float]:
        sentiment_pipeline = self._ensure_pipeline()
        if sentiment_pipeline is None:
            return None
        try:
            result = sentiment_pipeline((text or "")[:512])[0]
            return self._label_to_score(
                str(result.get("label", "")), float(result.get("score", 0.0))
            )
        except Exception:
            return None


class HybridSentimentBackend(SentimentBackendBase):
    name = "hybrid"

    def __init__(self, backends: Sequence[SentimentBackendBase]) -> None:
        self.backends = list(backends)

    def score(self, text: str) -> Optional[float]:
        scores = []
        for backend in self.backends:
            try:
                score = backend.score(text)
            except Exception:
                score = None
            if score is not None:
                scores.append(float(score))
        if not scores:
            return None
        return float(sum(scores) / len(scores))


class NewsSentimentEngine:
    """
    Fetch + score + cache engine for event/news-aware trading filters.

    Typical usage from a Freqtrade strategy:
      * Instantiate once in bot_start()
      * Prefetch in bot_loop_start()
      * Read cached signals in populate_entry_trend(), confirm_trade_entry(), etc.
    """

    def __init__(
        self,
        api_key: str,
        logger: Optional[logging.Logger] = None,
        backend: str = "hybrid",
        roberta_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        lookback_hours: int = 2,
        cache_minutes: int = 15,
        page_size: int = 10,
        min_articles_to_block: int = 2,
        min_articles_to_reduce: int = 1,
        block_abs_sentiment: float = 0.30,
        reduce_abs_sentiment: float = 0.18,
        min_impact_to_block: float = 0.25,
        min_stake_multiplier: float = 0.40,
        max_stake_multiplier: float = 1.10,
        stake_reduction_scale: float = 0.50,
        positive_boost_threshold: float = 0.45,
        min_confidence_to_act: float = 0.35,
        max_fallback_minutes: int = 60,
        request_timeout: int = 10,
        language: str = "en",
        source_weights: Optional[dict[str, float]] = None,
        trusted_positive_sources: Optional[Sequence[str]] = None,
    ) -> None:
        self.api_key = api_key or ""
        self.logger = logger or logging.getLogger(__name__)
        self.backend_name = backend.lower().strip()
        self.roberta_model = roberta_model
        self.lookback_hours = max(1, int(lookback_hours))
        self.cache_minutes = max(1, int(cache_minutes))
        self.page_size = max(1, min(int(page_size), 100))
        self.min_articles_to_block = max(1, int(min_articles_to_block))
        self.min_articles_to_reduce = max(1, int(min_articles_to_reduce))
        self.block_abs_sentiment = float(block_abs_sentiment)
        self.reduce_abs_sentiment = float(reduce_abs_sentiment)
        self.min_impact_to_block = float(min_impact_to_block)
        self.min_stake_multiplier = float(min_stake_multiplier)
        self.max_stake_multiplier = float(max_stake_multiplier)
        self.stake_reduction_scale = float(stake_reduction_scale)
        self.positive_boost_threshold = float(positive_boost_threshold)
        self.min_confidence_to_act = float(min_confidence_to_act)
        self.max_fallback_minutes = max(1, int(max_fallback_minutes))
        self.request_timeout = int(request_timeout)
        self.language = language
        self.news_api_url = "https://newsapi.org/v2/everything"
        self._cache: dict[str, NewsSignal] = {}
        self._backend = self._build_backend()
        self._backend_ready: Optional[bool] = None
        self._default_source_weight = 0.60
        self.source_weights = {
            self._normalize_source_name(key): float(value)
            for key, value in (source_weights or self._default_source_weights()).items()
        }
        self.trusted_positive_sources = {
            self._normalize_source_name(source)
            for source in (
                trusted_positive_sources
                or (
                    "cointelegraph",
                    "the block",
                    "decrypt",
                    "coindesk",
                    "binance",
                    "coinbase",
                    "kraken blog",
                    "okx",
                )
            )
        }

    def _build_backend(self) -> SentimentBackendBase:
        if self.backend_name == "none":
            return HybridSentimentBackend([])
        if self.backend_name == "vader":
            return VaderSentimentBackend()
        if self.backend_name == "roberta":
            return RobertaSentimentBackend(model_name=self.roberta_model)
        return HybridSentimentBackend(
            [
                VaderSentimentBackend(),
                RobertaSentimentBackend(model_name=self.roberta_model),
            ]
        )

    def backend_ready(self) -> bool:
        if self._backend_ready is None:
            sample_score = self._backend.score(
                "Bitcoin ETF approval improves market outlook."
            )
            self._backend_ready = sample_score is not None
        return self._backend_ready

    def backend_status(self) -> dict[str, Any]:
        ready = self.backend_ready()
        return {
            "backend": self._backend.name,
            "ready": ready,
            "degraded": not ready,
        }

    def peek_cached_signal(self, asset: str) -> Optional[NewsSignal]:
        return self._cache.get(asset.upper())

    def get_signal(
        self,
        asset: str,
        current_time: datetime,
        aliases: Optional[Sequence[str]] = None,
    ) -> NewsSignal:
        asset = asset.upper()
        current_time = self._ensure_utc(current_time)

        cached = self._cache.get(asset)
        if cached and current_time <= cached.checked_at + timedelta(
            minutes=self.cache_minutes
        ):
            return cached

        if not self.api_key:
            signal = NewsSignal(
                asset=asset,
                checked_at=current_time,
                article_count=0,
                sentiment_score=0.0,
                confidence=0.0,
                impact_score=0.0,
                block_entries=False,
                reduce_stake=False,
                recommended_stake_multiplier=1.0,
                backend="disabled",
                reasons=["No NewsAPI key configured."],
                backend_available=False,
            )
            self._cache[asset] = signal
            return signal

        if not self.backend_ready():
            signal = NewsSignal(
                asset=asset,
                checked_at=current_time,
                article_count=0,
                sentiment_score=0.0,
                confidence=0.0,
                impact_score=0.0,
                block_entries=False,
                reduce_stake=False,
                recommended_stake_multiplier=1.0,
                backend=self._backend.name,
                reasons=["Sentiment backend unavailable; news filter disabled."],
                fallback_used=True,
                backend_available=False,
            )
            self._cache[asset] = signal
            return signal

        query = self._build_query(asset, aliases or [])
        try:
            raw_articles = self._fetch_articles(query, current_time)
            signal = self._aggregate_signal(asset, current_time, raw_articles, aliases or [])
            self._cache[asset] = signal
            return signal
        except requests.exceptions.RequestException as exc:
            self.logger.warning("News fetch failed for %s: %s", asset, exc)
        except Exception as exc:
            self.logger.error(
                "Unexpected news engine error for %s: %s", asset, exc, exc_info=True
            )

        if cached is not None and current_time <= cached.checked_at + timedelta(
            minutes=self.max_fallback_minutes
        ):
            fallback = NewsSignal(
                **{
                    **cached.__dict__,
                    "fallback_used": True,
                    "stale_cache": False,
                    "reasons": [
                        *cached.reasons,
                        "Using cached signal after fetch failure.",
                    ],
                }
            )
            self._cache[asset] = fallback
            return fallback

        signal = NewsSignal(
            asset=asset,
            checked_at=current_time,
            article_count=0,
            sentiment_score=0.0,
            confidence=0.0,
            impact_score=0.0,
            block_entries=False,
            reduce_stake=False,
            recommended_stake_multiplier=1.0,
            backend=self._backend.name,
            reasons=["News fetch failed and no fresh cached signal was available."],
            fallback_used=True,
            backend_available=self.backend_ready(),
            stale_cache=cached is not None,
        )
        self._cache[asset] = signal
        return signal

    def _fetch_articles(
        self, query: str, current_time: datetime
    ) -> list[dict[str, Any]]:
        from_time = (current_time - timedelta(hours=self.lookback_hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        params = {
            "q": query,
            "from": from_time,
            "sortBy": "publishedAt",
            "language": self.language,
            "pageSize": self.page_size,
            "apiKey": self.api_key,
        }
        response = requests.get(
            self.news_api_url, params=params, timeout=self.request_timeout
        )
        if response.status_code == 429:
            raise requests.exceptions.RequestException("NewsAPI rate limit reached.")
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise requests.exceptions.RequestException(
                f"Unexpected NewsAPI payload status: {payload.get('status')}"
            )
        return list(payload.get("articles", []))

    def _build_query(self, asset: str, aliases: Sequence[str]) -> str:
        query_terms = [asset, *aliases]
        unique_terms = []
        seen = set()
        for term in query_terms:
            cleaned = str(term).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_terms.append(cleaned)

        asset_query = " OR ".join(f'"{term}"' for term in unique_terms) or f'"{asset}"'
        context = "(crypto OR cryptocurrency OR token OR coin OR blockchain OR defi OR exchange)"
        exclusions = (
            "NOT (sport OR soccer OR football OR baseball OR movie OR celebrity)"
        )
        return f"({asset_query}) AND {context} AND {exclusions}"

    def _aggregate_signal(
        self,
        asset: str,
        current_time: datetime,
        raw_articles: Iterable[dict[str, Any]],
        aliases: Sequence[str],
    ) -> NewsSignal:
        scored_articles: list[ScoredArticle] = []
        article_keys = set()
        match_terms = self._build_match_terms(asset, aliases)

        for article in raw_articles:
            title = str(article.get("title") or "").strip()
            description = str(article.get("description") or "").strip()
            text = title
            if description:
                text = f"{title}. {description}" if title else description
            if not text:
                continue

            matched_aliases = self._match_aliases(text, match_terms)
            if not matched_aliases:
                continue

            dedupe_key = self._build_article_key(article, title, description)
            if dedupe_key in article_keys:
                continue
            article_keys.add(dedupe_key)

            sentiment = self._backend.score(text)
            if sentiment is None:
                continue

            published_at = self._parse_published_at(article.get("publishedAt"))
            recency_weight = self._recency_weight(current_time, published_at)
            source_name = str((article.get("source") or {}).get("name") or "")
            source_weight = self._get_source_weight(source_name)
            event_tags, event_score = self._extract_event_signal(text)
            scored_articles.append(
                ScoredArticle(
                    title=title,
                    source=source_name,
                    url=str(article.get("url") or ""),
                    published_at=published_at,
                    sentiment=float(sentiment),
                    recency_weight=recency_weight,
                    source_weight=source_weight,
                    event_score=event_score,
                    event_tags=event_tags,
                    matched_aliases=matched_aliases,
                    raw=article,
                )
            )

        if not scored_articles:
            return NewsSignal(
                asset=asset,
                checked_at=current_time,
                article_count=0,
                sentiment_score=0.0,
                confidence=0.0,
                impact_score=0.0,
                block_entries=False,
                reduce_stake=False,
                recommended_stake_multiplier=1.0,
                backend=self._backend.name,
                reasons=["No recent relevant articles with a working sentiment backend were found."],
                backend_available=True,
            )

        total_weight = (
            sum(article.recency_weight * article.source_weight for article in scored_articles)
            or 1.0
        )
        sentiment_score = (
            sum(
                (article.sentiment + article.event_score) * 0.5 * article.recency_weight * article.source_weight
                for article in scored_articles
            )
            / total_weight
        )
        avg_abs_sentiment = (
            sum(
                max(abs(article.sentiment), abs(article.event_score)) * article.recency_weight * article.source_weight
                for article in scored_articles
            )
            / total_weight
        )
        trusted_article_count = sum(
            1
            for article in scored_articles
            if self._normalize_source_name(article.source) in self.trusted_positive_sources
        )

        article_count = len(scored_articles)
        coverage = min(1.0, article_count / max(self.min_articles_to_block, 1))
        confidence = min(
            1.0,
            max(
                0.0,
                coverage * max(avg_abs_sentiment, 0.25) * min(1.0, total_weight / max(article_count, 1)),
            ),
        )
        count_factor = min(
            1.0, math.log1p(article_count) / math.log1p(max(self.page_size, 2))
        )
        impact_score = min(1.0, count_factor * max(avg_abs_sentiment, 0.10))

        block_entries = (
            article_count >= self.min_articles_to_block
            and abs(sentiment_score) >= self.block_abs_sentiment
            and impact_score >= self.min_impact_to_block
            and confidence >= self.min_confidence_to_act
        )
        reduce_stake = (
            not block_entries
            and article_count >= self.min_articles_to_reduce
            and sentiment_score < 0
            and abs(sentiment_score) >= self.reduce_abs_sentiment
            and confidence >= self.min_confidence_to_act
        )

        stake_multiplier = 1.0
        reasons = []

        if block_entries:
            stake_multiplier = 0.0
            reasons.append(
                f"Blocking entries: {article_count} articles, sentiment={sentiment_score:.3f}, impact={impact_score:.3f}."
            )
        elif reduce_stake:
            reduction = min(0.60, abs(sentiment_score) * self.stake_reduction_scale)
            stake_multiplier = max(self.min_stake_multiplier, 1.0 - reduction)
            reasons.append(
                f"Reducing stake: negative news sentiment={sentiment_score:.3f}, multiplier={stake_multiplier:.2f}."
            )
        elif (
            sentiment_score >= self.positive_boost_threshold
            and article_count >= self.min_articles_to_reduce
            and trusted_article_count > 0
            and any(article.event_score > 0.35 for article in scored_articles)
            and confidence >= max(self.min_confidence_to_act, 0.50)
        ):
            boost = min(0.10, sentiment_score * 0.10)
            stake_multiplier = min(self.max_stake_multiplier, 1.0 + boost)
            reasons.append(
                f"Trusted positive event coverage detected: sentiment={sentiment_score:.3f}, multiplier={stake_multiplier:.2f}."
            )
        else:
            reasons.append(
                f"Neutral/moderate news sentiment: {article_count} articles, sentiment={sentiment_score:.3f}."
            )
            if confidence < self.min_confidence_to_act:
                reasons.append("Confidence below action threshold; signal is advisory only.")

        return NewsSignal(
            asset=asset,
            checked_at=current_time,
            article_count=article_count,
            sentiment_score=float(sentiment_score),
            confidence=float(confidence),
            impact_score=float(impact_score),
            block_entries=block_entries,
            reduce_stake=reduce_stake,
            recommended_stake_multiplier=float(stake_multiplier),
            backend=self._backend.name,
            reasons=reasons,
            articles=scored_articles,
            backend_available=True,
        )

    def _recency_weight(
        self, current_time: datetime, published_at: Optional[datetime]
    ) -> float:
        if published_at is None:
            return 0.50
        age_hours = max(0.0, (current_time - published_at).total_seconds() / 3600.0)
        if age_hours >= self.lookback_hours:
            return 0.25
        linear = 1.0 - (age_hours / self.lookback_hours)
        return max(0.25, linear)

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_published_at(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except Exception:
            return None

    @staticmethod
    def _normalize_source_name(source: str) -> str:
        return re.sub(r"\s+", " ", (source or "").strip().lower())

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()

    def _default_source_weights(self) -> dict[str, float]:
        return {
            "coindesk": 1.0,
            "cointelegraph": 0.95,
            "the block": 1.0,
            "decrypt": 0.9,
            "bloomberg": 1.0,
            "reuters": 1.0,
            "financial times": 1.0,
            "binance": 0.95,
            "coinbase": 0.95,
            "kraken blog": 0.90,
            "okx": 0.90,
            "yahoo finance": 0.75,
        }

    def _build_match_terms(self, asset: str, aliases: Sequence[str]) -> list[str]:
        terms = [asset, *aliases]
        normalized_terms = []
        seen = set()
        for term in terms:
            normalized = self._normalize_text(str(term))
            if not normalized or len(normalized) < 2:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            normalized_terms.append(normalized)
        return normalized_terms or [self._normalize_text(asset)]

    def _match_aliases(self, text: str, match_terms: Sequence[str]) -> list[str]:
        normalized_text = f" {self._normalize_text(text)} "
        matches = []
        for term in match_terms:
            if f" {term} " in normalized_text:
                matches.append(term)
        return matches

    def _build_article_key(self, article: dict[str, Any], title: str, description: str) -> str:
        url = str(article.get("url") or "").strip().lower()
        if url:
            return url
        return self._normalize_text(f"{title} {description}")

    def _get_source_weight(self, source_name: str) -> float:
        normalized = self._normalize_source_name(source_name)
        if not normalized:
            return self._default_source_weight
        if normalized in self.source_weights:
            return self.source_weights[normalized]
        for known_source, weight in self.source_weights.items():
            if known_source and known_source in normalized:
                return weight
        return self._default_source_weight

    def _extract_event_signal(self, text: str) -> tuple[list[str], float]:
        normalized = self._normalize_text(text)
        event_patterns = {
            "hack": (-0.90, ("hack", "exploit", "breach", "stolen", "drain")),
            "outage": (-0.75, ("outage", "halt", "downtime", "suspended withdrawals")),
            "lawsuit": (-0.70, ("lawsuit", "sued", "sec", "charges", "investigation")),
            "delisting": (-0.80, ("delist", "delisting", "removed from exchange")),
            "liquidation": (-0.55, ("liquidation", "liquidations", "forced selling")),
            "etf_approval": (0.85, ("etf approved", "etf approval", "sec approves")),
            "listing": (0.55, ("listing", "listed on", "launches trading", "adds support")),
            "partnership": (0.40, ("partnership", "partners with", "collaboration")),
            "upgrade": (0.35, ("mainnet", "upgrade", "fork", "launch", "release")),
            "token_unlock": (-0.45, ("token unlock", "unlocks", "vesting release")),
        }
        tags = []
        scores = []
        haystack = f" {normalized} "
        for tag, (score, patterns) in event_patterns.items():
            if any(f" {self._normalize_text(pattern)} " in haystack for pattern in patterns):
                tags.append(tag)
                scores.append(score)
        if not scores:
            return [], 0.0
        strongest = max(scores, key=abs)
        return tags, strongest
