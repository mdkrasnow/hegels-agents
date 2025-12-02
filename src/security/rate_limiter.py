"""
API Rate Limiting Module

Provides rate limiting functionality for API calls to prevent abuse and control costs.
Implements token bucket and sliding window algorithms for different use cases.
"""

import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10
    cost_per_request: float = 0.0  # Estimated cost per request
    max_daily_cost: float = 100.0


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            
            # Add tokens based on time passed
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until specified tokens will be available."""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.refill_rate


class SlidingWindowCounter:
    """Sliding window rate limiter for time-based limits."""
    
    def __init__(self, window_size: int, limit: int):
        self.window_size = window_size  # in seconds
        self.limit = limit
        self.requests = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed within current window."""
        with self._lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we're under limit
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False
    
    def time_until_available(self) -> float:
        """Calculate time until next request will be allowed."""
        with self._lock:
            if len(self.requests) < self.limit:
                return 0.0
            
            now = time.time()
            oldest_request = self.requests[0]
            return max(0.0, (oldest_request + self.window_size) - now)


class RateLimiter:
    """
    Comprehensive rate limiter with multiple strategies and cost control.
    
    Provides:
    - Token bucket for burst control
    - Sliding window for time-based limits
    - Cost tracking and daily budget enforcement
    - Per-endpoint and global limiting
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = threading.Lock()
        
        # Token bucket for burst control
        self.token_bucket = TokenBucket(
            capacity=config.burst_allowance,
            refill_rate=config.requests_per_minute / 60.0  # tokens per second
        )
        
        # Sliding windows for time-based limits
        self.minute_window = SlidingWindowCounter(60, config.requests_per_minute)
        self.hour_window = SlidingWindowCounter(3600, config.requests_per_hour)
        self.day_window = SlidingWindowCounter(86400, config.requests_per_day)
        
        # Cost tracking
        self.daily_cost = 0.0
        self.last_cost_reset = time.time()
        self.request_costs = deque()  # Track costs for daily reset
        
        # Per-endpoint rate limiting
        self.endpoint_limiters: Dict[str, 'RateLimiter'] = {}
        
        logger.info(f"RateLimiter initialized: {config.requests_per_minute}/min, "
                   f"{config.requests_per_hour}/hour, {config.requests_per_day}/day")
    
    def check_and_consume(self, endpoint: str = "default", estimated_cost: float = 0.0) -> bool:
        """
        Check if request is allowed and consume rate limit tokens.
        
        Args:
            endpoint: API endpoint identifier for per-endpoint limiting
            estimated_cost: Estimated cost of the request
            
        Returns:
            True if request is allowed
            
        Raises:
            RateLimitError: If rate limit is exceeded
        """
        with self._lock:
            # Reset daily cost if needed
            self._reset_daily_cost_if_needed()
            
            # Check daily cost budget
            if self.daily_cost + estimated_cost > self.config.max_daily_cost:
                raise RateLimitError(
                    f"Daily cost budget exceeded: ${self.daily_cost:.2f} + ${estimated_cost:.2f} > ${self.config.max_daily_cost}",
                    retry_after=self._time_until_cost_reset()
                )
            
            # Check token bucket (burst control)
            if not self.token_bucket.consume():
                retry_after = self.token_bucket.time_until_available()
                raise RateLimitError(
                    f"Burst rate limit exceeded. Too many requests too quickly.",
                    retry_after=retry_after
                )
            
            # Check sliding windows
            if not self.minute_window.is_allowed():
                retry_after = self.minute_window.time_until_available()
                raise RateLimitError(
                    f"Per-minute rate limit exceeded: {self.config.requests_per_minute}/min",
                    retry_after=retry_after
                )
            
            if not self.hour_window.is_allowed():
                retry_after = self.hour_window.time_until_available()
                raise RateLimitError(
                    f"Per-hour rate limit exceeded: {self.config.requests_per_hour}/hour",
                    retry_after=retry_after
                )
            
            if not self.day_window.is_allowed():
                retry_after = self.day_window.time_until_available()
                raise RateLimitError(
                    f"Daily rate limit exceeded: {self.config.requests_per_day}/day",
                    retry_after=retry_after
                )
            
            # Track cost
            if estimated_cost > 0:
                self.daily_cost += estimated_cost
                self.request_costs.append((time.time(), estimated_cost))
            
            logger.debug(f"Rate limit check passed for {endpoint}. Daily cost: ${self.daily_cost:.4f}")
            return True
    
    def _reset_daily_cost_if_needed(self) -> None:
        """Reset daily cost counter if 24 hours have passed."""
        now = time.time()
        if now - self.last_cost_reset >= 86400:  # 24 hours
            self.daily_cost = 0.0
            self.last_cost_reset = now
            self.request_costs.clear()
            logger.info("Daily cost counter reset")
    
    def _time_until_cost_reset(self) -> float:
        """Calculate time until daily cost resets."""
        now = time.time()
        return max(0.0, 86400 - (now - self.last_cost_reset))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        with self._lock:
            self._reset_daily_cost_if_needed()
            
            return {
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day,
                    "max_daily_cost": self.config.max_daily_cost,
                },
                "current_status": {
                    "tokens_available": self.token_bucket.tokens,
                    "daily_cost": self.daily_cost,
                    "cost_budget_remaining": self.config.max_daily_cost - self.daily_cost,
                    "time_until_cost_reset": self._time_until_cost_reset(),
                },
                "windows": {
                    "minute_requests": len([r for r in self.minute_window.requests if r > time.time() - 60]),
                    "hour_requests": len([r for r in self.hour_window.requests if r > time.time() - 3600]),
                    "day_requests": len([r for r in self.day_window.requests if r > time.time() - 86400]),
                }
            }
    
    def wait_if_needed(self, endpoint: str = "default", estimated_cost: float = 0.0, max_wait: float = 60.0) -> None:
        """
        Wait if rate limited, up to max_wait seconds.
        
        Args:
            endpoint: API endpoint identifier
            estimated_cost: Estimated cost of the request
            max_wait: Maximum time to wait in seconds
            
        Raises:
            RateLimitError: If wait time exceeds max_wait
        """
        try:
            self.check_and_consume(endpoint, estimated_cost)
        except RateLimitError as e:
            if e.retry_after is None or e.retry_after > max_wait:
                raise
            
            logger.warning(f"Rate limited, waiting {e.retry_after:.1f} seconds: {e}")
            time.sleep(e.retry_after)
            
            # Try again after waiting
            self.check_and_consume(endpoint, estimated_cost)


# Global rate limiter instances
_default_limiter: Optional[RateLimiter] = None
_endpoint_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(endpoint: str = "default", config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """
    Get or create rate limiter for endpoint.
    
    Args:
        endpoint: API endpoint identifier
        config: Rate limit configuration (uses default if None)
        
    Returns:
        RateLimiter instance
    """
    global _default_limiter, _endpoint_limiters
    
    if endpoint == "default":
        if _default_limiter is None:
            _default_limiter = RateLimiter(config or RateLimitConfig())
        return _default_limiter
    
    if endpoint not in _endpoint_limiters:
        _endpoint_limiters[endpoint] = RateLimiter(config or RateLimitConfig())
    
    return _endpoint_limiters[endpoint]


def reset_rate_limiters() -> None:
    """Reset all rate limiters (mainly for testing)."""
    global _default_limiter, _endpoint_limiters
    _default_limiter = None
    _endpoint_limiters.clear()