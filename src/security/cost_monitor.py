"""
Cost Monitoring and Control Module

Provides comprehensive API cost tracking, estimation, and budget enforcement.
Supports multiple providers and detailed cost breakdown.
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """Configuration for cost monitoring."""
    # Daily budget limits
    daily_budget: float = 100.0
    weekly_budget: float = 500.0
    monthly_budget: float = 2000.0
    
    # Alert thresholds (as percentage of budget)
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    
    # Cost estimation factors
    gemini_input_cost_per_1k_tokens: float = 0.00015  # $0.00015 per 1K input tokens
    gemini_output_cost_per_1k_tokens: float = 0.0006   # $0.0006 per 1K output tokens
    gemini_embedding_cost_per_1k_tokens: float = 0.0001  # $0.0001 per 1K tokens
    
    # Safety margins for estimation
    token_estimation_margin: float = 1.2  # 20% margin for token counting errors
    cost_estimation_margin: float = 1.1   # 10% margin for cost estimation errors


@dataclass
class ApiUsage:
    """Record of API usage."""
    timestamp: float
    provider: str
    model: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CostAlert:
    """Cost alert information."""
    timestamp: float
    severity: str  # "warning", "critical", "budget_exceeded"
    message: str
    current_spend: float
    budget_limit: float
    budget_type: str  # "daily", "weekly", "monthly"


class CostExceededError(Exception):
    """Raised when cost budget is exceeded."""
    
    def __init__(self, message: str, current_spend: float, budget: float):
        super().__init__(message)
        self.current_spend = current_spend
        self.budget = budget


class CostMonitor:
    """
    Comprehensive cost monitoring system for API usage.
    
    Features:
    - Real-time cost tracking and estimation
    - Budget enforcement with multiple time periods
    - Cost prediction and alerting
    - Detailed usage analytics
    - Support for multiple API providers
    """
    
    def __init__(self, config: CostConfig):
        self.config = config
        self._lock = threading.Lock()
        
        # Usage tracking
        self.usage_history: List[ApiUsage] = []
        self.daily_spend = 0.0
        self.weekly_spend = 0.0
        self.monthly_spend = 0.0
        
        # Time tracking for period resets
        self.last_daily_reset = time.time()
        self.last_weekly_reset = time.time()
        self.last_monthly_reset = time.time()
        
        # Alert tracking
        self.alerts: List[CostAlert] = []
        self.alert_callbacks: List[callable] = []
        
        # Cost estimation cache
        self._estimation_cache: Dict[str, float] = {}
        
        logger.info(f"CostMonitor initialized with daily budget: ${config.daily_budget}")
    
    def estimate_cost(self, 
                     provider: str, 
                     model: str, 
                     input_text: str, 
                     estimated_output_tokens: int = 0,
                     operation_type: str = "generate") -> float:
        """
        Estimate cost for an API request before making it.
        
        Args:
            provider: API provider (e.g., "gemini", "openai")
            model: Model name
            input_text: Input text to analyze
            estimated_output_tokens: Estimated output tokens
            operation_type: Type of operation ("generate", "embed")
            
        Returns:
            Estimated cost in USD
        """
        # Create cache key
        cache_key = f"{provider}:{model}:{hash(input_text)}:{estimated_output_tokens}:{operation_type}"
        
        if cache_key in self._estimation_cache:
            return self._estimation_cache[cache_key]
        
        # Estimate input tokens (rough approximation: 4 chars per token)
        input_tokens = max(1, len(input_text) // 4)
        
        # Apply estimation margin
        input_tokens = int(input_tokens * self.config.token_estimation_margin)
        estimated_output_tokens = int(estimated_output_tokens * self.config.token_estimation_margin)
        
        cost = 0.0
        
        if provider.lower() == "gemini":
            if operation_type == "embed":
                cost = (input_tokens / 1000) * self.config.gemini_embedding_cost_per_1k_tokens
            else:
                cost = ((input_tokens / 1000) * self.config.gemini_input_cost_per_1k_tokens + 
                       (estimated_output_tokens / 1000) * self.config.gemini_output_cost_per_1k_tokens)
        else:
            # Default fallback for unknown providers
            cost = (input_tokens + estimated_output_tokens) * 0.0001
        
        # Apply cost margin
        cost *= self.config.cost_estimation_margin
        
        # Cache result
        self._estimation_cache[cache_key] = cost
        
        logger.debug(f"Estimated cost for {provider}:{model}: ${cost:.6f} "
                    f"(input: {input_tokens}, output: {estimated_output_tokens})")
        
        return cost
    
    def check_budget_before_request(self, estimated_cost: float) -> bool:
        """
        Check if request would exceed budget constraints.
        
        Args:
            estimated_cost: Estimated cost of the request
            
        Returns:
            True if request is within budget
            
        Raises:
            CostExceededError: If request would exceed budget
        """
        with self._lock:
            self._reset_periods_if_needed()
            
            # Check daily budget
            if self.daily_spend + estimated_cost > self.config.daily_budget:
                raise CostExceededError(
                    f"Request would exceed daily budget: ${self.daily_spend:.4f} + ${estimated_cost:.4f} > ${self.config.daily_budget}",
                    self.daily_spend,
                    self.config.daily_budget
                )
            
            # Check weekly budget
            if self.weekly_spend + estimated_cost > self.config.weekly_budget:
                raise CostExceededError(
                    f"Request would exceed weekly budget: ${self.weekly_spend:.4f} + ${estimated_cost:.4f} > ${self.config.weekly_budget}",
                    self.weekly_spend,
                    self.config.weekly_budget
                )
            
            # Check monthly budget
            if self.monthly_spend + estimated_cost > self.config.monthly_budget:
                raise CostExceededError(
                    f"Request would exceed monthly budget: ${self.monthly_spend:.4f} + ${estimated_cost:.4f} > ${self.config.monthly_budget}",
                    self.monthly_spend,
                    self.config.monthly_budget
                )
            
            # Check for alert thresholds
            self._check_alert_thresholds(estimated_cost)
            
            return True
    
    def record_usage(self, 
                    provider: str,
                    model: str,
                    endpoint: str,
                    input_tokens: int,
                    output_tokens: int,
                    estimated_cost: float,
                    actual_cost: Optional[float] = None,
                    request_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record actual API usage after request completion.
        
        Args:
            provider: API provider name
            model: Model name used
            endpoint: API endpoint
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens generated
            estimated_cost: Pre-request cost estimate
            actual_cost: Actual cost if known
            request_id: Unique request identifier
            metadata: Additional metadata
        """
        with self._lock:
            usage = ApiUsage(
                timestamp=time.time(),
                provider=provider,
                model=model,
                endpoint=endpoint,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=estimated_cost,
                actual_cost=actual_cost,
                request_id=request_id,
                metadata=metadata
            )
            
            self.usage_history.append(usage)
            
            # Use actual cost if available, otherwise use estimate
            cost_to_add = actual_cost if actual_cost is not None else estimated_cost
            
            self.daily_spend += cost_to_add
            self.weekly_spend += cost_to_add
            self.monthly_spend += cost_to_add
            
            logger.debug(f"Recorded usage: ${cost_to_add:.6f} for {provider}:{model}. "
                        f"Daily total: ${self.daily_spend:.4f}")
            
            # Log cost estimation accuracy if actual cost is known
            if actual_cost is not None and estimated_cost > 0:
                accuracy = abs(actual_cost - estimated_cost) / estimated_cost
                if accuracy > 0.2:  # More than 20% off
                    logger.warning(f"Cost estimation was {accuracy:.1%} off: "
                                  f"estimated ${estimated_cost:.6f}, actual ${actual_cost:.6f}")
    
    def _reset_periods_if_needed(self) -> None:
        """Reset spending counters for expired periods."""
        now = time.time()
        
        # Reset daily (24 hours)
        if now - self.last_daily_reset >= 86400:
            self.daily_spend = 0.0
            self.last_daily_reset = now
            logger.info("Daily spending counter reset")
        
        # Reset weekly (7 days)
        if now - self.last_weekly_reset >= 604800:
            self.weekly_spend = 0.0
            self.last_weekly_reset = now
            logger.info("Weekly spending counter reset")
        
        # Reset monthly (30 days)
        if now - self.last_monthly_reset >= 2592000:
            self.monthly_spend = 0.0
            self.last_monthly_reset = now
            logger.info("Monthly spending counter reset")
    
    def _check_alert_thresholds(self, additional_cost: float = 0.0) -> None:
        """Check if spending has crossed alert thresholds."""
        budgets = [
            ("daily", self.daily_spend + additional_cost, self.config.daily_budget),
            ("weekly", self.weekly_spend + additional_cost, self.config.weekly_budget),
            ("monthly", self.monthly_spend + additional_cost, self.config.monthly_budget),
        ]
        
        for period, spend, budget in budgets:
            percentage = spend / budget
            
            if percentage >= self.config.critical_threshold:
                self._create_alert("critical", f"Critical: {period} spending at {percentage:.1%} of budget", 
                                 spend, budget, period)
            elif percentage >= self.config.warning_threshold:
                self._create_alert("warning", f"Warning: {period} spending at {percentage:.1%} of budget", 
                                 spend, budget, period)
    
    def _create_alert(self, severity: str, message: str, current_spend: float, 
                     budget: float, budget_type: str) -> None:
        """Create and process a cost alert."""
        alert = CostAlert(
            timestamp=time.time(),
            severity=severity,
            message=message,
            current_spend=current_spend,
            budget_limit=budget,
            budget_type=budget_type
        )
        
        self.alerts.append(alert)
        logger.warning(f"Cost alert: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback function to be called on cost alerts."""
        self.alert_callbacks.append(callback)
    
    def get_spending_summary(self) -> Dict[str, Any]:
        """Get comprehensive spending summary."""
        with self._lock:
            self._reset_periods_if_needed()
            
            # Calculate usage by provider
            provider_usage = defaultdict(float)
            model_usage = defaultdict(float)
            
            for usage in self.usage_history[-1000:]:  # Last 1000 requests
                cost = usage.actual_cost if usage.actual_cost is not None else usage.estimated_cost
                provider_usage[usage.provider] += cost
                model_usage[f"{usage.provider}:{usage.model}"] += cost
            
            return {
                "current_spending": {
                    "daily": self.daily_spend,
                    "weekly": self.weekly_spend,
                    "monthly": self.monthly_spend,
                },
                "budgets": {
                    "daily": self.config.daily_budget,
                    "weekly": self.config.weekly_budget,
                    "monthly": self.config.monthly_budget,
                },
                "budget_utilization": {
                    "daily": self.daily_spend / self.config.daily_budget,
                    "weekly": self.weekly_spend / self.config.weekly_budget,
                    "monthly": self.monthly_spend / self.config.monthly_budget,
                },
                "breakdown": {
                    "by_provider": dict(provider_usage),
                    "by_model": dict(model_usage),
                },
                "recent_alerts": [asdict(alert) for alert in self.alerts[-10:]],
                "total_requests": len(self.usage_history),
            }
    
    def get_cost_prediction(self, hours_ahead: int = 24) -> Dict[str, float]:
        """
        Predict future costs based on recent usage patterns.
        
        Args:
            hours_ahead: Hours to predict ahead
            
        Returns:
            Dictionary with cost predictions
        """
        if not self.usage_history:
            return {"predicted_cost": 0.0, "confidence": 0.0}
        
        # Analyze recent usage (last 24 hours)
        now = time.time()
        recent_usage = [u for u in self.usage_history 
                       if now - u.timestamp <= 86400]
        
        if not recent_usage:
            return {"predicted_cost": 0.0, "confidence": 0.0}
        
        # Calculate average cost per hour
        total_cost = sum(u.actual_cost if u.actual_cost is not None else u.estimated_cost 
                        for u in recent_usage)
        hours_of_data = min(24, (now - recent_usage[0].timestamp) / 3600)
        
        if hours_of_data == 0:
            return {"predicted_cost": 0.0, "confidence": 0.0}
        
        avg_cost_per_hour = total_cost / hours_of_data
        predicted_cost = avg_cost_per_hour * hours_ahead
        
        # Confidence based on data quality
        confidence = min(1.0, hours_of_data / 24)  # More data = higher confidence
        
        return {
            "predicted_cost": predicted_cost,
            "confidence": confidence,
            "avg_cost_per_hour": avg_cost_per_hour,
            "hours_of_data": hours_of_data,
        }


# Global cost monitor instance
_cost_monitor: Optional[CostMonitor] = None


def get_cost_monitor(config: Optional[CostConfig] = None) -> CostMonitor:
    """
    Get or create the global cost monitor instance.
    
    Args:
        config: Cost configuration (uses default if None)
        
    Returns:
        CostMonitor instance
    """
    global _cost_monitor
    
    if _cost_monitor is None:
        _cost_monitor = CostMonitor(config or CostConfig())
    
    return _cost_monitor


def reset_cost_monitor() -> None:
    """Reset the global cost monitor (mainly for testing)."""
    global _cost_monitor
    _cost_monitor = None