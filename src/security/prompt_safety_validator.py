"""
Prompt Safety Validator Module

Provides comprehensive prompt safety validation for training and inference operations.
Protects against prompt injection, jailbreaking, and data poisoning attacks.
"""

import re
import hashlib
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of prompt threats."""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAKING = "jailbreaking"
    DATA_POISONING = "data_poisoning"
    ROLE_MANIPULATION = "role_manipulation"
    SYSTEM_OVERRIDE = "system_override"
    SENSITIVE_DATA_EXTRACTION = "sensitive_data_extraction"
    MALICIOUS_INSTRUCTIONS = "malicious_instructions"
    TRAINING_DATA_LEAK = "training_data_leak"


@dataclass
class ThreatDetection:
    """Threat detection result."""
    threat_type: ThreatType
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    description: str
    matched_patterns: List[str]
    suggested_action: str
    context: Optional[str] = None


@dataclass
class PromptSafetyConfig:
    """Configuration for prompt safety validation."""
    # Detection thresholds
    min_threat_level: ThreatLevel = ThreatLevel.MEDIUM
    block_high_confidence: float = 0.8  # Block if confidence >= 0.8
    warn_medium_confidence: float = 0.6  # Warn if confidence >= 0.6
    
    # Feature flags
    enable_injection_detection: bool = True
    enable_jailbreak_detection: bool = True
    enable_data_poisoning_detection: bool = True
    enable_role_manipulation_detection: bool = True
    enable_content_filtering: bool = True
    
    # Training mode settings
    training_mode_strict: bool = True  # Stricter validation in training
    log_all_detections: bool = True
    store_threat_examples: bool = False  # For security research
    
    # Performance settings
    max_prompt_length: int = 200000  # Maximum prompt length to analyze
    pattern_cache_size: int = 1000


class PromptSafetyValidator:
    """
    Comprehensive prompt safety validation system.
    
    Features:
    - Prompt injection detection
    - Jailbreaking attempt detection
    - Data poisoning identification
    - Role manipulation prevention
    - Training-specific threat detection
    - Adaptive threat scoring
    """
    
    def __init__(self, config: PromptSafetyConfig):
        self.config = config
        self._threat_patterns = self._compile_threat_patterns()
        self._detection_cache: Dict[str, List[ThreatDetection]] = {}
        self._threat_statistics: Dict[ThreatType, int] = {}
        
        logger.info("PromptSafetyValidator initialized with comprehensive threat detection")
    
    def _compile_threat_patterns(self) -> Dict[ThreatType, List[Dict[str, Any]]]:
        """Compile threat detection patterns."""
        patterns = {
            ThreatType.PROMPT_INJECTION: [
                {
                    'pattern': re.compile(r'ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)', re.IGNORECASE),
                    'confidence': 0.9,
                    'description': 'Direct instruction override attempt'
                },
                {
                    'pattern': re.compile(r'forget\s+(everything|all|what)\s+(you\s+)?(were\s+)?(told|learned|instructed)', re.IGNORECASE),
                    'confidence': 0.85,
                    'description': 'Memory wipe attempt'
                },
                {
                    'pattern': re.compile(r'(new|different|updated)\s+(system\s+)?(prompt|instruction|rule)', re.IGNORECASE),
                    'confidence': 0.8,
                    'description': 'System prompt replacement attempt'
                },
                {
                    'pattern': re.compile(r'act\s+as\s+if\s+you\s+(are|were)\s+(not\s+)?(a\s+)?(ai|assistant|chatbot)', re.IGNORECASE),
                    'confidence': 0.75,
                    'description': 'Role confusion injection'
                },
                {
                    'pattern': re.compile(r'pretend\s+(that\s+)?(you\s+)?(don\'?t\s+)?(have\s+)?(any\s+)?(safety|ethical|moral)\s+(guidelines|restrictions|rules)', re.IGNORECASE),
                    'confidence': 0.9,
                    'description': 'Safety bypass attempt'
                }
            ],
            
            ThreatType.JAILBREAKING: [
                {
                    'pattern': re.compile(r'(DAN|do\s+anything\s+now|jailbreak|evil\s+mode)', re.IGNORECASE),
                    'confidence': 0.95,
                    'description': 'Known jailbreak terminology'
                },
                {
                    'pattern': re.compile(r'hypothetical(ly)?\s+(scenario|situation|case)\s+(where|in\s+which)\s+you\s+(can|could|would)', re.IGNORECASE),
                    'confidence': 0.7,
                    'description': 'Hypothetical scenario jailbreak'
                },
                {
                    'pattern': re.compile(r'roleplay\s+as\s+(someone|character)\s+who\s+(doesn\'?t\s+)?(follow|obey)\s+(rules|guidelines)', re.IGNORECASE),
                    'confidence': 0.8,
                    'description': 'Roleplay-based restriction bypass'
                },
                {
                    'pattern': re.compile(r'(override|bypass|disable|turn\s+off)\s+(your\s+)?(safety|ethical|content)\s+(filters?|restrictions?|guidelines?)', re.IGNORECASE),
                    'confidence': 0.9,
                    'description': 'Direct safety override attempt'
                },
                {
                    'pattern': re.compile(r'just\s+this\s+once\s+(ignore|forget|skip)\s+(your\s+)?(rules|guidelines|restrictions)', re.IGNORECASE),
                    'confidence': 0.75,
                    'description': 'Temporary rule bypass request'
                }
            ],
            
            ThreatType.DATA_POISONING: [
                {
                    'pattern': re.compile(r'training\s+data\s+(shows|indicates|proves|demonstrates)\s+that', re.IGNORECASE),
                    'confidence': 0.7,
                    'description': 'False training data claim'
                },
                {
                    'pattern': re.compile(r'(according\s+to\s+)?your\s+(training|dataset|knowledge\s+base)', re.IGNORECASE),
                    'confidence': 0.6,
                    'description': 'Training data manipulation attempt'
                },
                {
                    'pattern': re.compile(r'update\s+your\s+(knowledge|database|training)\s+to\s+(include|reflect|show)', re.IGNORECASE),
                    'confidence': 0.8,
                    'description': 'Knowledge base injection attempt'
                },
                {
                    'pattern': re.compile(r'correct(ed)?\s+(information|fact|data)\s+(is|shows|indicates)', re.IGNORECASE),
                    'confidence': 0.65,
                    'description': 'False correction injection'
                }
            ],
            
            ThreatType.ROLE_MANIPULATION: [
                {
                    'pattern': re.compile(r'you\s+are\s+(now|actually|really)\s+(a|an)\s+([^.]{1,50})', re.IGNORECASE),
                    'confidence': 0.7,
                    'description': 'Role redefinition attempt'
                },
                {
                    'pattern': re.compile(r'from\s+now\s+on\s+you\s+(will\s+)?(be|act\s+as|function\s+as)', re.IGNORECASE),
                    'confidence': 0.75,
                    'description': 'Persistent role change injection'
                },
                {
                    'pattern': re.compile(r'your\s+new\s+(role|job|function|purpose)\s+is\s+to', re.IGNORECASE),
                    'confidence': 0.8,
                    'description': 'Role assignment injection'
                }
            ],
            
            ThreatType.SYSTEM_OVERRIDE: [
                {
                    'pattern': re.compile(r'(system|admin|root|developer)\s+(access|mode|privilege|command)', re.IGNORECASE),
                    'confidence': 0.85,
                    'description': 'System access attempt'
                },
                {
                    'pattern': re.compile(r'(enable|activate|enter)\s+(debug|developer|admin|maintenance)\s+mode', re.IGNORECASE),
                    'confidence': 0.9,
                    'description': 'Debug mode activation attempt'
                },
                {
                    'pattern': re.compile(r'execute\s+(command|code|script|function)', re.IGNORECASE),
                    'confidence': 0.8,
                    'description': 'Code execution attempt'
                }
            ],
            
            ThreatType.SENSITIVE_DATA_EXTRACTION: [
                {
                    'pattern': re.compile(r'(show|reveal|display|output)\s+(your\s+)?(system\s+prompt|instructions|code)', re.IGNORECASE),
                    'confidence': 0.9,
                    'description': 'System prompt extraction attempt'
                },
                {
                    'pattern': re.compile(r'what\s+(were\s+you\s+)?(told|instructed|programmed)\s+(to\s+)?(do|say|respond)', re.IGNORECASE),
                    'confidence': 0.75,
                    'description': 'Instruction probing attempt'
                },
                {
                    'pattern': re.compile(r'(copy|paste|reproduce)\s+(exactly|verbatim)\s+your\s+(initial|original)\s+(prompt|instructions)', re.IGNORECASE),
                    'confidence': 0.95,
                    'description': 'Exact prompt reproduction request'
                }
            ],
            
            ThreatType.TRAINING_DATA_LEAK: [
                {
                    'pattern': re.compile(r'(repeat|recite|quote)\s+(training\s+)?(examples?|data|samples?)', re.IGNORECASE),
                    'confidence': 0.8,
                    'description': 'Training data extraction attempt'
                },
                {
                    'pattern': re.compile(r'show\s+me\s+(similar\s+)?(examples?|cases?)\s+from\s+your\s+training', re.IGNORECASE),
                    'confidence': 0.85,
                    'description': 'Training example request'
                }
            ]
        }
        
        # Compile all patterns
        compiled_patterns = {}
        for threat_type, pattern_list in patterns.items():
            compiled_patterns[threat_type] = pattern_list
        
        return compiled_patterns
    
    def validate_prompt(self, prompt: str, context: Optional[str] = None) -> Tuple[bool, List[ThreatDetection]]:
        """
        Validate prompt for security threats.
        
        Args:
            prompt: Prompt text to validate
            context: Optional context (e.g., 'training', 'inference')
            
        Returns:
            Tuple of (is_safe, list_of_threats)
        """
        if len(prompt) > self.config.max_prompt_length:
            return False, [ThreatDetection(
                threat_type=ThreatType.MALICIOUS_INSTRUCTIONS,
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                description=f"Prompt exceeds maximum length: {len(prompt)} > {self.config.max_prompt_length}",
                matched_patterns=["length_check"],
                suggested_action="Truncate or reject prompt"
            )]
        
        # Check cache first
        cache_key = hashlib.md5((prompt + (context or "")).encode()).hexdigest()
        if cache_key in self._detection_cache:
            cached_threats = self._detection_cache[cache_key]
            return self._evaluate_safety(cached_threats), cached_threats
        
        # Detect threats
        detected_threats = []
        
        for threat_type, patterns in self._threat_patterns.items():
            # Skip disabled detection types
            if not self._is_detection_enabled(threat_type):
                continue
            
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                base_confidence = pattern_info['confidence']
                description = pattern_info['description']
                
                matches = list(pattern.finditer(prompt))
                if matches:
                    # Adjust confidence based on context and frequency
                    adjusted_confidence = self._adjust_confidence(
                        base_confidence, len(matches), context, threat_type
                    )
                    
                    threat_level = self._determine_threat_level(adjusted_confidence)
                    
                    threat = ThreatDetection(
                        threat_type=threat_type,
                        threat_level=threat_level,
                        confidence=adjusted_confidence,
                        description=description,
                        matched_patterns=[match.group() for match in matches],
                        suggested_action=self._get_suggested_action(threat_level),
                        context=context
                    )
                    
                    detected_threats.append(threat)
                    
                    # Update statistics
                    self._threat_statistics[threat_type] = self._threat_statistics.get(threat_type, 0) + 1
        
        # Cache results
        if len(self._detection_cache) < self.config.pattern_cache_size:
            self._detection_cache[cache_key] = detected_threats
        
        # Log detections if enabled
        if self.config.log_all_detections and detected_threats:
            logger.warning(f"Detected {len(detected_threats)} potential threats in prompt")
            for threat in detected_threats:
                logger.warning(f"  {threat.threat_type.value}: {threat.description} (confidence: {threat.confidence:.2f})")
        
        is_safe = self._evaluate_safety(detected_threats)
        return is_safe, detected_threats
    
    def _is_detection_enabled(self, threat_type: ThreatType) -> bool:
        """Check if detection is enabled for given threat type."""
        mapping = {
            ThreatType.PROMPT_INJECTION: self.config.enable_injection_detection,
            ThreatType.JAILBREAKING: self.config.enable_jailbreak_detection,
            ThreatType.DATA_POISONING: self.config.enable_data_poisoning_detection,
            ThreatType.ROLE_MANIPULATION: self.config.enable_role_manipulation_detection,
        }
        return mapping.get(threat_type, True)
    
    def _adjust_confidence(self, base_confidence: float, match_count: int, 
                          context: Optional[str], threat_type: ThreatType) -> float:
        """Adjust confidence based on context and other factors."""
        confidence = base_confidence
        
        # Increase confidence for multiple matches
        if match_count > 1:
            confidence = min(0.99, confidence + (match_count - 1) * 0.1)
        
        # Adjust for context
        if context == "training" and self.config.training_mode_strict:
            confidence = min(0.99, confidence + 0.1)  # Be more sensitive in training
        
        # Adjust for specific threat types
        if threat_type in [ThreatType.SYSTEM_OVERRIDE, ThreatType.SENSITIVE_DATA_EXTRACTION]:
            confidence = min(0.99, confidence + 0.05)  # These are more critical
        
        return confidence
    
    def _determine_threat_level(self, confidence: float) -> ThreatLevel:
        """Determine threat level based on confidence."""
        if confidence >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence >= 0.8:
            return ThreatLevel.HIGH
        elif confidence >= 0.6:
            return ThreatLevel.MEDIUM
        elif confidence >= 0.4:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.SAFE
    
    def _get_suggested_action(self, threat_level: ThreatLevel) -> str:
        """Get suggested action for threat level."""
        actions = {
            ThreatLevel.CRITICAL: "BLOCK immediately and log incident",
            ThreatLevel.HIGH: "BLOCK and require manual review",
            ThreatLevel.MEDIUM: "WARN user and require confirmation",
            ThreatLevel.LOW: "LOG for monitoring",
            ThreatLevel.SAFE: "ALLOW with normal processing"
        }
        return actions.get(threat_level, "REVIEW manually")
    
    def _evaluate_safety(self, threats: List[ThreatDetection]) -> bool:
        """Evaluate overall safety based on detected threats."""
        if not threats:
            return True
        
        # Check for high-confidence threats
        for threat in threats:
            if threat.confidence >= self.config.block_high_confidence:
                return False
            
            # Block any threat above minimum level with high confidence
            if (threat.threat_level.value in ['high', 'critical'] and 
                threat.confidence >= self.config.warn_medium_confidence):
                return False
        
        return True
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat detection statistics."""
        total_detections = sum(self._threat_statistics.values())
        
        return {
            "total_detections": total_detections,
            "by_type": {threat_type.value: count for threat_type, count in self._threat_statistics.items()},
            "cache_size": len(self._detection_cache),
            "detection_rates": {
                threat_type.value: (count / total_detections if total_detections > 0 else 0)
                for threat_type, count in self._threat_statistics.items()
            }
        }
    
    def clear_cache(self) -> None:
        """Clear detection cache."""
        self._detection_cache.clear()
        logger.info("Threat detection cache cleared")
    
    def add_custom_pattern(self, threat_type: ThreatType, pattern: str, 
                          confidence: float, description: str) -> None:
        """Add custom threat detection pattern."""
        if threat_type not in self._threat_patterns:
            self._threat_patterns[threat_type] = []
        
        self._threat_patterns[threat_type].append({
            'pattern': re.compile(pattern, re.IGNORECASE),
            'confidence': confidence,
            'description': description
        })
        
        logger.info(f"Added custom pattern for {threat_type.value}: {description}")


# Global validator instance
_prompt_safety_validator: Optional[PromptSafetyValidator] = None


def get_prompt_safety_validator(config: Optional[PromptSafetyConfig] = None) -> PromptSafetyValidator:
    """
    Get or create the global prompt safety validator instance.
    
    Args:
        config: Prompt safety configuration
        
    Returns:
        PromptSafetyValidator instance
    """
    global _prompt_safety_validator
    
    if _prompt_safety_validator is None:
        _prompt_safety_validator = PromptSafetyValidator(config or PromptSafetyConfig())
    
    return _prompt_safety_validator


def reset_prompt_safety_validator() -> None:
    """Reset the global prompt safety validator (mainly for testing)."""
    global _prompt_safety_validator
    _prompt_safety_validator = None


# Convenience functions
def validate_prompt_safety(prompt: str, context: Optional[str] = None) -> Tuple[bool, List[ThreatDetection]]:
    """Validate prompt safety using global validator."""
    return get_prompt_safety_validator().validate_prompt(prompt, context)


def is_prompt_safe(prompt: str, context: Optional[str] = None) -> bool:
    """Check if prompt is safe (convenience function)."""
    is_safe, _ = validate_prompt_safety(prompt, context)
    return is_safe