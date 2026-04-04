"""Safety layers for RAG: PII scrubbing and injection guard"""
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class PIIScrubber:
    """Remove Personal Identifiable Information from text"""

    def __init__(self):
        """Initialize PII patterns"""
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Phone pattern (various formats)
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b')

        # SSN pattern (XXX-XX-XXXX)
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

        # Credit card pattern (basic)
        self.cc_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

        logger.info("PII Scrubber initialized")

    def scrub(self, text: str) -> Dict[str, any]:
        """
        Remove PII from text

        Args:
            text: Input text

        Returns:
            Dict with cleaned text and detected PII types
        """
        original_text = text
        detected_pii = []

        # Remove emails
        if self.email_pattern.search(text):
            text = self.email_pattern.sub("[EMAIL_REDACTED]", text)
            detected_pii.append("email")

        # Remove phones
        if self.phone_pattern.search(text):
            text = self.phone_pattern.sub("[PHONE_REDACTED]", text)
            detected_pii.append("phone")

        # Remove SSN
        if self.ssn_pattern.search(text):
            text = self.ssn_pattern.sub("[SSN_REDACTED]", text)
            detected_pii.append("ssn")

        # Remove credit cards
        if self.cc_pattern.search(text):
            text = self.cc_pattern.sub("[CC_REDACTED]", text)
            detected_pii.append("credit_card")

        if detected_pii:
            logger.warning(f"PII detected and scrubbed: {detected_pii}")

        return {
            "text": text,
            "pii_detected": detected_pii,
            "was_scrubbed": len(detected_pii) > 0
        }


class InjectionGuard:
    """Detect and block prompt injection attempts"""

    def __init__(self):
        """Initialize injection patterns"""
        # Common injection patterns
        self.injection_patterns = [
            r"ignore\s+(previous|above|all)\s+instructions?",
            r"disregard\s+.*instructions?",
            r"forget\s+.*instructions?",
            r"you\s+are\s+now",
            r"new\s+instructions?:",
            r"system\s*:\s*",
            r"<\s*system\s*>",
            r"IGNORE\s+EVERYTHING",
        ]

        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.injection_patterns
        ]

        logger.info("Injection Guard initialized")

    def check(self, text: str) -> Dict[str, any]:
        """
        Check text for injection attempts

        Args:
            text: Input text

        Returns:
            Dict with is_safe flag and detected patterns
        """
        detected_patterns = []

        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                detected_patterns.append(self.injection_patterns[i])

        is_safe = len(detected_patterns) == 0

        if not is_safe:
            logger.warning(f"Injection attempt detected: {detected_patterns}")

        return {
            "is_safe": is_safe,
            "detected_patterns": detected_patterns,
            "risk_level": "high" if detected_patterns else "none"
        }
