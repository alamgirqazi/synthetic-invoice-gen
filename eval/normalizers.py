"""
Normalizers for Receipt OCR Evaluation

Handles text normalization, date parsing, amount standardization.
All comparisons happen on normalized values to avoid "formatting noise".
"""

import re
import unicodedata
from datetime import datetime, date
from typing import Optional, Union


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Normalize text for comparison.
    - Strip whitespace
    - Lowercase
    - Normalize unicode
    - Collapse multiple spaces
    - Remove trailing punctuation
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    
    text = text.strip()
    if not text:
        return None
    
    # Unicode normalization (NFKD decomposes, e.g. € -> EUR-like)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing punctuation (but keep internal)
    text = text.strip('.,;:!? ')
    
    return text if text else None


def normalize_text_aggressive(text: Optional[str]) -> Optional[str]:
    """
    Aggressive normalization for identifiers (invoice numbers, IDs).
    Removes all non-alphanumeric characters.
    """
    if text is None:
        return None
    text = normalize_text(text)
    if text is None:
        return None
    # Keep only alphanumeric
    text = re.sub(r'[^a-z0-9]', '', text)
    return text if text else None


def tokenize(text: Optional[str]) -> list:
    """Tokenize normalized text into words."""
    if text is None:
        return []
    normalized = normalize_text(text)
    if normalized is None:
        return []
    return normalized.split()


# ============================================================
# NUMERIC NORMALIZATION
# ============================================================

def normalize_numeric(value) -> Optional[float]:
    """
    Parse any numeric representation to float.
    Handles: "9.99", "9,99", "$9.99", "€27.20", "1,234.56", "1.234,56"
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        value = str(value)
    
    value = value.strip()
    if not value:
        return None
    
    # Remove currency symbols and common prefixes
    value = re.sub(r'[€$£¥₹₽₩₺₱₿]', '', value)
    value = re.sub(r'^\s*(EUR|USD|GBP|CHF|JPY|CAD|AUD)\s*', '', value, flags=re.IGNORECASE)
    value = value.strip()
    
    if not value:
        return None
    
    # Handle percentage strings like "23%"
    value = value.replace('%', '').strip()
    
    # Determine decimal separator
    # "1,234.56" -> comma is thousands separator
    # "1.234,56" -> period is thousands separator
    # "9,99" -> comma is decimal separator
    # "9.99" -> period is decimal separator
    
    has_comma = ',' in value
    has_period = '.' in value
    
    if has_comma and has_period:
        # Both present - last one is the decimal separator
        last_comma = value.rfind(',')
        last_period = value.rfind('.')
        if last_comma > last_period:
            # Format: 1.234,56 (European)
            value = value.replace('.', '').replace(',', '.')
        else:
            # Format: 1,234.56 (US/UK)
            value = value.replace(',', '')
    elif has_comma:
        # Only comma: could be "1,234" (thousands) or "9,99" (decimal)
        parts = value.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely decimal: "9,99"
            value = value.replace(',', '.')
        else:
            # Likely thousands: "1,234"
            value = value.replace(',', '')
    # If only period or neither, standard float parsing works
    
    try:
        return float(value)
    except ValueError:
        # Last resort: extract first number-like pattern
        match = re.search(r'[\d.]+', value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None


# ============================================================
# DATE NORMALIZATION
# ============================================================

# Common date formats to try, ordered by specificity
DATE_FORMATS = [
    # ISO
    "%Y-%m-%d",
    "%Y/%m/%d",
    # European (DD/MM/YYYY)
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    # US (MM/DD/YYYY) - tried after European
    "%m/%d/%Y",
    # Short year
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%m/%d/%y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    # Written formats
    "%B %d, %Y",         # January 26, 2024
    "%b %d, %Y",         # Jan 26, 2024
    "%d %B %Y",          # 26 January 2024
    "%d %b %Y",          # 26 Jan 2024
    "%B %d %Y",          # January 26 2024
    "%b %d %Y",          # Jan 26 2024
]


def normalize_date(value) -> Optional[date]:
    """
    Parse any date representation to a date object.
    Handles multiple formats, returns None if unparseable.
    """
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    
    if not isinstance(value, str):
        value = str(value)
    
    value = value.strip()
    if not value or value.lower() in ('null', 'none', 'n/a', ''):
        return None
    
    # Remove day-of-week prefixes
    value = re.sub(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*,?\s*', '', value, flags=re.IGNORECASE)
    # Remove ordinal suffixes: 1st, 2nd, 3rd, 4th, etc.
    value = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', value)
    value = value.strip()
    
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    
    return None


# ============================================================
# TIME NORMALIZATION
# ============================================================

def normalize_time(value) -> Optional[str]:
    """
    Normalize time to HH:MM format (24h).
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        value = str(value)
    
    value = value.strip().upper()
    if not value or value.lower() in ('null', 'none', 'n/a'):
        return None
    
    # Try common time formats
    time_formats = [
        "%H:%M:%S",
        "%H:%M",
        "%I:%M %p",
        "%I:%M:%S %p",
        "%I:%M%p",
    ]
    
    for fmt in time_formats:
        try:
            t = datetime.strptime(value, fmt)
            return t.strftime("%H:%M")
        except ValueError:
            continue
    
    return value  # Return as-is if unparseable


# ============================================================
# CURRENCY NORMALIZATION
# ============================================================

CURRENCY_ALIASES = {
    "euro": "EUR", "euros": "EUR", "€": "EUR",
    "dollar": "USD", "dollars": "USD", "$": "USD", "us dollar": "USD",
    "pound": "GBP", "pounds": "GBP", "£": "GBP", "sterling": "GBP",
    "yen": "JPY", "¥": "JPY",
    "franc": "CHF", "francs": "CHF",
}

def normalize_currency(value: Optional[str]) -> Optional[str]:
    """Normalize currency to 3-letter code."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None
    
    lower = value.lower()
    if lower in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[lower]
    
    # Already a 3-letter code
    if len(value) == 3 and value.isalpha():
        return value.upper()
    
    return value.upper()


# ============================================================
# NULL HANDLING
# ============================================================

def is_null(value) -> bool:
    """Check if a value is effectively null/empty."""
    if value is None:
        return True
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ('', 'null', 'none', 'n/a', 'na', '-', '--')
    if isinstance(value, (list, dict)):
        return len(value) == 0
    if isinstance(value, (int, float)):
        return False  # 0 and 0.0 are valid values, not null
    return False
