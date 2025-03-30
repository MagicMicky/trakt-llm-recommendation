"""
Utility functions for the TV show recommender.
"""

import os
import logging
import json
from datetime import datetime

def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def save_to_json(data, filename):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_from_json(filename):
    """Load data from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def format_date(date_string):
    """Format date string to a more readable format."""
    try:
        date_obj = datetime.strptime(date_string, "%Y-%m-%d")
        return date_obj.strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return date_string

def truncate_text(text, max_length=200):
    """Truncate text to a maximum length and add ellipsis."""
    if text and len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text

def calculate_runtime_hours(minutes):
    """Convert runtime from minutes to hours and minutes."""
    if not minutes:
        return "Unknown"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours > 0 and remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    elif hours > 0:
        return f"{hours}h"
    else:
        return f"{remaining_minutes}m" 