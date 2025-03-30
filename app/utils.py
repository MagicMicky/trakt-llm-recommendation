"""
Utility functions for the TV show recommender.
"""

import os
import logging
import json
from datetime import datetime
import logging.handlers

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration with improved organization and focus.
    
    Args:
        level: The default logging level (default: INFO)
    """
    # Ensure log directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create formatter for different log types
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger (console handler)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers if any
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Console handler - shows important logs in the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation - main app logs
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Dedicated handler for OpenAI prompts with rotation
    openai_handler = logging.handlers.RotatingFileHandler(
        'logs/openai_prompts.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=2
    )
    openai_handler.setLevel(logging.INFO)
    openai_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    
    # Create OpenAI logger
    openai_logger = logging.getLogger('openai.prompt')
    openai_logger.propagate = False  # Prevent duplicate logging
    openai_logger.addHandler(openai_handler)
    
    # Set more specific log levels for verbose libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Register logging completion
    logging.info("Logging configured at level %s", logging.getLevelName(level))
    logging.info("OpenAI prompt logging enabled at level %s", logging.getLevelName(logging.INFO))

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

def cache_recommendations(recommendations, cache_file='data/recommendations_cache.json'):
    """
    Save recommendations to a cache file.
    
    Args:
        recommendations: The recommendations data to cache
        cache_file: Path to the cache file
        
    Returns:
        bool: True if caching was successful, False otherwise
    """
    try:
        save_to_json(recommendations, cache_file)
        return True
    except Exception as e:
        logging.error(f"Error caching recommendations: {e}")
        return False

def get_cached_recommendations(cache_file='data/recommendations_cache.json', max_age_hours=24):
    """
    Retrieve cached recommendations if available and not expired.
    
    Args:
        cache_file: Path to the cache file
        max_age_hours: Maximum age of cache in hours before considered stale
        
    Returns:
        dict: Cached recommendations or None if not available or expired
    """
    try:
        if not os.path.exists(cache_file):
            return None
            
        # Check if cache file is too old
        file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
        if file_age > (max_age_hours * 3600):  # Convert hours to seconds
            logging.info(f"Cache file is older than {max_age_hours} hours, will regenerate recommendations")
            return None
            
        return load_from_json(cache_file)
    except Exception as e:
        logging.error(f"Error retrieving cached recommendations: {e}")
        return None 