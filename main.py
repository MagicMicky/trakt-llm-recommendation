#!/usr/bin/env python3
"""
TV Show Recommender System
--------------------------
A self-hosted TV show recommender that fetches your watch history from Trakt,
enriches it with TMDB metadata, builds your taste profile, and uses OpenAI to
generate personalized recommendations.
"""

import os
import logging
import argparse
from dotenv import load_dotenv
from flask import Flask, render_template

# Import modules
from app.trakt_fetcher import TraktFetcher
from app.tmdb_enricher import TMDBEnricher
from app.profile_builder import ProfileBuilder
from app.candidate_fetcher import CandidateFetcher 
from app.recommender import Recommender
from app.output_web import create_app
from app.utils import setup_logging, cache_recommendations, get_cached_recommendations

# Load environment variables
load_dotenv()

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def generate_recommendations():
    """Generate recommendations and return them."""
    try:
        logger.info("Starting TV show recommendation process")
        
        # Check for cached recommendations
        cache_file = 'data/recommendations_cache.json'
        cached_recommendations = get_cached_recommendations(cache_file)
        if cached_recommendations:
            logger.info("Using cached recommendations")
            return cached_recommendations
        
        # Initialize components
        trakt_fetcher = TraktFetcher(
            client_id=os.getenv("TRAKT_CLIENT_ID"),
            client_secret=os.getenv("TRAKT_CLIENT_SECRET"),
            access_token=os.getenv("TRAKT_ACCESS_TOKEN"),
            username=os.getenv("TRAKT_USERNAME")
        )
        
        tmdb_enricher = TMDBEnricher(
            api_key=os.getenv("TMDB_API_KEY")
        )
        
        profile_builder = ProfileBuilder()
        
        candidate_fetcher = CandidateFetcher(
            tmdb_enricher=tmdb_enricher
        )
        
        recommender = Recommender(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Enable prompt visibility - set environment variable to control if we want to see the prompt
        os.environ["SHOW_OPENAI_PROMPT"] = "1"
        
        # Step 1: Fetch watch history from Trakt
        logger.info("Fetching watch history from Trakt")
        watch_history = trakt_fetcher.get_watched_shows()
        
        # Step 2: Enrich watch history with TMDB metadata
        logger.info("Enriching watch history with TMDB metadata")
        enriched_history = tmdb_enricher.enrich_shows(watch_history)
        
        # Step 3: Build user taste profile
        logger.info("Building user taste profile")
        user_profile = profile_builder.build_profile(enriched_history)
        
        # Step 4: Get candidate shows (trending and unseen)
        logger.info("Fetching candidate shows")
        candidates = candidate_fetcher.get_candidates(watch_history)
        
        # Step 5: Generate recommendations using OpenAI
        logger.info("Generating recommendations")
        recommendations = recommender.generate_recommendations(
            user_profile=user_profile,
            candidates=candidates
        )
        
        # Cache the recommendations for future use
        logger.info("Caching recommendations")
        cache_recommendations(recommendations, cache_file)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in recommendation process: {e}", exc_info=True)
        return {"recommendations": [], "overall_explanation": f"Error: {str(e)}"}

def main():
    """Main function that orchestrates the system."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='TV Show Recommender')
        parser.add_argument('--force-refresh', action='store_true', 
                            help='Force refresh recommendations ignoring cache')
        parser.add_argument('--port', type=int, default=5000,
                            help='Port for the web server (default: 5000)')
        parser.add_argument('--debug', action='store_true',
                            help='Run Flask in debug mode (enables auto-reload)')
        args = parser.parse_args()
        
        # If force refresh, remove the cache file
        if args.force_refresh:
            cache_file = 'data/recommendations_cache.json'
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info("Removed cache file, will generate fresh recommendations")
        
        # Generate recommendations (will use cache if available and not forced refresh)
        recommendations = generate_recommendations()
        
        # Configure Flask environment
        # By default, disable auto-reloading to prevent regenerating recommendations
        os.environ['FLASK_ENV'] = 'production'  # Use production mode by default
        os.environ['FLASK_DEBUG'] = '0'  # Disable debug mode by default
        
        if args.debug:
            os.environ['FLASK_ENV'] = 'development'
            os.environ['FLASK_DEBUG'] = '1'
            logger.warning("Running in debug mode with auto-reload enabled - recommendation process may run multiple times")
        
        # Start the web server
        logger.info(f"Starting web server on port {args.port}")
        app = create_app(recommendations)
        app.run(host="0.0.0.0", port=args.port, use_reloader=args.debug)
        
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 