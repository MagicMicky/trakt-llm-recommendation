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
from app.affinity_scorer import AffinityScorer
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
        
        profile_builder = ProfileBuilder(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        recommender = Recommender(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Step 1: Fetch watch history from Trakt
        logger.info("Fetching watch history from Trakt")
        watch_history = trakt_fetcher.get_watched_shows()
        
        if not watch_history:
            logger.error("No watch history found")
            return {"recommendations": [], "overall_explanation": "No watch history found. Please make sure you have watched shows on Trakt."}
        
        logger.info(f"Fetched {len(watch_history)} shows from Trakt")
        
        # Step 2: Fetch additional data for affinity scoring
        logger.info("Fetching ratings and episode history for affinity scoring")
        
        # Fetch ratings data - this will be used for affinity scoring
        show_ratings = trakt_fetcher.get_ratings()
        
        # Fetch episode-level watch history for affinity scoring
        sorted_shows = sorted(
            [s for s in watch_history if s.get('last_watched_at')],
            key=lambda x: x.get('last_watched_at', ''),
            reverse=True
        )
        
        # Get all show IDs for fetching episode history
        all_show_ids = [str(show.get('trakt_id')) for show in sorted_shows if show.get('trakt_id')]
        
        # Fetch episode history for all shows with a much higher episode limit for long-running shows
        logger.info("Fetching detailed episode history - this might take some time for users with extensive watch history")
        episode_history = trakt_fetcher.get_all_episode_history(
            all_show_ids, 
            max_shows=0,  # 0 means no limit on number of shows
            episode_limit=500  # Higher limit to ensure we capture more episodes for shows like The Simpsons
        )
        
        logger.info(f"Fetched ratings for {len(show_ratings)} shows and episode history for {len(episode_history)} shows")
        
        # Step 3: Enrich watch history with TMDB metadata
        logger.info("Enriching watch history with TMDB metadata")
        enriched_history = tmdb_enricher.enrich_shows(watch_history)
        
        # Step 4: Build user taste profile with affinity data
        logger.info("Building user taste profile")
        user_profile = profile_builder.build_profile(
            enriched_history, 
            episode_history=episode_history, 
            ratings=show_ratings
        )
        
        # Step 5: Generate recommendations using OpenAI
        logger.info("Generating personalized recommendations")
        
        # Use the enhanced user profile with affinity scores for recommendations
        recommendations = recommender.generate_recommendations(
            user_profile=user_profile,
            candidates=recommender.get_recommendation_candidates(user_profile, enriched_history)
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