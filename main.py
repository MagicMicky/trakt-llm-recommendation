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
        
        # Enable prompt visibility - set environment variable to control if we want to see the prompt
        os.environ["SHOW_OPENAI_PROMPT"] = "1"
        
        # Step 1: Fetch watch history from Trakt
        logger.info("Fetching watch history from Trakt")
        watch_history = trakt_fetcher.get_watched_shows()
        
        if not watch_history:
            logger.error("No watch history found")
            return {"recommendations": [], "overall_explanation": "No watch history found. Please make sure you have watched shows on Trakt."}
        
        # Step 2: Fetch additional data for affinity scoring
        logger.info("Fetching detailed watch history and ratings for affinity scoring")
        
        # Fetch ratings data - this will be used for affinity scoring
        show_ratings = trakt_fetcher.get_ratings()
        
        # Fetch episode-level watch history for affinity scoring
        # Previously limited to 50 most recent shows, now we'll fetch all shows
        # Sort by last watched for better prioritization if we hit API limits
        sorted_shows = sorted(
            [s for s in watch_history if s.get('last_watched_at')],
            key=lambda x: x.get('last_watched_at', ''),
            reverse=True
        )
        
        # Get all show IDs for fetching episode history
        all_show_ids = [str(show.get('trakt_id')) for show in sorted_shows if show.get('trakt_id')]
        logger.info(f"Fetching episode history for all {len(all_show_ids)} shows (previously limited to 50)")
        
        # Fetch episode history for all shows
        episode_history = trakt_fetcher.get_all_episode_history(all_show_ids, max_shows=0)  # 0 means no limit
        
        logger.info(f"Fetched ratings for {len(show_ratings)} shows and episode history for {len(episode_history)} shows")
        
        # Add more detailed logging about the episode history
        if episode_history:
            total_episodes = sum(len(episodes) for episodes in episode_history.values())
            logger.info(f"Total episodes in watch history: {total_episodes}")
            
            # Log the distribution of episode counts per show
            episode_counts = [(next((s.get('title', f'Show ID {show_id}') for s in watch_history 
                                   if str(s.get('trakt_id')) == show_id), f'Show ID {show_id}'), 
                              len(episodes)) 
                             for show_id, episodes in episode_history.items()]
            
            episode_counts.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 shows by episode count:")
            for i, (title, count) in enumerate(episode_counts[:10], 1):
                logger.info(f"  {i}. {title}: {count} episodes")
            
            # Sample a few shows to examine their episode history structure
            sample_count = min(3, len(episode_history))
            if sample_count > 0:
                logger.info(f"Examining episode history data for {sample_count} sample shows:")
                sample_show_ids = list(episode_history.keys())[:sample_count]
                for show_id in sample_show_ids:
                    show_episodes = episode_history[show_id]
                    if show_episodes:
                        show_title = next((s.get('title', 'Unknown') for s in watch_history 
                                          if str(s.get('trakt_id')) == show_id), f"Show ID {show_id}")
                        logger.info(f"  Show '{show_title}' has {len(show_episodes)} episode records")
                        
                        # Check for unique episodes (in case of rewatches)
                        unique_episodes = set()
                        for record in show_episodes:
                            if 'episode' in record and 'ids' in record['episode'] and 'trakt' in record['episode']['ids']:
                                unique_episodes.add(record['episode']['ids']['trakt'])
                        
                        logger.info(f"    Unique episodes: {len(unique_episodes)} (out of {len(show_episodes)} total records)")
                        
                        # Check if episodes have proper timestamps
                        has_timestamps = all('watched_at' in ep for ep in show_episodes)
                        logger.info(f"    All episodes have timestamps: {has_timestamps}")
                        if not has_timestamps:
                            logger.warning(f"    Missing timestamps in episode history for show '{show_title}'")
                        
                        # Log a sample episode
                        sample_ep = show_episodes[0]
                        logger.info(f"    Sample episode data: {sample_ep}")
                        
                        # Compare with show metadata
                        matched_show = next((s for s in watch_history if str(s.get('trakt_id')) == show_id), None)
                        if matched_show:
                            logger.info(f"    Reported watched_episodes in show data: {matched_show.get('watched_episodes', 0)}")
                            logger.info(f"    Reported plays in show data: {matched_show.get('plays', 0)}")
        
        # Step 3: Enrich watch history with TMDB metadata
        logger.info("Enriching watch history with TMDB metadata")
        enriched_history = tmdb_enricher.enrich_shows(watch_history)
        
        # Step 4: Build user taste profile with affinity data
        logger.info("Building user taste profile with affinity scoring")
        user_profile = profile_builder.build_profile(
            enriched_history, 
            episode_history=episode_history, 
            ratings=show_ratings
        )
        
        # Step 5: Generate recommendations using OpenAI
        logger.info("Generating recommendations using affinity data")
        
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