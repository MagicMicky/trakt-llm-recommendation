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
from dotenv import load_dotenv
from flask import Flask, render_template

# Import modules
from app.trakt_fetcher import TraktFetcher
from app.tmdb_enricher import TMDBEnricher
from app.profile_builder import ProfileBuilder
from app.candidate_fetcher import CandidateFetcher 
from app.recommender import Recommender
from app.output_web import create_app
from app.utils import setup_logging

# Load environment variables
load_dotenv()

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Main function that orchestrates the recommendation process."""
    try:
        logger.info("Starting TV show recommendation process")
        
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
        
        # Step 6: Start the web server to display recommendations
        logger.info("Starting web server to display recommendations")
        app = create_app(recommendations)
        app.run(host="0.0.0.0", port=5000)
        
    except Exception as e:
        logger.error(f"Error in recommendation process: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 