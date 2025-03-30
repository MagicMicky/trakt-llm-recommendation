"""
Module for generating TV show recommendations using OpenAI.
"""

import logging
import json
from typing import List, Dict, Any, Set
import os
from app.llm_service import LLMService
from app.tmdb_client import TMDBClient
from app.prompt_engineering import PromptEngineering

logger = logging.getLogger(__name__)

class Recommender:
    """Class to generate TV show recommendations using OpenAI."""
    
    def __init__(self, openai_api_key: str, tmdb_api_key: str = None):
        """
        Initialize the Recommender.
        
        Args:
            openai_api_key: OpenAI API key
            tmdb_api_key: TMDB API key, if not provided will use TMDB_API_KEY env var
        """
        # Initialize the LLM service
        self.llm_service = LLMService(api_key=openai_api_key)
        
        # Initialize the TMDB client
        self.tmdb_client = TMDBClient(api_key=tmdb_api_key)
        
        # Initialize the prompt engineering
        self.prompt_engineering = PromptEngineering()
    
    def get_recommendation_candidates(self, user_profile: Dict[str, Any], watched_shows: List[Dict[str, Any]], max_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Get candidate shows for recommendations based on trending shows from the last week.
        
        Args:
            user_profile: User taste profile with affinity data
            watched_shows: List of shows the user has already watched
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate shows for recommendations
        """
        logger.info(f"Fetching trending shows for recommendations (max: {max_candidates})")
        
        # Extract watched show IDs to avoid recommending shows the user has already seen
        watched_tmdb_ids = {show.get('tmdb_id') for show in watched_shows if show.get('tmdb_id')}
        
        # Always request 100 trending shows (or the specified max_candidates)
        trending_candidates = self.tmdb_client.get_trending_shows(watched_tmdb_ids, limit=max_candidates)
        
        logger.info(f"Found {len(trending_candidates)} trending shows for recommendation (from {len(watched_shows)} watched shows)")
        return trending_candidates
    
    def _remove_duplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate candidates while preserving order.
        
        Args:
            candidates: List of candidate shows
            
        Returns:
            List of unique candidate shows
        """
        seen_ids = set()
        unique_candidates = []
        
        for candidate in candidates:
            tmdb_id = candidate.get("tmdb_id")
            if tmdb_id and tmdb_id not in seen_ids:
                seen_ids.add(tmdb_id)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def generate_recommendations(self, user_profile: Dict[str, Any], candidates: List[Dict[str, Any]], num_recommendations: int = 5) -> Dict[str, Any]:
        """
        Generate TV show recommendations using OpenAI.
        
        Args:
            user_profile: User taste profile
            candidates: List of candidate shows
            num_recommendations: Number of recommendations to generate
            
        Returns:
            Recommendations with explanation
        """
        logger.info(f"Generating {num_recommendations} recommendations using strategy: personalized")
        
        # Check if we have enough candidates
        available_candidates = len(candidates)
        if available_candidates == 0:
            logger.error("No candidate shows available for recommendations")
            return {
                "recommendations": [],
                "overall_explanation": "We couldn't find any new shows to recommend at this time. This could be due to temporary API limitations. Please try again later."
            }
        
        # Adjust number of recommendations if we don't have enough candidates
        if available_candidates < num_recommendations:
            logger.warning(f"Only {available_candidates} candidates available, reducing recommendations from {num_recommendations}")
            num_recommendations = available_candidates
        
        # Prepare user profile summary using PromptEngineering
        profile_summary = self.prompt_engineering.format_profile_summary(user_profile)
        
        # Prepare candidate shows data using PromptEngineering
        candidates_data = self.prompt_engineering.format_candidates_data(candidates)
        
        # Generate recommendations using OpenAI
        recommendations_data = self._get_openai_recommendations(
            profile_summary=profile_summary,
            candidates_data=candidates_data,
            num_recommendations=num_recommendations
        )
        
        # Process and format the recommendations
        recommendations = self._process_recommendations(recommendations_data, candidates)
        
        # Log a concise summary of the recommendations
        if recommendations.get('recommendations'):
            show_titles = [f"{rec.get('title')}" for rec in recommendations.get('recommendations', [])]
            logger.info(f"Generated {len(show_titles)} recommendations: {', '.join(show_titles)}")
        
        return recommendations
    
    def _get_openai_recommendations(self, profile_summary: str, candidates_data: str, num_recommendations: int) -> str:
        """
        Get recommendations from OpenAI.
        
        Args:
            profile_summary: User profile summary
            candidates_data: Formatted candidate data
            num_recommendations: Number of recommendations to generate
            
        Returns:
            OpenAI response containing recommendations
        """
        # Get prompt and system message from PromptEngineering
        prompt_data = self.prompt_engineering.format_recommendation_prompt(
            profile_summary=profile_summary,
            candidates_data=candidates_data,
            num_recommendations=num_recommendations
        )
        
        try:
            # Use the LLM service to generate recommendations
            return self.llm_service.generate_text(
                prompt=prompt_data["prompt"],
                system_message=prompt_data["system_message"],
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=2000
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations with OpenAI: {e}")
            # Return a simple error message in JSON format
            return json.dumps({
                "recommendations": [],
                "overall_explanation": f"Error generating recommendations: {str(e)}"
            })
    
    def _process_recommendations(self, recommendations_data: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and format the recommendations from OpenAI.
        
        Args:
            recommendations_data: Raw recommendations data from OpenAI
            candidates: List of candidate shows
            
        Returns:
            Processed recommendations
        """
        try:
            # Parse the JSON response using the service's JSON parsing function
            recommendations_json = self.llm_service._parse_json(recommendations_data)
            
            # Map of TMDB IDs to candidate show data
            candidates_map = {str(show.get('tmdb_id')): show for show in candidates if show.get('tmdb_id')}
            
            # Process each recommendation
            for recommendation in recommendations_json.get('recommendations', []):
                tmdb_id = str(recommendation.get('tmdb_id'))
                
                # Add additional show data from candidates
                if tmdb_id in candidates_map:
                    candidate = candidates_map[tmdb_id]
                    recommendation['poster_url'] = candidate.get('poster_url')
                    recommendation['backdrop_url'] = candidate.get('backdrop_url')
                    recommendation['overview'] = candidate.get('overview')
                    recommendation['first_air_date'] = candidate.get('first_air_date')
                
                # Ensure all fields exist, even if empty
                if 'viewing_suggestion' not in recommendation:
                    recommendation['viewing_suggestion'] = ""
                if 'unique_appeal' not in recommendation:
                    recommendation['unique_appeal'] = ""
                if 'explanation' not in recommendation:
                    recommendation['explanation'] = "Recommended based on your viewing history."
            
            return recommendations_json
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI recommendations: {e}")
            logger.error(f"Raw data: {recommendations_data}")
            
            # Return a fallback response
            return {
                "recommendations": [],
                "overall_explanation": f"Error parsing recommendations. Please try again."
            } 