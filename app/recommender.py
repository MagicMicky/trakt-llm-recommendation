"""
Module for generating TV show recommendations using OpenAI.
"""

import logging
import json
from typing import List, Dict, Any
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class Recommender:
    """Class to generate TV show recommendations using OpenAI."""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the Recommender.
        
        Args:
            openai_api_key: OpenAI API key
        """
        # Set API key as environment variable which is the recommended approach
        os.environ["OPENAI_API_KEY"] = openai_api_key
        # Initialize client without any parameters
        self.client = OpenAI()
    
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
        logger.info(f"Generating {num_recommendations} TV show recommendations")
        
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
        
        # Prepare user profile summary
        profile_summary = self._prepare_profile_summary(user_profile)
        
        # Prepare candidate shows data
        candidates_data = self._prepare_candidates_data(candidates)
        
        # Generate recommendations using OpenAI
        recommendations_data = self._get_openai_recommendations(
            profile_summary=profile_summary,
            candidates_data=candidates_data,
            num_recommendations=num_recommendations
        )
        
        # Process and format the recommendations
        recommendations = self._process_recommendations(recommendations_data, candidates)
        
        # Log a preview of the recommendations
        logger.info("Recommendations generated:")
        rec_titles = [rec.get('title', 'Unknown') for rec in recommendations.get('recommendations', [])]
        logger.info(f"Shows recommended: {', '.join(rec_titles)}")
        logger.info(f"Overall explanation: {recommendations.get('overall_explanation', '')[:100]}...")
        
        logger.info("TV show recommendations generated successfully")
        
        return recommendations
    
    def _prepare_profile_summary(self, profile: Dict[str, Any]) -> str:
        """
        Prepare a summary of the user's taste profile for the OpenAI prompt.
        
        Args:
            profile: User taste profile
            
        Returns:
            Summary string
        """
        summary = []
        
        # Basic stats
        summary.append(f"TOTAL SHOWS WATCHED: {profile['total_shows_watched']}")
        
        # Favorite genres
        if profile['genres']['top']:
            genres = ", ".join([f"{g['name']} ({g['count']})" for g in profile['genres']['top'][:5]])
            summary.append(f"FAVORITE GENRES: {genres}")
        
        # Favorite decades
        if profile['decades']['distribution']:
            decades = ", ".join([f"{decade}s ({count})" for decade, count in profile['decades']['distribution'].items()])
            summary.append(f"DECADES: {decades}")
        
        # Keywords/themes
        if profile['keywords']['top']:
            keywords = ", ".join([f"{k['name']} ({k['count']})" for k in profile['keywords']['top'][:10]])
            summary.append(f"COMMON THEMES: {keywords}")
        
        # Creators
        if profile['creators']['top']:
            creators = ", ".join([f"{c['name']} ({c['count']})" for c in profile['creators']['top'][:5]])
            summary.append(f"FAVORITE CREATORS: {creators}")
        
        # Actors
        if profile['actors']['top']:
            actors = ", ".join([f"{a['name']} ({a['count']})" for a in profile['actors']['top'][:5]])
            summary.append(f"FREQUENTLY WATCHED ACTORS: {actors}")
        
        # Recent shows
        if profile['recent_shows']:
            recent = ", ".join([show.get('title', 'Unknown') for show in profile['recent_shows'][:5]])
            summary.append(f"RECENTLY WATCHED: {recent}")
        
        # Top-rated shows
        if profile['top_rated_shows']:
            top_rated = ", ".join([show.get('title', 'Unknown') for show in profile['top_rated_shows'][:5]])
            summary.append(f"HIGHEST RATED SHOWS: {top_rated}")
        
        return "\n".join(summary)
    
    def _prepare_candidates_data(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Prepare candidate shows data for the OpenAI prompt.
        
        Args:
            candidates: List of candidate shows
            
        Returns:
            Formatted candidate data string
        """
        candidate_items = []
        
        for i, show in enumerate(candidates, 1):
            item = [
                f"SHOW #{i}:",
                f"TITLE: {show.get('title', 'Unknown')}",
                f"OVERVIEW: {show.get('overview', 'No overview available')}",
                f"FIRST AIR DATE: {show.get('first_air_date', 'Unknown')}"
            ]
            
            # Add TMDB ID for reference
            if show.get('tmdb_id'):
                item.append(f"TMDB_ID: {show.get('tmdb_id')}")
            
            # Add genres if available
            genres = show.get('genres', [])
            if genres:
                item.append(f"GENRES: {', '.join(genres)}")
            
            # Add vote average if available
            vote_average = show.get('vote_average')
            if vote_average:
                item.append(f"RATING: {vote_average}")
            
            candidate_items.append("\n".join(item))
        
        return "\n\n".join(candidate_items)
    
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
        # Construct the prompt
        prompt = f"""
You are a TV show recommendation expert. Based on a user's watching history and preferences, you'll recommend {num_recommendations} TV shows from a list of candidates that they would enjoy the most.

# USER PROFILE
{profile_summary}

# CANDIDATE SHOWS
{candidates_data}

Based on the user's viewing history and preferences, select the {num_recommendations} TV shows from the candidates that they would most likely enjoy and explain why.

For each recommendation, provide:
1. The show title and TMDB_ID
2. A personalized explanation of why this show matches their taste profile (be specific about which aspects of their profile match this show)
3. How this show differs from the others they've watched (what makes it fresh or new)

Your response should be in this JSON format:
{{
  "recommendations": [
    {{
      "title": "Show Title",
      "tmdb_id": 12345,
      "explanation": "Detailed explanation of why this matches their preferences...",
      "unique_appeal": "What makes this show fresh or different from what they've already watched..."
    }},
    // Additional recommendations...
  ],
  "overall_explanation": "A brief summary explaining the overall recommendation strategy and how these shows complement each other..."
}}

Only provide the JSON output, nothing else.
"""
        
        # Log the prompt
        logger.info("OpenAI Prompt:")
        logger.info(prompt)
        
        try:
            # Make the OpenAI API call using the new OpenAI API syntax
            logger.info("Calling OpenAI API with configuration:")
            logger.info(f"Model: gpt-4, Temperature: 0.7, Max tokens: 2000")
            
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for better recommendations
                messages=[
                    {"role": "system", "content": "You are a TV show recommendation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Log successful response
            logger.info("OpenAI API call completed successfully")
            
            # Extract and return the response content - new API returns content differently
            return response.choices[0].message.content.strip()
            
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
            # Parse the JSON response
            recommendations_json = json.loads(recommendations_data)
            
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
            
            return recommendations_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI recommendations: {e}")
            logger.error(f"Raw data: {recommendations_data}")
            
            # Return a fallback response
            return {
                "recommendations": [],
                "overall_explanation": f"Error parsing recommendations. Please try again."
            } 