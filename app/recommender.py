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
        
        # Taste Clusters - include this new section
        if profile.get('taste_clusters'):
            summary.append("\nTASTE CLUSTERS:")
            for i, cluster in enumerate(profile['taste_clusters'], 1):
                cluster_summary = f"  Cluster {i}: {cluster.get('name', 'Unnamed Cluster')}"
                cluster_summary += f"\n    Description: {cluster.get('description', 'No description')}"
                
                if cluster.get('genres'):
                    cluster_summary += f"\n    Genres: {', '.join(cluster['genres'][:5])}"
                    
                if cluster.get('keywords'):
                    cluster_summary += f"\n    Keywords: {', '.join(cluster['keywords'][:5])}"
                    
                summary.append(cluster_summary)
        
        # Removed "Recently Watched" and "Highest Rated Shows" sections to reduce bias
        
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
You are a TV show recommendation expert with deep knowledge of storytelling, genres, and viewing preferences. Based on a user's watching history and taste profile, you'll recommend {num_recommendations} TV shows from a list of candidates that they would enjoy the most.

# USER PROFILE
{profile_summary}

# CANDIDATE SHOWS
{candidates_data}

IMPORTANT: Avoid anchoring bias in your recommendations. Don't overemphasize any single aspect of the user's profile (like the most frequent genre or most recent shows). Consider their full spectrum of preferences holistically and with equal weight.

Based on the user's viewing history and taste profile, select {num_recommendations} TV shows from the candidates list that the user would most likely enjoy. Your selections should:

1. BALANCED CLUSTER MATCHING: Each recommendation should clearly match one or more of the user's taste clusters, but don't favor any single cluster exclusively. Look for candidates that might appeal across multiple clusters or represent an evolution of their tastes.

2. PROVIDE DIVERSITY: Include at least one recommendation that might surprise the user but still aligns with their underlying preferences in a less obvious way. Look beyond surface-level genre matching to deeper thematic or stylistic connections.

3. RESPECT VIEWING MODES: Consider different viewing experiences the user might enjoy based on their history:
   - Binge-worthy shows with compelling season-long arcs
   - Standalone episodic content for casual viewing
   - Genre-bending or innovative format shows that push boundaries
   - Shows that match their preferred episode length and season count

4. COMPLEMENT EXISTING TASTES: Look for shows that fill gaps in their viewing history or represent high-quality examples of genres/themes they've shown interest in.

5. AVOID RECOMMENDATION TRAPS: Don't fall into these common recommendation pitfalls:
   - Overemphasizing the most frequent genres and ignoring occasional interests
   - Focusing too much on prestige or popularity metrics
   - Assuming that more recent viewing choices are more important than older ones
   - Recommending only what is most similar rather than what might genuinely interest them

For each recommendation, provide:
1. The show title and TMDB_ID
2. A personalized explanation that connects this show to specific aspects of their taste profile (mention cluster names when relevant)
3. Why this particular show stands out from the candidates and how it differs from what they've already watched
4. One specific viewing mode or context that makes this show particularly enjoyable (e.g., "Perfect for weekend binge-watching," "Great for episodic viewing," etc.)

Your response should be in this JSON format:
{{
  "recommendations": [
    {{
      "title": "Show Title",
      "tmdb_id": 12345,
      "explanation": "Detailed explanation of why this matches their preferences, including which taste clusters it aligns with...",
      "unique_appeal": "What makes this show fresh or different from what they've already watched...",
      "viewing_suggestion": "A specific viewing context recommendation"
    }},
    // Additional recommendations...
  ],
  "overall_explanation": "A brief summary explaining the overall recommendation strategy, how these shows complement each other, and how they address different aspects of the user's taste profile..."
}}

Only provide the JSON output, nothing else. Ensure it's valid JSON.
"""
        
        # Log the prompt
        logger.info("OpenAI Prompt:")
        logger.info(prompt)
        
        # Print the prompt to the console if enabled
        # if os.environ.get("SHOW_OPENAI_PROMPT", "0") == "1":
        #     print("\n=== OPENAI PROMPT ===\n")
        #     print(prompt)
        #     print("\n=== END PROMPT ===\n")
        
        try:
            # Make the OpenAI API call using the new OpenAI API syntax
            logger.info("Calling OpenAI API with configuration:")
            logger.info(f"Model: gpt-4o-mini, Temperature: 0.7, Max tokens: 2000")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using GPT-4o mini for faster, more efficient recommendations
                messages=[
                    {"role": "system", "content": "You are a TV show recommendation expert. Focus on providing balanced, unbiased recommendations by considering the user's entire taste profile holistically. Avoid common anchoring biases like overweighting recent shows, most frequent genres, or most prominent keywords. Pay special attention to the user's diverse taste clusters and consider them with equal importance. Aim to provide recommendations that both match existing preferences and thoughtfully expand their horizons with carefully selected new experiences."},
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
                
                # Ensure all fields exist, even if empty
                if 'viewing_suggestion' not in recommendation:
                    recommendation['viewing_suggestion'] = ""
                if 'unique_appeal' not in recommendation:
                    recommendation['unique_appeal'] = ""
                if 'explanation' not in recommendation:
                    recommendation['explanation'] = "Recommended based on your viewing history."
            
            return recommendations_json
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OpenAI recommendations: {e}")
            logger.error(f"Raw data: {recommendations_data}")
            
            # Return a fallback response
            return {
                "recommendations": [],
                "overall_explanation": f"Error parsing recommendations. Please try again."
            } 