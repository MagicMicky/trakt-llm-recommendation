"""
Module for generating TV show recommendations using OpenAI.
"""

import logging
import json
from typing import List, Dict, Any
import os
from openai import OpenAI
import time
import random
import requests

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
    
    def get_recommendation_candidates(self, user_profile: Dict[str, Any], watched_shows: List[Dict[str, Any]], max_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Get candidate shows for recommendations based on user profile and affinity scores.
        
        Args:
            user_profile: User taste profile with affinity data
            watched_shows: List of shows the user has already watched
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate shows for recommendations
        """
        logger.info(f"Fetching candidate shows based on user profile")
        
        # Extract watched show IDs to avoid recommending shows the user has already seen
        watched_tmdb_ids = set()
        for show in watched_shows:
            tmdb_id = show.get('tmdb_id')
            if tmdb_id:
                watched_tmdb_ids.add(tmdb_id)
        
        logger.info(f"User has watched {len(watched_tmdb_ids)} unique shows with TMDB IDs")
        
        # Get candidate shows from multiple sources
        candidates = []
        
        # 1. Discover shows based on favorite genres from the profile
        favorite_genres = []
        if user_profile.get('genres', {}).get('top'):
            favorite_genres = [g['name'] for g in user_profile['genres']['top'][:5]]
        
        # Add genre-based candidates
        genre_candidates = self._discover_shows_by_genres(favorite_genres, watched_tmdb_ids)
        candidates.extend(genre_candidates)
        logger.info(f"Found {len(genre_candidates)} genre-based candidates")
        
        # 2. Find similar shows based on favorites if available
        favorite_shows = user_profile.get('favorite_shows', [])
        if favorite_shows:
            # Use only top 5 favorites to avoid too many API calls
            for show in favorite_shows[:5]:
                tmdb_id = show.get('tmdb_id')
                if tmdb_id:
                    similar_shows = self._get_similar_shows(tmdb_id, watched_tmdb_ids)
                    candidates.extend(similar_shows)
            
            logger.info(f"Added similar shows based on user favorites")
        
        # 3. Consider viewing patterns when fetching candidates
        viewing_patterns = user_profile.get('viewing_patterns', {})
        viewing_mode = viewing_patterns.get('viewing_mode', 'unknown')
        
        # If user is a binge watcher, prioritize shows with multiple seasons
        if viewing_mode == 'binge_watcher':
            bingeable_candidates = self._discover_shows_with_params(
                {"with_runtime.gte": 40, "vote_average.gte": 7},
                watched_tmdb_ids,
                limit=20
            )
            candidates.extend(bingeable_candidates)
            logger.info(f"Added {len(bingeable_candidates)} bingeable show candidates")
        
        # If user is a completionist, find shows with completed runs
        elif viewing_mode == 'completionist':
            completed_candidates = self._discover_shows_with_params(
                {"status": "Ended", "vote_average.gte": 7.5},
                watched_tmdb_ids,
                limit=20
            )
            candidates.extend(completed_candidates)
            logger.info(f"Added {len(completed_candidates)} completed show candidates")
        
        # 4. Add trending shows to ensure we have enough candidates
        if len(candidates) < max_candidates:
            trending_candidates = self._get_trending_shows(watched_tmdb_ids)
            candidates.extend(trending_candidates)
            logger.info(f"Added {len(trending_candidates)} trending show candidates")
        
        # Remove duplicates while preserving order
        unique_candidates = self._remove_duplicate_candidates(candidates)
        
        # Limit to max_candidates
        candidates = unique_candidates[:max_candidates]
        
        logger.info(f"Selected {len(candidates)} total candidate shows for recommendations")
        return candidates
    
    def _discover_shows_by_genres(self, genres: List[str], watched_ids: set, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Discover shows based on genres.
        
        Args:
            genres: List of genres to match
            watched_ids: Set of watched show IDs to exclude
            limit: Maximum number of recommendations to return
            
        Returns:
            List of discovered shows
        """
        if not genres:
            return []
            
        logger.info(f"Discovering shows with genres: {genres}")
        
        discovered_shows = []
        
        # Convert genre names to TMDB genre IDs
        genre_id_map = {
            "Action & Adventure": 10759,
            "Animation": 16,
            "Comedy": 35,
            "Crime": 80,
            "Documentary": 99,
            "Drama": 18,
            "Family": 10751,
            "Kids": 10762,
            "Mystery": 9648,
            "News": 10763,
            "Reality": 10764,
            "Sci-Fi & Fantasy": 10765,
            "Soap": 10766,
            "Talk": 10767,
            "War & Politics": 10768,
            "Western": 37
        }
        
        # Match genre names to IDs
        genre_ids = []
        for genre in genres:
            if genre in genre_id_map:
                genre_ids.append(genre_id_map[genre])
        
        if not genre_ids:
            return []
            
        try:
            # Make API request to TMDB
            for genre_id in genre_ids[:3]:  # Limit to 3 genres to avoid too many API calls
                params = {
                    "api_key": os.getenv("TMDB_API_KEY"),
                    "with_genres": genre_id,
                    "sort_by": "popularity.desc",
                    "page": 1,
                    "language": "en-US"
                }
                
                response = requests.get(
                    "https://api.themoviedb.org/3/discover/tv",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for show in results:
                        # Skip shows the user has already watched
                        if show.get("id") in watched_ids:
                            continue
                            
                        # Process the show data
                        processed_show = self._process_tmdb_show(show)
                        discovered_shows.append(processed_show)
                        
                        # Break if we have enough shows
                        if len(discovered_shows) >= limit:
                            break
                
                # Add delay to avoid rate limiting
                time.sleep(0.25)
                
                # Break if we have enough shows
                if len(discovered_shows) >= limit:
                    break
                    
        except Exception as e:
            logger.error(f"Error discovering shows by genres: {e}")
        
        return discovered_shows[:limit]
    
    def _discover_shows_with_params(self, params: Dict[str, Any], watched_ids: set, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Discover shows with specific parameters.
        
        Args:
            params: Parameters for discovery
            watched_ids: Set of watched show IDs to exclude
            limit: Maximum number of recommendations to return
            
        Returns:
            List of discovered shows
        """
        logger.info(f"Discovering shows with params: {params}")
        
        discovered_shows = []
        
        try:
            # Make API request to TMDB
            api_params = {
                "api_key": os.getenv("TMDB_API_KEY"),
                "sort_by": "popularity.desc",
                "page": 1,
                "language": "en-US"
            }
            
            # Add custom params
            for key, value in params.items():
                api_params[key] = value
            
            response = requests.get(
                "https://api.themoviedb.org/3/discover/tv",
                params=api_params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for show in results:
                    # Skip shows the user has already watched
                    if show.get("id") in watched_ids:
                        continue
                        
                    # Process the show data
                    processed_show = self._process_tmdb_show(show)
                    discovered_shows.append(processed_show)
                    
                    # Break if we have enough shows
                    if len(discovered_shows) >= limit:
                        break
                        
        except Exception as e:
            logger.error(f"Error discovering shows with params: {e}")
        
        return discovered_shows[:limit]
    
    def _get_similar_shows(self, tmdb_id: int, watched_ids: set, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get shows similar to a given show.
        
        Args:
            tmdb_id: TMDB ID of the show
            watched_ids: Set of watched show IDs to exclude
            limit: Maximum number of recommendations to return
            
        Returns:
            List of similar shows
        """
        logger.info(f"Getting shows similar to TMDB ID: {tmdb_id}")
        
        similar_shows = []
        
        try:
            # Make API request to TMDB
            params = {
                "api_key": os.getenv("TMDB_API_KEY"),
                "language": "en-US",
                "page": 1
            }
            
            response = requests.get(
                f"https://api.themoviedb.org/3/tv/{tmdb_id}/similar",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for show in results:
                    # Skip shows the user has already watched
                    if show.get("id") in watched_ids:
                        continue
                        
                    # Process the show data
                    processed_show = self._process_tmdb_show(show)
                    similar_shows.append(processed_show)
                    
                    # Break if we have enough shows
                    if len(similar_shows) >= limit:
                        break
                        
        except Exception as e:
            logger.error(f"Error getting similar shows: {e}")
        
        return similar_shows[:limit]
    
    def _get_trending_shows(self, watched_ids: set, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get trending TV shows.
        
        Args:
            watched_ids: Set of watched show IDs to exclude
            limit: Maximum number of recommendations to return
            
        Returns:
            List of trending shows
        """
        logger.info(f"Getting trending TV shows")
        
        trending_shows = []
        
        try:
            # Make API request to TMDB
            params = {
                "api_key": os.getenv("TMDB_API_KEY"),
                "language": "en-US"
            }
            
            response = requests.get(
                "https://api.themoviedb.org/3/trending/tv/week",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                for show in results:
                    # Skip shows the user has already watched
                    if show.get("id") in watched_ids:
                        continue
                        
                    # Process the show data
                    processed_show = self._process_tmdb_show(show)
                    trending_shows.append(processed_show)
                    
                    # Break if we have enough shows
                    if len(trending_shows) >= limit:
                        break
                        
        except Exception as e:
            logger.error(f"Error getting trending shows: {e}")
        
        return trending_shows[:limit]
    
    def _process_tmdb_show(self, show: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process show data from TMDB API response.
        
        Args:
            show: Show data from TMDB API
            
        Returns:
            Processed show data
        """
        # Extract genres if available
        genre_ids = show.get("genre_ids", [])
        genres = []
        
        # Map genre IDs to names
        genre_id_map = {
            10759: "Action & Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Family",
            10762: "Kids",
            9648: "Mystery",
            10763: "News",
            10764: "Reality",
            10765: "Sci-Fi & Fantasy",
            10766: "Soap",
            10767: "Talk",
            10768: "War & Politics",
            37: "Western"
        }
        
        for genre_id in genre_ids:
            if genre_id in genre_id_map:
                genres.append(genre_id_map[genre_id])
        
        # Create processed show
        processed_show = {
            "tmdb_id": show.get("id"),
            "title": show.get("name", "Unknown"),
            "overview": show.get("overview", ""),
            "first_air_date": show.get("first_air_date", ""),
            "vote_average": show.get("vote_average", 0),
            "popularity": show.get("popularity", 0),
            "genres": genres
        }
        
        # Add poster and backdrop URLs if available
        poster_path = show.get("poster_path")
        if poster_path:
            processed_show["poster_url"] = f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        backdrop_path = show.get("backdrop_path")
        if backdrop_path:
            processed_show["backdrop_url"] = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
        
        return processed_show
    
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
        
        # Add viewing patterns section
        if 'viewing_patterns' in profile:
            viewing_patterns = profile['viewing_patterns']
            summary.append("\nVIEWING PATTERNS:")
            
            # Describe viewing mode
            viewing_mode = viewing_patterns.get('viewing_mode', 'unknown')
            if viewing_mode == 'binge_watcher':
                summary.append("  Viewing Style: Tends to binge-watch shows rapidly")
            elif viewing_mode == 'completionist':
                summary.append("  Viewing Style: Typically watches shows to completion")
            elif viewing_mode == 'rewatcher':
                summary.append("  Viewing Style: Often rewatches favorite content")
            else:
                summary.append("  Viewing Style: Casual viewer")
            
            # Add format preferences
            format_prefs = viewing_patterns.get('format_preferences', [])
            if format_prefs:
                formatted_prefs = []
                for pref in format_prefs:
                    if pref == 'mini_series':
                        formatted_prefs.append("mini-series")
                    elif pref == 'long_running_series':
                        formatted_prefs.append("long-running series")
                    elif pref == 'anthology':
                        formatted_prefs.append("anthology series")
                    else:
                        formatted_prefs.append(pref)
                
                summary.append(f"  Format Preferences: {', '.join(formatted_prefs)}")
            
            # Add dropped genres (genres the user tends to abandon)
            dropped_genres = viewing_patterns.get('dropped_genres', [])
            if dropped_genres:
                summary.append(f"  Often Drops: Shows in these genres: {', '.join(dropped_genres[:3])}")
        
        # Add favorite shows based on affinity scores
        if 'favorite_shows' in profile and profile['favorite_shows']:
            summary.append("\nFAVORITE SHOWS (Based on Watch Patterns):")
            for i, show in enumerate(profile['favorite_shows'][:5], 1):
                title = show.get('title', 'Unknown')
                genres = ", ".join(show.get('genres', [])[:3])
                score = show.get('affinity_score', 0)
                summary.append(f"  {i}. {title} - Genres: {genres} - Affinity Score: {score}")
        
        # Add binged shows
        if 'binged_shows' in profile and profile['binged_shows']:
            summary.append("\nBINGED SHOWS:")
            for i, show in enumerate(profile['binged_shows'][:3], 1):
                title = show.get('title', 'Unknown')
                days = show.get('time_span_days', 'Unknown')
                episodes = show.get('episodes', 0)
                if days and episodes > 1:
                    rate = f"{days / episodes:.1f} days/episode"
                    summary.append(f"  {i}. {title} - {episodes} episodes in {days} days ({rate})")
                else:
                    summary.append(f"  {i}. {title} - Binged rapidly")
        
        # Taste Clusters
        if profile.get('taste_clusters'):
            summary.append("\nTASTE CLUSTERS:")
            for i, cluster in enumerate(profile['taste_clusters'], 1):
                cluster_summary = f"  Cluster {i}: {cluster.get('name', 'Unnamed Cluster')}"
                cluster_summary += f"\n    Description: {cluster.get('description', 'No description')}"
                
                if cluster.get('genres'):
                    cluster_summary += f"\n    Genres: {', '.join(cluster['genres'][:5])}"
                    
                if cluster.get('keywords'):
                    cluster_summary += f"\n    Keywords: {', '.join(cluster['keywords'][:5])}"
                
                if cluster.get('example_shows'):
                    cluster_summary += f"\n    Example Shows: {', '.join(cluster['example_shows'][:3])}"
                    
                summary.append(cluster_summary)
                
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