"""
Module for building a user taste profile from watch history.
"""

import logging
import json
import os
from typing import List, Dict, Any, Counter
from collections import Counter
import random
from openai import OpenAI
import datetime
import re

logger = logging.getLogger(__name__)

class ProfileBuilder:
    """Class to build a user taste profile from watched shows."""
    
    def __init__(self, openai_api_key=None):
        """
        Initialize the ProfileBuilder.
        
        Args:
            openai_api_key: Optional OpenAI API key for clustering. If not provided,
                           will use the OPENAI_API_KEY environment variable.
        """
        # If openai_api_key is provided, use it. Otherwise, rely on environment variable
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize the OpenAI client
        self.client = OpenAI()
    
    def build_profile(self, shows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a user taste profile from enriched watch history.
        
        Args:
            shows: List of enriched shows with TMDB data
            
        Returns:
            User taste profile
        """
        logger.info("Building user taste profile")
        
        # Initialize counters for various attributes
        genre_counter = Counter()
        network_counter = Counter()
        keyword_counter = Counter()
        creator_counter = Counter()
        actor_counter = Counter()
        
        # Initialize lists for top-rated and recently watched shows
        top_rated_shows = []
        recent_shows = []
        
        # Count shows by decade
        decades = {}
        
        # Process each show to extract profile data
        for show in shows:
            self._process_show_for_profile(
                show, 
                genre_counter, 
                network_counter, 
                keyword_counter, 
                creator_counter, 
                actor_counter,
                top_rated_shows,
                recent_shows,
                decades
            )
        
        # Sort the recently watched shows by last_watched_at
        recent_shows.sort(key=lambda x: x.get('last_watched_at', ''), reverse=True)
        recent_shows = recent_shows[:10]  # Limit to 10 most recent
        
        # Sort the top-rated shows by vote_average
        top_rated_shows.sort(key=lambda x: x.get('tmdb_data', {}).get('vote_average', 0), reverse=True)
        top_rated_shows = top_rated_shows[:10]  # Limit to 10 top-rated
        
        # Build taste clusters
        taste_clusters = self.build_clusters(shows)
        
        # Build the user profile
        profile = {
            'total_shows_watched': len(shows),
            'genres': {
                'top': [{'name': genre, 'count': count} for genre, count in genre_counter.most_common(10)],
                'all': dict(genre_counter)
            },
            'networks': {
                'top': [{'name': network, 'count': count} for network, count in network_counter.most_common(5)],
                'all': dict(network_counter)
            },
            'keywords': {
                'top': [{'name': keyword, 'count': count} for keyword, count in keyword_counter.most_common(20)],
                'all': dict(keyword_counter)
            },
            'creators': {
                'top': [{'name': creator, 'count': count} for creator, count in creator_counter.most_common(10)],
                'all': dict(creator_counter)
            },
            'actors': {
                'top': [{'name': actor, 'count': count} for actor, count in actor_counter.most_common(15)],
                'all': dict(actor_counter)
            },
            'decades': {
                'distribution': decades,
                'favorite': max(decades.items(), key=lambda x: x[1], default=(None, 0))[0]
            },
            'recent_shows': recent_shows,
            'top_rated_shows': top_rated_shows,
            'taste_clusters': taste_clusters
        }
        
        logger.info("User taste profile built successfully")
        
        return profile
    
    def build_clusters(self, shows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build taste clusters from the user's watched shows.
        
        Args:
            shows: List of enriched shows with TMDB data
            
        Returns:
            List of taste clusters, each with a description, genres, keywords, and example shows
        """
        logger.info("Building taste clusters from watch history")
        
        # Check for cached clusters
        cache_file = 'data/taste_clusters_cache.json'
        cached_clusters = self._get_cached_clusters(cache_file, shows)
        if cached_clusters:
            logger.info("Using cached taste clusters")
            return cached_clusters
            
        # Check if we've had a recent failure to avoid hitting the API again too soon
        failure_file = 'data/taste_clusters_failure.txt'
        if os.path.exists(failure_file):
            try:
                # Check when the last failure was
                failure_timestamp = os.path.getmtime(failure_file)
                current_time = datetime.datetime.now().timestamp()
                
                # If it's been less than 1 hour since the last failure, use the fallback
                if current_time - failure_timestamp < 3600:  # 3600 seconds = 1 hour
                    logger.info("Recent clustering failure detected, using fallback clusters")
                    shows_data = self._prepare_shows_for_clustering(shows)
                    clusters = self._generate_fallback_clusters(shows_data)
                    self._cache_clusters(clusters, cache_file, shows)
                    return clusters
            except Exception as e:
                logger.error(f"Error checking failure timestamp: {e}")
                # Continue with normal flow if error checking timestamp
        
        # Prepare show data for clustering
        shows_data = self._prepare_shows_for_clustering(shows)
        
        # Use OpenAI to identify clusters
        try:
            clusters = self._generate_clusters_with_llm(shows_data)
            
            # If successful, remove the failure marker if it exists
            if os.path.exists(failure_file):
                try:
                    os.remove(failure_file)
                    logger.info("Removed taste clustering failure marker")
                except Exception as e:
                    logger.error(f"Error removing failure marker: {e}")
                    
            # Cache the clusters
            self._cache_clusters(clusters, cache_file, shows)
            
            logger.info(f"Built {len(clusters)} taste clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error in taste cluster generation process: {e}")
            
            # Mark that we had a failure to avoid immediate retries
            try:
                os.makedirs(os.path.dirname(failure_file), exist_ok=True)
                with open(failure_file, 'w') as f:
                    f.write(f"Taste clustering failed at {datetime.datetime.now()}: {str(e)}")
                logger.info("Created taste clustering failure marker")
            except Exception as write_err:
                logger.error(f"Error writing failure marker: {write_err}")
            
            # Use fallback clustering instead
            clusters = self._generate_fallback_clusters(shows_data)
            self._cache_clusters(clusters, cache_file, shows)
            return clusters
    
    def _prepare_shows_for_clustering(self, shows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare show data for clustering by extracting relevant information.
        
        Args:
            shows: List of enriched shows
            
        Returns:
            List of simplified show data for clustering
        """
        prepared_shows = []
        
        for show in shows:
            tmdb_data = show.get('tmdb_data', {})
            
            prepared_show = {
                'title': show.get('title', 'Unknown'),
                'genres': tmdb_data.get('genres', []),
                'keywords': tmdb_data.get('keywords', []),
                'overview': tmdb_data.get('overview', ''),
                'first_air_date': tmdb_data.get('first_air_date', ''),
                'vote_average': tmdb_data.get('vote_average', 0)
            }
            
            prepared_shows.append(prepared_show)
        
        # If we have too many shows, sample a representative subset for efficiency
        if len(prepared_shows) > 50:
            # Sort by rating first to ensure we include some highly-rated shows
            sorted_shows = sorted(prepared_shows, key=lambda x: x.get('vote_average', 0), reverse=True)
            # Take top 20 and a random sample of 30 more
            sampled_shows = sorted_shows[:20] + random.sample(sorted_shows[20:], min(30, len(sorted_shows) - 20))
            logger.info(f"Sampled {len(sampled_shows)} shows for clustering out of {len(prepared_shows)} total")
            return sampled_shows
        
        return prepared_shows
    
    def _generate_clusters_with_llm(self, shows_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use the OpenAI LLM to generate taste clusters from show data.
        
        Args:
            shows_data: Prepared show data for clustering
            
        Returns:
            List of taste clusters
        """
        # Convert shows data to a formatted string for the prompt
        shows_text = self._format_shows_for_prompt(shows_data)
        
        # Create the prompt for the LLM
        prompt = f"""
You are an expert in analyzing media consumption patterns and clustering similar content.

Below is a list of TV shows that a user has watched:

{shows_text}

Based on this watch history, identify 3-5 distinct taste clusters or viewing preferences.

For each cluster:
1. Provide a short descriptive name (e.g., "Gritty Crime Dramas", "Lighthearted Comedies", "Thought-Provoking Sci-Fi")
2. List the key genres that define this cluster
3. List 5-8 keywords or themes common in this cluster
4. Include 3-5 example shows from the user's watch history that best represent this cluster
5. Write a brief 1-2 sentence description explaining what characterizes this taste profile

Your response should be in this JSON format:
{{
  "clusters": [
    {{
      "name": "Cluster name",
      "genres": ["Genre1", "Genre2", ...],
      "keywords": ["Keyword1", "Keyword2", ...],
      "example_shows": ["Show1", "Show2", ...],
      "description": "Brief description of this taste cluster"
    }},
    // Additional clusters...
  ]
}}

Only provide the JSON output, nothing else. Ensure it's valid JSON without any comments or trailing commas.
"""
        
        try:
            logger.info("Calling OpenAI to generate taste clusters")
            
            # Make the API call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using the same model as our recommender
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing media consumption patterns and identifying taste clusters. You will respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract the response content
            result = response.choices[0].message.content.strip()
            
            # Debug: print the raw response to console for debugging
            print("\n=== RAW OPENAI RESPONSE FOR TASTE CLUSTERS ===")
            print(result)
            print("=== END RAW RESPONSE ===\n")
            
            # Log the raw response for debugging
            logger.info(f"Raw OpenAI response length: {len(result)}")
            if len(result) < 50:  # If it's suspiciously short, log the whole thing
                logger.warning(f"Short response from OpenAI: '{result}'")
            else:
                logger.info(f"First 100 chars: {result[:100]}...")
            
            # Parse the JSON response
            try:
                # Sometimes OpenAI returns results with markdown code blocks, try to extract JSON
                if result.startswith("```json"):
                    result = result.split("```json", 1)[1]
                    if "```" in result:
                        result = result.split("```", 1)[0]
                elif result.startswith("```"):
                    result = result.split("```", 1)[1]
                    if "```" in result:
                        result = result.split("```", 1)[0]
                
                # Some basic cleanup
                result = result.strip()
                
                # Additional cleaning to fix common JSON issues
                # Try to fix trailing commas which are invalid in JSON but common in JavaScript
                result = result.replace(",\n}", "\n}")
                result = result.replace(",\n  }", "\n  }")
                result = result.replace(",\n    }", "\n    }")
                result = result.replace(",\n]", "\n]")
                result = result.replace(",\n  ]", "\n  ]")
                result = result.replace(",\n    ]", "\n    ]")
                
                # Fix missing quote marks around keys
                result = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', result)
                
                # Fix single quotes to double quotes if needed
                if "'" in result and '"' not in result:
                    result = result.replace("'", '"')
                
                # Try to parse JSON
                clusters_data = json.loads(result)
                clusters = clusters_data.get('clusters', [])
                
                # Check if we got a valid response
                if not clusters:
                    logger.warning("OpenAI returned an empty clusters list or missing 'clusters' key")
                    return self._generate_fallback_clusters(shows_data)
                
                return clusters
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing clusters JSON from OpenAI: {e}")
                logger.error(f"Raw response causing JSON error: {result}")
                # If JSON parsing fails, use fallback
                return self._generate_fallback_clusters(shows_data)
            
        except Exception as e:
            logger.error(f"Error generating taste clusters with OpenAI: {e}")
            # If an error occurs, use fallback
            return self._generate_fallback_clusters(shows_data)
    
    def _generate_fallback_clusters(self, shows_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate basic clusters without using the LLM as a fallback.
        
        Args:
            shows_data: Prepared show data
            
        Returns:
            List of basic taste clusters
        """
        logger.info("Using fallback clustering method")
        
        # Create a simplified genre-based clustering
        genre_counts = Counter()
        shows_by_genre = {}
        
        # Count genres and group shows by genre
        for show in shows_data:
            title = show.get('title', 'Unknown')
            genres = show.get('genres', [])
            
            for genre in genres:
                genre_counts[genre] += 1
                if genre not in shows_by_genre:
                    shows_by_genre[genre] = []
                shows_by_genre[genre].append(title)
        
        # Take top genres (up to 4)
        top_genres = [genre for genre, _ in genre_counts.most_common(4)]
        
        # Create basic clusters
        clusters = []
        
        for genre in top_genres:
            # Skip if we don't have enough shows for this genre
            if len(shows_by_genre.get(genre, [])) < 2:
                continue
                
            # Create a cluster
            cluster = {
                "name": f"{genre} Shows",
                "genres": [genre],
                "keywords": [genre.lower(), "entertainment", "television"],
                "example_shows": shows_by_genre[genre][:5],  # Up to 5 example shows
                "description": f"Shows in the {genre} genre that the user has watched."
            }
            
            clusters.append(cluster)
        
        # If we somehow ended up with no clusters, create a generic one
        if not clusters and shows_data:
            all_titles = [show.get('title', 'Unknown') for show in shows_data[:5]]
            clusters.append({
                "name": "General Entertainment",
                "genres": ["Mixed"],
                "keywords": ["entertainment", "television", "drama"],
                "example_shows": all_titles,
                "description": "Various shows the user has watched across different genres."
            })
        
        logger.info(f"Generated {len(clusters)} fallback clusters")
        return clusters
    
    def _format_shows_for_prompt(self, shows_data: List[Dict[str, Any]]) -> str:
        """
        Format show data into a string for the LLM prompt.
        
        Args:
            shows_data: Prepared show data
            
        Returns:
            Formatted string of show data
        """
        formatted_shows = []
        
        for i, show in enumerate(shows_data, 1):
            genres_str = ", ".join(show.get('genres', []))
            keywords_str = ", ".join(show.get('keywords', [])[:5])  # Limit to 5 keywords for brevity
            
            show_text = f"Show #{i}: {show.get('title', 'Unknown')}\n"
            show_text += f"Genres: {genres_str}\n"
            
            if keywords_str:
                show_text += f"Keywords: {keywords_str}\n"
                
            if show.get('overview'):
                # Truncate overview if it's too long
                overview = show.get('overview', '')
                if len(overview) > 150:
                    overview = overview[:147] + "..."
                show_text += f"Overview: {overview}\n"
                
            formatted_shows.append(show_text)
        
        return "\n\n".join(formatted_shows)
    
    def _get_cached_clusters(self, cache_file: str, shows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get cached taste clusters if available.
        
        Args:
            cache_file: Path to the cache file
            shows: List of shows to calculate a hash/fingerprint for comparison
            
        Returns:
            Cached clusters or None if cache doesn't exist or is invalid
        """
        try:
            if not os.path.exists(cache_file):
                return None
                
            # Calculate a simple fingerprint of the watch history to determine if cache is still valid
            # Just use the number of shows and a hash of their titles
            show_titles = sorted([show.get('title', '') for show in shows])
            current_fingerprint = f"{len(shows)}_{hash(tuple(show_titles))}"
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Check if the fingerprint matches
            if cache_data.get('fingerprint') == current_fingerprint:
                return cache_data.get('clusters', [])
            else:
                logger.info("Watch history has changed, regenerating taste clusters")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached clusters: {e}")
            return None
    
    def _cache_clusters(self, clusters: List[Dict[str, Any]], cache_file: str, shows: List[Dict[str, Any]] = None) -> None:
        """
        Cache taste clusters to a file.
        
        Args:
            clusters: List of taste clusters
            cache_file: Path to the cache file
            shows: List of shows to calculate a fingerprint for
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Create a simple fingerprint based on number of shows and titles
            fingerprint = None
            if shows:
                show_titles = sorted([show.get('title', '') for show in shows])
                fingerprint = f"{len(shows)}_{hash(tuple(show_titles))}"
            
            # This will be used to check if the cache is still valid when watch history changes
            cache_data = {
                'clusters': clusters,
                'timestamp': str(datetime.datetime.now()),
                'fingerprint': fingerprint
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Cached {len(clusters)} taste clusters")
            
        except Exception as e:
            logger.error(f"Error caching taste clusters: {e}")
    
    def _process_show_for_profile(
        self, 
        show: Dict[str, Any], 
        genre_counter: Counter, 
        network_counter: Counter, 
        keyword_counter: Counter, 
        creator_counter: Counter, 
        actor_counter: Counter,
        top_rated_shows: List,
        recent_shows: List,
        decades: Dict
    ) -> None:
        """
        Process a show to extract profile data.
        
        Args:
            show: Show data
            genre_counter: Counter for genres
            network_counter: Counter for networks
            keyword_counter: Counter for keywords
            creator_counter: Counter for creators
            actor_counter: Counter for actors
            top_rated_shows: List of top-rated shows
            recent_shows: List of recently watched shows
            decades: Dictionary of shows by decade
        """
        tmdb_data = show.get('tmdb_data', {})
        
        # Count genres
        genres = tmdb_data.get('genres', [])
        for genre in genres:
            genre_counter[genre] += 1
        
        # Count networks
        networks = tmdb_data.get('networks', [])
        for network in networks:
            network_counter[network] += 1
        
        # Count keywords
        keywords = tmdb_data.get('keywords', [])
        for keyword in keywords:
            keyword_counter[keyword] += 1
        
        # Count creators
        creators = tmdb_data.get('creators', [])
        for creator in creators:
            if isinstance(creator, dict) and 'name' in creator:
                creator_counter[creator['name']] += 1
            else:
                creator_counter[creator] += 1
        
        # Count actors
        cast = tmdb_data.get('cast', [])
        for actor in cast:
            if isinstance(actor, dict) and 'name' in actor:
                actor_counter[actor['name']] += 1
        
        # Add to top-rated shows if vote average is high
        vote_average = tmdb_data.get('vote_average', 0)
        if vote_average and vote_average >= 7.5:
            top_rated_shows.append(show)
        
        # Add to recently watched shows if last_watched_at is present
        if 'last_watched_at' in show:
            recent_shows.append(show)
        
        # Count by decade
        first_air_date = tmdb_data.get('first_air_date', '')
        if first_air_date and len(first_air_date) >= 4:
            year = int(first_air_date[:4])
            decade = (year // 10) * 10
            decades[decade] = decades.get(decade, 0) + 1
    
    def generate_profile_summary(self, profile: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the user taste profile.
        
        Args:
            profile: User taste profile
            
        Returns:
            Formatted summary string
        """
        summary = []
        
        # General stats
        summary.append(f"You've watched {profile['total_shows_watched']} TV shows.")
        
        # Genres
        if profile['genres']['top']:
            favorite_genres = ", ".join([genre['name'] for genre in profile['genres']['top'][:3]])
            summary.append(f"Your favorite genres are {favorite_genres}.")
        
        # Decades
        if profile['decades']['favorite']:
            summary.append(f"You seem to enjoy shows from the {profile['decades']['favorite']}s.")
        
        # Networks
        if profile['networks']['top']:
            favorite_networks = ", ".join([network['name'] for network in profile['networks']['top'][:2]])
            summary.append(f"You watch a lot of content from {favorite_networks}.")
        
        # Keywords/themes
        if profile['keywords']['top']:
            themes = ", ".join([keyword['name'] for keyword in profile['keywords']['top'][:5]])
            summary.append(f"Themes common in your viewing: {themes}.")
        
        # Creators
        if profile['creators']['top']:
            creators = ", ".join([creator['name'] for creator in profile['creators']['top'][:2]])
            summary.append(f"You enjoy shows from creators like {creators}.")
        
        # Actors
        if profile['actors']['top']:
            actors = ", ".join([actor['name'] for actor in profile['actors']['top'][:3]])
            summary.append(f"You frequently watch shows featuring {actors}.")
        
        return "\n".join(summary) 