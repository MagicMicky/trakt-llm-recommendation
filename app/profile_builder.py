"""
Module for building user profiles for TV show recommendations.
"""
import logging
import json
import os
import collections
import tqdm
import re
from typing import Dict, List, Any, Optional
from statistics import mean
from app.llm_service import LLMService
from app.utils import load_from_json, save_to_json
from .affinity_scorer import AffinityScorer

logger = logging.getLogger(__name__)

class ProfileBuilder:
    """Class to build a user taste profile from watched shows."""
    
    def __init__(self, openai_api_key=None, watched_shows=None, kmeans=None):
        """
        Initialize the ProfileBuilder.
        
        Args:
            openai_api_key: OpenAI API key. If not provided,
                           will use the OPENAI_API_KEY environment variable.
            watched_shows: List of shows watched by the user. Can be set later.
            kmeans: Optional KMeans model for clustering.
        """
        # Initialize the LLM service
        self.llm_service = LLMService(api_key=openai_api_key)
        
        # Initialize other properties
        self.watched_shows = watched_shows or []
        self.kmeans = kmeans
        
        # Initialize the AffinityScorer
        self.affinity_scorer = AffinityScorer()
    
    def build_profile(self, 
                     shows: List[Dict[str, Any]], 
                     episode_history: Dict[str, List[Dict[str, Any]]] = None,
                     ratings: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a user taste profile from enriched watch history.
        
        Args:
            shows: List of enriched shows with TMDB data
            episode_history: Optional episode-level watch history by show
            ratings: Optional list of show ratings
            
        Returns:
            User taste profile
        """
        logger.info("Building user taste profile")
        
        # First, calculate affinity scores if we have detailed history data
        if episode_history or ratings:
            logger.info("Calculating affinity scores for watched shows")
            shows = self.affinity_scorer.calculate_affinity_scores(
                shows, episode_history, ratings
            )
            
            # Save affinity data to JSON for debugging
            try:
                # Create a simplified version with just the essential affinity data
                affinity_data = [
                    {
                        'title': show.get('title', 'Unknown'),
                        'trakt_id': show.get('trakt_id'),
                        'tmdb_id': show.get('tmdb_id'),
                        'affinity_score': show.get('affinity', {}).get('score', 0),
                        'metrics': show.get('affinity', {}).get('metrics', {}),
                        'flags': show.get('affinity', {}).get('flags', {})
                    }
                    for show in shows if 'affinity' in show
                ]
                
                # Sort by score (descending) for easier analysis
                affinity_data.sort(key=lambda x: x.get('affinity_score', 0), reverse=True)
                
                # Save to file
                save_to_json(affinity_data, 'data/affinity_scores.json')
                    
                logger.info(f"Saved affinity data for {len(affinity_data)} shows to data/affinity_scores.json")
                
                # Log detailed information about completion ratios for diagnosis
                completion_ratios = [(
                    show.get('title', 'Unknown'),
                    show.get('affinity', {}).get('metrics', {}).get('completion_ratio', 0),
                    show.get('affinity', {}).get('metrics', {}).get('watched_episodes', 0),
                    show.get('affinity', {}).get('metrics', {}).get('total_episodes', 0),
                    show.get('affinity', {}).get('flags', {}).get('is_completed', False)
                ) for show in shows if 'affinity' in show]
                
                # Sort by completion ratio
                completion_ratios.sort(key=lambda x: x[1], reverse=True)
                
                logger.info("Top 10 shows by completion ratio:")
                for i, (title, ratio, watched, total, completed) in enumerate(completion_ratios[:10], 1):
                    logger.info(f"  {i}. {title}: {ratio:.2f} - {watched}/{total} episodes - Completed: {completed}")
                
                # Check for shows with inconsistent completion ratios (high ratio but not marked completed)
                inconsistent = [(title, ratio, watched, total) for title, ratio, watched, total, completed in completion_ratios 
                               if ratio >= 0.8 and not completed]
                if inconsistent:
                    logger.warning(f"Found {len(inconsistent)} shows with high completion ratio (≥0.8) but not marked as completed:")
                    for title, ratio, watched, total in inconsistent[:5]:  # Show first 5 examples
                        logger.warning(f"  {title}: {ratio:.2f} - {watched}/{total} episodes")
            except Exception as e:
                logger.error(f"Error saving affinity data to JSON: {e}")
            
            # Log some affinity stats if scores were calculated
            if any('affinity' in show for show in shows):
                favorite_count = len(self.affinity_scorer.get_favorites(shows))
                binged_count = len(self.affinity_scorer.get_binged_shows(shows))
                dropped_count = len(self.affinity_scorer.get_dropped_shows(shows))
                
                logger.info(f"Affinity analysis identified: {favorite_count} favorites, "
                           f"{binged_count} binged shows, {dropped_count} dropped shows")
                
                # Add additional analysis using the potential completionist method
                potential_completionist = len(self.affinity_scorer.get_potential_completionist_shows(shows))
                logger.info(f"Additional analysis found {potential_completionist} potential completionist shows")
                
                # Check scores distribution to see if thresholds might need adjustment
                scores = [show.get('affinity', {}).get('score', 0) for show in shows if 'affinity' in show]
                if scores:
                    scores.sort(reverse=True)
                    logger.info(f"Top affinity scores: {scores[:10]}")
                    logger.info(f"Score distribution - min: {min(scores)}, max: {max(scores)}, "
                               f"avg: {sum(scores)/len(scores):.2f}")
                    
                    # Count shows by score range
                    score_ranges = {
                        "high (≥7)": sum(1 for s in scores if s >= 7),
                        "good (5-6)": sum(1 for s in scores if 5 <= s < 7),
                        "neutral (0-4)": sum(1 for s in scores if 0 <= s < 5),
                        "negative (<0)": sum(1 for s in scores if s < 0)
                    }
                    logger.info(f"Score ranges: {score_ranges}")
        
        # Initialize counters for various attributes
        genre_counter = collections.Counter()
        network_counter = collections.Counter()
        keyword_counter = collections.Counter()
        creator_counter = collections.Counter()
        actor_counter = collections.Counter()
        
        # Initialize lists for top-rated and recently watched shows
        top_rated_shows = []
        recent_shows = []
        favorite_shows = []
        binged_shows = []
        
        # Count shows by decade
        decades = {}
        
        # Highest affinity shows (separate from explicit ratings)
        high_affinity_shows = self.affinity_scorer.get_top_shows(shows, min_score=7, limit=10) if any('affinity' in show for show in shows) else []
        
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
                decades,
                favorite_shows,
                binged_shows
            )
        
        # Sort the recently watched shows by last_watched_at
        recent_shows.sort(key=lambda x: x.get('last_watched_at', ''), reverse=True)
        recent_shows = recent_shows[:10]  # Limit to 10 most recent
        
        # Sort the top-rated shows by vote_average
        top_rated_shows.sort(key=lambda x: x.get('tmdb_data', {}).get('vote_average', 0), reverse=True)
        top_rated_shows = top_rated_shows[:10]  # Limit to 10 top-rated
        
        # Build taste clusters
        taste_clusters = self.build_clusters(shows)
        
        # Build viewing patterns information
        viewing_patterns = self._analyze_viewing_patterns(shows)
        
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
            'favorite_shows': favorite_shows[:10],  # Limit to top 10 favorites
            'binged_shows': binged_shows[:10],     # Limit to top 10 binged shows
            'high_affinity_shows': [
                {
                    'title': show.get('title', 'Unknown'),
                    'tmdb_id': show.get('tmdb_id'),
                    'affinity_score': show.get('affinity', {}).get('score', 0),
                    'watched_episodes': show.get('affinity', {}).get('metrics', {}).get('watched_episodes', 0),
                    'total_episodes': show.get('affinity', {}).get('metrics', {}).get('total_episodes', 0)
                } for show in high_affinity_shows
            ],
            'viewing_patterns': viewing_patterns,
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
            clusters = self._cluster_shows(min_shows_per_cluster=3, max_clusters=5, reclustering_allowed=True)
            
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
            
            # Get affinity data if available
            affinity_data = show.get('affinity', {})
            affinity_score = affinity_data.get('score', 0)
            affinity_flags = affinity_data.get('flags', {})
            
            prepared_show = {
                'title': show.get('title', 'Unknown'),
                'genres': tmdb_data.get('genres', []),
                'keywords': tmdb_data.get('keywords', []),
                'overview': tmdb_data.get('overview', ''),
                'first_air_date': tmdb_data.get('first_air_date', ''),
                'vote_average': tmdb_data.get('vote_average', 0),
                # Add affinity information
                'affinity_score': affinity_score,
                'is_favorite': affinity_flags.get('is_favorite', False),
                'was_binged': affinity_flags.get('was_binged', False),
                'is_completed': affinity_flags.get('is_completed', False),
                'is_dropped': affinity_flags.get('is_dropped', False)
            }
            
            prepared_shows.append(prepared_show)
        
        # Log the total number of shows being used for clustering
        logger.info(f"Using all {len(prepared_shows)} shows for taste clustering")
        
        return prepared_shows
    
    def _cluster_shows(self, min_shows_per_cluster: int = 3, max_clusters: int = 5,
                        reclustering_allowed: bool = True) -> Dict[str, Any]:
        """
        Cluster shows based on their descriptive text.
        
        Args:
            min_shows_per_cluster: Minimum shows per cluster
            max_clusters: Maximum number of clusters
            reclustering_allowed: Whether reclustering is allowed
            
        Returns:
            Dictionary with clustering results
        """
        # Try to get cached clusters first for faster loading
        cached_results = self._get_cached_clusters(min_shows_per_cluster, max_clusters)
        if cached_results:
            return cached_results
        
        # Prepare data for clustering with OpenAI
        watched_data = []
        for show in self.watched_shows:
            watched_data.append({
                "id": show.get("tmdb_id"),
                "title": show.get("name"),
                "overview": show.get("overview", ""),
                "genres": ", ".join(genre["name"] for genre in show.get("genres", [])),
                "first_air_date": show.get("first_air_date", ""),
                "vote_average": show.get("vote_average", 0),
                "watched_ratio": show.get("watched_ratio", 0),
                "rating": show.get("rating", 0)
            })
        
        # Construct the prompt for OpenAI
        prompt = f"""
I need to analyze a user's TV watching history to identify their main taste clusters. Each cluster should represent a distinct preference pattern.

# WATCHED SHOWS DATA
```
{json.dumps(watched_data, indent=2)}
```

Analyze this data and identify {max_clusters} distinct taste clusters (or fewer if appropriate). Each cluster should represent a coherent group of shows that share important characteristics which might explain why the user enjoys them.

Consider these factors when creating clusters:
1. Genre combinations and sub-genres
2. Themes, tones, and narrative styles
3. Character types and relationships
4. Time periods and settings
5. Creative teams (if patterns exist)
6. Viewing patterns (shows with high watch completion vs. partial viewing)
7. User ratings (if available)

For each cluster:
1. Give it a descriptive name that captures its essence (e.g. "Character-Driven Sci-Fi Drama" rather than just "Sci-Fi")
2. List the show IDs that belong to this cluster
3. Provide a detailed explanation of what unifies these shows, focusing on specific elements likely to appeal to the user
4. Describe the viewing experience or emotional satisfaction this cluster provides

Rules:
- Each show MUST be assigned to exactly ONE cluster
- Each cluster MUST contain at least {min_shows_per_cluster} shows
- Create only as many clusters as needed (maximum {max_clusters})
- Strongly favor clusters with at least {min_shows_per_cluster} shows
- Name clusters based on content patterns, not viewing behaviors
- Avoid clusters that solely reflect networks, platforms or release years
- Focus on WHY the user enjoys these shows, not just their surface-level categorization

Your response should be valid JSON in this exact format:
{{
  "clusters": [
    {{
      "name": "Descriptive Cluster Name",
      "show_ids": [123, 456, 789],
      "explanation": "Detailed explanation of what unifies these shows...",
      "viewing_experience": "Description of the viewing experience or emotional satisfaction..."
    }},
    ...more clusters...
  ],
  "profile_summary": "An overall analysis of this viewer's TV watching preferences, viewing patterns, and taste profile based on all the clusters together. Highlight dominant preferences, viewing balance, and any insights about their TV watching identity..."
}}

Only return the JSON, nothing else."""

        # System message for the clustering prompt
        system_message = "You are a TV show analysis expert. Your task is to identify meaningful patterns in a user's TV viewing history. Focus on extracting distinct taste clusters that represent different aspects of their preferences. Each cluster should capture a unique facet of their viewing identity with thoughtful analysis."
        
        try:
            # Use LLM service to generate taste clusters
            clustering_response = self.llm_service.generate_json(
                prompt=prompt,
                system_message=system_message,
                fallback_response={"clusters": [], "profile_summary": "Unable to generate profile"}
            )
            
            # Check if we have valid clusters
            clusters = clustering_response.get('clusters', [])
            if not clusters:
                logger.error("OpenAI did not return any clusters")
                return {"clusters": [], "profile_summary": "Unable to generate profile"}
            
            # Filter out clusters with too few shows
            valid_clusters = [c for c in clusters if len(c.get('show_ids', [])) >= min_shows_per_cluster]
            
            # If we don't have enough valid clusters and reclustering is allowed, try again with fewer minimums
            if len(valid_clusters) < 2 and reclustering_allowed:
                logger.warning(f"Only found {len(valid_clusters)} valid clusters. Retrying with lower minimum...")
                return self._cluster_shows(min_shows_per_cluster=2, max_clusters=max_clusters, reclustering_allowed=False)
            
            # Save clusters to cache for future use
            result = {
                "clusters": valid_clusters,
                "profile_summary": clustering_response.get('profile_summary', "")
            }
            self._cache_clusters(result, min_shows_per_cluster, max_clusters)
            
            return result
            
        except Exception as e:
            logger.error(f"Error clustering shows with OpenAI: {e}")
            return {"clusters": [], "profile_summary": f"Error generating profile: {str(e)}"}
    
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
        genre_counts = collections.Counter()
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
        # Check if we have a very large dataset
        total_shows = len(shows_data)
        logger.info(f"Formatting {total_shows} shows for prompt")
        
        # For very large datasets, we'll provide a summary and representative samples
        if total_shows > 100:
            return self._format_large_dataset_for_prompt(shows_data)
        
        # For smaller datasets, include all shows with details
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
            
            # Add affinity information if available
            if 'affinity_score' in show:
                affinity_score = show.get('affinity_score', 0)
                user_preference = "High Preference" if affinity_score >= 7 else \
                                 "Medium Preference" if affinity_score >= 4 else \
                                 "Low Preference" if affinity_score >= 0 else \
                                 "Disliked"
                
                show_text += f"User Affinity: {user_preference} (score: {affinity_score})\n"
                
                # Add flags if they're true
                affinity_flags = []
                if show.get('is_favorite', False):
                    affinity_flags.append("Favorite")
                if show.get('was_binged', False):
                    affinity_flags.append("Binged")
                if show.get('is_completed', False):
                    affinity_flags.append("Completed")
                if show.get('is_dropped', False):
                    affinity_flags.append("Dropped")
                
                if affinity_flags:
                    show_text += f"Viewing Behavior: {', '.join(affinity_flags)}\n"
                
            formatted_shows.append(show_text)
        
        return "\n\n".join(formatted_shows)
        
    def _format_large_dataset_for_prompt(self, shows_data: List[Dict[str, Any]]) -> str:
        """
        Format a large dataset of shows for the prompt by providing a summary and representative samples.
        
        Args:
            shows_data: Prepared show data (large dataset)
            
        Returns:
            Formatted string with summary and samples
        """
        total_shows = len(shows_data)
        logger.info(f"Creating optimized prompt format for large dataset ({total_shows} shows)")
        
        # Count genres to find most common
        genre_counter = collections.Counter()
        for show in shows_data:
            for genre in show.get('genres', []):
                genre_counter[genre] += 1
                
        # Get top genres
        top_genres = [genre for genre, count in genre_counter.most_common(10)]
        
        # Create groups by genre
        shows_by_genre = {genre: [] for genre in top_genres}
        
        # Group shows by their primary genre (use first listed genre)
        for show in shows_data:
            genres = show.get('genres', [])
            if genres:
                # Find the first genre that's in our top genres
                for genre in genres:
                    if genre in top_genres:
                        shows_by_genre[genre].append(show)
                        break
        
        # Format output with summary and samples
        output_parts = []
        
        # Overall summary
        output_parts.append(f"WATCH HISTORY SUMMARY:\nTotal Shows: {total_shows}\nTop Genres: {', '.join(top_genres)}\n")
        
        # Add affinity summary
        shows_with_affinity = [s for s in shows_data if 'affinity_score' in s]
        if shows_with_affinity:
            # Count shows by affinity category
            high_affinity = sum(1 for s in shows_with_affinity if s.get('affinity_score', 0) >= 7)
            medium_affinity = sum(1 for s in shows_with_affinity if 4 <= s.get('affinity_score', 0) < 7)
            low_affinity = sum(1 for s in shows_with_affinity if 0 <= s.get('affinity_score', 0) < 4)
            disliked = sum(1 for s in shows_with_affinity if s.get('affinity_score', 0) < 0)
            
            # Count behavior flags
            favorites = sum(1 for s in shows_with_affinity if s.get('is_favorite', False))
            binged = sum(1 for s in shows_with_affinity if s.get('was_binged', False))
            completed = sum(1 for s in shows_with_affinity if s.get('is_completed', False))
            dropped = sum(1 for s in shows_with_affinity if s.get('is_dropped', False))
            
            affinity_summary = f"USER PREFERENCE BREAKDOWN:\n"
            affinity_summary += f"High Preference: {high_affinity} shows\n"
            affinity_summary += f"Medium Preference: {medium_affinity} shows\n"
            affinity_summary += f"Low Preference: {low_affinity} shows\n"
            affinity_summary += f"Disliked: {disliked} shows\n\n"
            
            affinity_summary += f"VIEWING BEHAVIOR:\n"
            affinity_summary += f"Favorites: {favorites} shows\n"
            affinity_summary += f"Binged: {binged} shows\n"
            affinity_summary += f"Completed: {completed} shows\n"
            affinity_summary += f"Dropped: {dropped} shows\n"
            
            output_parts.append(affinity_summary)
        
        # Add representative samples from each genre
        output_parts.append("REPRESENTATIVE SAMPLES BY GENRE:")
        
        for genre in top_genres:
            genre_shows = shows_by_genre[genre]
            if not genre_shows:
                continue
                
            output_parts.append(f"\n--- {genre.upper()} SHOWS ({len(genre_shows)} total) ---")
            
            # Sort by affinity score first, then by rating
            if any('affinity_score' in show for show in genre_shows):
                genre_shows.sort(key=lambda x: (x.get('affinity_score', 0), x.get('vote_average', 0)), reverse=True)
            else:
                genre_shows.sort(key=lambda x: x.get('vote_average', 0), reverse=True)
                
            sample_size = min(5, len(genre_shows))
            samples = genre_shows[:sample_size]
            
            # Format each sample
            for i, show in enumerate(samples, 1):
                show_text = f"  {i}. {show.get('title', 'Unknown')}"
                
                # Add year if available
                first_air_date = show.get('first_air_date', '')
                if first_air_date and len(first_air_date) >= 4:
                    show_text += f" ({first_air_date[:4]})"
                    
                # Add rating if available
                vote_average = show.get('vote_average', 0)
                if vote_average:
                    show_text += f" - Rating: {vote_average}"
                    
                # Add affinity score if available
                if 'affinity_score' in show:
                    affinity_score = show.get('affinity_score', 0)
                    show_text += f" - Affinity: {affinity_score}"
                    
                    # Add simple flags
                    flags = []
                    if show.get('is_favorite', False):
                        flags.append("Favorite")
                    if show.get('was_binged', False):
                        flags.append("Binged")
                    if show.get('is_dropped', False):
                        flags.append("Dropped")
                        
                    if flags:
                        show_text += f" ({', '.join(flags)})"
                
                # Add keywords if available
                keywords = show.get('keywords', [])[:3]  # Limit to 3 keywords
                if keywords:
                    show_text += f" - Keywords: {', '.join(keywords)}"
                    
                output_parts.append(show_text)
        
        # Add a full listing of all shows (just titles) for reference
        output_parts.append("\nFULL SHOW LIST:")
        
        # Sort by affinity score for better reference
        if any('affinity_score' in show for show in shows_data):
            sorted_shows = sorted(shows_data, key=lambda x: (x.get('affinity_score', 0), x.get('title', '').lower()), reverse=True)
            
            # For full listing, add affinity categories
            high_affinity_shows = [s for s in sorted_shows if s.get('affinity_score', 0) >= 7]
            medium_affinity_shows = [s for s in sorted_shows if 4 <= s.get('affinity_score', 0) < 7]
            low_affinity_shows = [s for s in sorted_shows if 0 <= s.get('affinity_score', 0) < 4]
            disliked_shows = [s for s in sorted_shows if s.get('affinity_score', 0) < 0]
            
            if high_affinity_shows:
                output_parts.append("\nHIGH PREFERENCE SHOWS:")
                show_titles = [f"  • {show.get('title', 'Unknown')} ({show.get('affinity_score', 0)})" 
                              for show in high_affinity_shows]
                output_parts.append("\n".join(show_titles))
                
            if medium_affinity_shows:
                output_parts.append("\nMEDIUM PREFERENCE SHOWS:")
                show_titles = [f"  • {show.get('title', 'Unknown')} ({show.get('affinity_score', 0)})" 
                              for show in medium_affinity_shows]
                output_parts.append("\n".join(show_titles))
                
            if low_affinity_shows:
                output_parts.append("\nLOW PREFERENCE SHOWS:")
                show_titles = [f"  • {show.get('title', 'Unknown')} ({show.get('affinity_score', 0)})" 
                              for show in low_affinity_shows]
                output_parts.append("\n".join(show_titles))
                
            if disliked_shows:
                output_parts.append("\nDISLIKED SHOWS:")
                show_titles = [f"  • {show.get('title', 'Unknown')} ({show.get('affinity_score', 0)})" 
                              for show in disliked_shows]
                output_parts.append("\n".join(show_titles))
        else:
            # If no affinity scores, just sort alphabetically
            sorted_shows = sorted(shows_data, key=lambda x: x.get('title', '').lower())
            show_titles = [f"  • {show.get('title', 'Unknown')}" for show in sorted_shows]
            output_parts.append("\n".join(show_titles))
        
        return "\n".join(output_parts)
    
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
            # Check if cache exists
            if not os.path.exists(cache_file):
                return None
                
            # Calculate a simple fingerprint of the watch history to determine if cache is still valid
            # Just use the number of shows and a hash of their titles
            show_titles = sorted([show.get('title', '') for show in shows])
            current_fingerprint = f"{len(shows)}_{hash(tuple(show_titles))}"
            
            # Load cache data using utils function
            cache_data = load_from_json(cache_file)
            if not cache_data:
                return None
                
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
            
            # Use utils.save_to_json instead of direct JSON operations
            save_to_json(cache_data, cache_file)
                
            logger.info(f"Cached {len(clusters)} taste clusters")
            
        except Exception as e:
            logger.error(f"Error caching taste clusters: {e}")
    
    def _process_show_for_profile(self, 
        show: Dict[str, Any], 
        genre_counter: collections.Counter, 
        network_counter: collections.Counter, 
        keyword_counter: collections.Counter, 
        creator_counter: collections.Counter, 
        actor_counter: collections.Counter,
        top_rated_shows: List,
        recent_shows: List,
        decades: Dict,
        favorite_shows: List,
        binged_shows: List
    ) -> None:
        """
        Process a single show to extract profile data.
        
        Args:
            show: Show data
            genre_counter: Counter for genres
            network_counter: Counter for networks
            keyword_counter: Counter for keywords
            creator_counter: Counter for creators
            actor_counter: Counter for actors
            top_rated_shows: List of top-rated shows
            recent_shows: List of recently watched shows
            decades: Dictionary of decades
            favorite_shows: List of favorite shows
            binged_shows: List of shows that were binged
        """
        # Extract TMDB data
        tmdb_data = show.get('tmdb_data', {})
        
        # Count genres
        genres = tmdb_data.get('genres', [])
        for genre in genres:
            genre_counter[genre] += 1
        
        # Count networks
        networks = tmdb_data.get('networks', [])
        for network in networks:
            network_counter[network] += 1
        
        # Count keywords/themes
        keywords = tmdb_data.get('keywords', [])
        for keyword in keywords:
            keyword_counter[keyword] += 1
        
        # Count creators
        creators = tmdb_data.get('creators', [])
        if creators:
            for creator in creators:
                if isinstance(creator, dict) and 'name' in creator:
                    creator_counter[creator['name']] += 1
                else:
                    creator_counter[str(creator)] += 1
        
        # Count actors
        cast = tmdb_data.get('cast', [])
        for actor in cast:
            if isinstance(actor, dict) and 'name' in actor:
                actor_counter[actor['name']] += 1
        
        # Check if it's a top-rated show
        if tmdb_data.get('vote_average', 0) >= 7.5 and tmdb_data.get('vote_count', 0) >= 1000:
            top_rated_shows.append(show)
        
        # Check if it's a recent show
        if show.get('last_watched_at'):
            recent_shows.append(show)
        
        # Count shows by decade
        first_air_date = tmdb_data.get('first_air_date', '')
        if first_air_date and len(first_air_date) >= 4:
            year = int(first_air_date[:4])
            decade = f"{year // 10 * 10}"
            decades[decade] = decades.get(decade, 0) + 1
        
        # Check for affinity data
        if 'affinity' in show:
            affinity = show['affinity']
            
            # Add to favorites if marked as favorite
            if affinity.get('flags', {}).get('is_favorite', False):
                favorite_shows.append({
                    'title': show.get('title', 'Unknown'),
                    'tmdb_id': show.get('tmdb_id'),
                    'affinity_score': affinity.get('score', 0),
                    'genres': tmdb_data.get('genres', [])
                })
            
            # Add to binged shows if it was binged
            if affinity.get('flags', {}).get('was_binged', False):
                binged_shows.append({
                    'title': show.get('title', 'Unknown'),
                    'tmdb_id': show.get('tmdb_id'),
                    'time_span_days': affinity.get('metrics', {}).get('watch_time_span_days', None),
                    'episodes': affinity.get('metrics', {}).get('watched_episodes', 0)
                })
    
    def _analyze_viewing_patterns(self, shows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze viewing patterns to understand user preferences better.
        
        Args:
            shows: List of watched shows with metadata
            
        Returns:
            Dictionary with viewing pattern insights
        """
        viewing_patterns = {
            'completion_rate': 0.0,
            'binge_watching': 0.0,
            'rewatch_rate': 0.0,
            'favorite_genres': [],
            'dropped_genres': [],
            'viewing_mode': 'unknown',
            'format_preferences': []
        }
        
        shows_with_affinity = [show for show in shows if 'affinity' in show]
        
        if not shows_with_affinity:
            return viewing_patterns
            
        # Calculate completion rate
        completed_shows = len([s for s in shows_with_affinity 
                            if s['affinity'].get('flags', {}).get('is_completed', False)])
        viewing_patterns['completion_rate'] = completed_shows / len(shows_with_affinity) if shows_with_affinity else 0
        
        # Calculate binge watching percentage
        binged_shows = len([s for s in shows_with_affinity 
                          if s['affinity'].get('flags', {}).get('was_binged', False)])
        viewing_patterns['binge_watching'] = binged_shows / len(shows_with_affinity) if shows_with_affinity else 0
        
        # Calculate rewatch rate
        rewatched_shows = len([s for s in shows_with_affinity 
                             if s['affinity'].get('flags', {}).get('is_rewatch', False)])
        viewing_patterns['rewatch_rate'] = rewatched_shows / len(shows_with_affinity) if shows_with_affinity else 0
        
        # Identify favorite genres (genres of favorite shows)
        favorite_genres = collections.Counter()
        for show in shows_with_affinity:
            if show['affinity'].get('flags', {}).get('is_favorite', False):
                for genre in show.get('tmdb_data', {}).get('genres', []):
                    favorite_genres[genre] += 1
        
        viewing_patterns['favorite_genres'] = [g for g, c in favorite_genres.most_common(5)]
        
        # Identify dropped genres (genres of dropped shows)
        dropped_genres = collections.Counter()
        for show in shows_with_affinity:
            if show['affinity'].get('flags', {}).get('is_dropped', False):
                for genre in show.get('tmdb_data', {}).get('genres', []):
                    dropped_genres[genre] += 1
        
        viewing_patterns['dropped_genres'] = [g for g, c in dropped_genres.most_common(5)]
        
        # Determine preferred viewing mode
        if viewing_patterns['binge_watching'] > 0.6:
            viewing_patterns['viewing_mode'] = 'binge_watcher'
        elif viewing_patterns['completion_rate'] > 0.8:
            viewing_patterns['viewing_mode'] = 'completionist'
        elif viewing_patterns['rewatch_rate'] > 0.2:
            viewing_patterns['viewing_mode'] = 'rewatcher'
        else:
            viewing_patterns['viewing_mode'] = 'casual'
        
        # Analyze format preferences (episode counts, etc.)
        episode_counts = [show.get('affinity', {}).get('metrics', {}).get('total_episodes', 0) 
                         for show in shows_with_affinity]
        episode_counts = [count for count in episode_counts if count > 0]
        
        if episode_counts:
            avg_episode_count = sum(episode_counts) / len(episode_counts)
            
            if avg_episode_count < 15:
                viewing_patterns['format_preferences'].append('mini_series')
            elif avg_episode_count > 50:
                viewing_patterns['format_preferences'].append('long_running_series')
                
        # Check for anthology preference
        anthology_keywords = ['anthology', 'episodic', 'standalone']
        anthology_count = sum(1 for show in shows_with_affinity
                             if any(kw in show.get('tmdb_data', {}).get('keywords', []) 
                                   for kw in anthology_keywords))
        
        if anthology_count / len(shows_with_affinity) > 0.2 if shows_with_affinity else False:
            viewing_patterns['format_preferences'].append('anthology')
        
        return viewing_patterns 