"""
Prompt Engineering Module - Provides a centralized location for all prompt templates and formatting.

This module contains utilities for creating and formatting prompts for LLM interactions
across different features of the application. It centralizes all prompt templates to:
- Maintain consistent prompt quality and style
- Make prompt modifications easier to implement across the application
- Separate prompt engineering concerns from business logic
- Provide a clear API for generating different types of prompts

The module handles formatting for:
- TV show recommendations based on user profiles
- Show clustering for profile building
- Profile summaries for various LLM interactions
- Candidate show data formatting for recommendation prompts
"""

import logging
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PromptEngineering:
    """
    Class for creating and formatting prompts for LLM interactions.
    
    This class serves as a centralized repository for all prompt templates used in the
    application. It provides static methods for formatting different types of prompts,
    ensuring consistency across LLM interactions and making it easier to update prompt
    templates when necessary.
    
    Each formatting method returns a dictionary containing:
    - The formatted prompt text
    - A system message that guides the LLM's behavior
    
    Usage example:
    ```python
    prompt_engineering = PromptEngineering()
    recommendation_prompt = prompt_engineering.format_recommendation_prompt(
        profile_summary="User likes sci-fi and drama...",
        candidates_data="Show 1: ...\nShow 2: ...",
        num_recommendations=3
    )
    
    # Use with LLMService
    llm_service = LLMService()
    response = llm_service.generate_json(
        prompt=recommendation_prompt["prompt"],
        system_message=recommendation_prompt["system_message"]
    )
    ```
    """
    
    @staticmethod
    def format_recommendation_prompt(profile_summary: str, candidates_data: str, num_recommendations: int) -> Dict[str, Any]:
        """
        Format a prompt for generating TV show recommendations.
        
        Creates a detailed prompt that instructs the LLM to recommend TV shows based on
        a user's taste profile. The prompt includes specific instructions to ensure
        balanced, diverse recommendations that match the user's preferences.
        
        Args:
            profile_summary: A formatted summary of the user's taste profile
            candidates_data: Formatted information about candidate shows
            num_recommendations: The number of recommendations to generate
            
        Returns:
            Dictionary with:
              - prompt: The formatted prompt text
              - system_message: A system message to guide the LLM's behavior
              
        Note:
            The prompt instructs the LLM to return valid JSON in a specific format.
        """
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
        
        system_message = "You are a TV show recommendation expert. Focus on providing balanced, unbiased recommendations by considering the user's entire taste profile holistically. Avoid common anchoring biases like overweighting recent shows, most frequent genres, or most prominent keywords. Pay special attention to the user's diverse taste clusters and consider them with equal importance. Aim to provide recommendations that both match existing preferences and thoughtfully expand their horizons with carefully selected new experiences."
        
        return {
            "prompt": prompt,
            "system_message": system_message
        }
    
    @staticmethod
    def format_clustering_prompt(watched_data: List[Dict[str, Any]], min_shows_per_cluster: int = 3, max_clusters: int = 5) -> Dict[str, Any]:
        """
        Format a prompt for clustering shows based on user preferences.
        
        Creates a detailed prompt that instructs the LLM to analyze a user's watch history
        and identify distinct taste clusters. These clusters help understand the user's
        preferences and are used to build their taste profile.
        
        Args:
            watched_data: List of dictionaries containing information about watched shows
            min_shows_per_cluster: The minimum number of shows that should be in each cluster
            max_clusters: The maximum number of clusters to identify
            
        Returns:
            Dictionary with:
              - prompt: The formatted prompt text
              - system_message: A system message to guide the LLM's behavior
              
        Note:
            The prompt includes specific rules for clustering and instructs the LLM
            to return valid JSON in a specific format.
        """
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

Only return the JSON, nothing else.
"""
        
        system_message = "You are a TV show analysis expert. Your task is to identify meaningful patterns in a user's TV viewing history. Focus on extracting distinct taste clusters that represent different aspects of their preferences. Each cluster should capture a unique facet of their viewing identity with thoughtful analysis."
        
        return {
            "prompt": prompt,
            "system_message": system_message
        }
    
    @staticmethod
    def format_profile_summary(profile: Dict[str, Any]) -> str:
        """
        Format a user profile summary for LLM prompts.
        
        Creates a human-readable summary of a user's taste profile, which can be
        included in prompts for recommendations or other LLM interactions. The summary
        includes information about favorite genres, decades, themes, creators, actors,
        viewing patterns, and favorite shows.
        
        Args:
            profile: A dictionary containing the user's taste profile
            
        Returns:
            A formatted string summarizing the user's taste profile
            
        Note:
            This method is typically used as input to other prompt formatting methods,
            such as format_recommendation_prompt.
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
                # Handle case where cluster might be a string instead of a dictionary
                if isinstance(cluster, dict):
                    cluster_summary = f"  Cluster {i}: {cluster.get('name', 'Unnamed Cluster')}"
                    
                    # Try to get description from either description OR explanation field
                    description = cluster.get('description') or cluster.get('explanation') or "No description"
                    cluster_summary += f"\n    Description: {description}"
                    
                    if cluster.get('viewing_experience'):
                        cluster_summary += f"\n    Viewing Experience: {cluster.get('viewing_experience')}"
                    
                    if cluster.get('genres'):
                        cluster_summary += f"\n    Genres: {', '.join(cluster['genres'][:5])}"
                        
                    if cluster.get('keywords'):
                        cluster_summary += f"\n    Keywords: {', '.join(cluster['keywords'][:5])}"
                    
                    # Show example_shows if available, otherwise show_ids
                    if cluster.get('example_shows'):
                        cluster_summary += f"\n    Example Shows: {', '.join(cluster['example_shows'][:3])}"
                    elif cluster.get('show_ids'):
                        cluster_summary += f"\n    Show IDs: {', '.join(str(id) for id in cluster['show_ids'][:5])}"
                else:
                    # If cluster is a string or other non-dict type, handle it gracefully
                    cluster_summary = f"  Cluster {i}: {str(cluster)}"
                    logger.warning(f"Unexpected cluster format in taste_clusters (item {i}): Expected dict, got {type(cluster)}")
                
                summary.append(cluster_summary)
                
        return "\n".join(summary)
    
    @staticmethod
    def format_candidates_data(candidates: List[Dict[str, Any]]) -> str:
        """
        Format candidate show data for recommendation prompts.
        
        Creates a formatted representation of candidate shows that can be included
        in a recommendation prompt. The formatted data includes information about
        each show's title, ID, genres, overview, and other relevant details.
        
        Args:
            candidates: List of dictionaries containing information about candidate shows
            
        Returns:
            A formatted string representing the candidate shows
            
        Note:
            This method is typically used as input to format_recommendation_prompt.
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