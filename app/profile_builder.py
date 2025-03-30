"""
Module for building a user taste profile from watch history.
"""

import logging
from typing import List, Dict, Any, Counter
from collections import Counter

logger = logging.getLogger(__name__)

class ProfileBuilder:
    """Class to build a user taste profile from watched shows."""
    
    def __init__(self):
        """Initialize the ProfileBuilder."""
        pass
    
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
            'top_rated_shows': top_rated_shows
        }
        
        logger.info("User taste profile built successfully")
        
        return profile
    
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