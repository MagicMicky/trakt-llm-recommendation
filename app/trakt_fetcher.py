"""
Module for fetching watch history from Trakt API.
"""

import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TraktFetcher:
    """Class to interact with the Trakt API and fetch a user's watch history."""
    
    BASE_URL = "https://api.trakt.tv"
    
    def __init__(self, client_id: str, client_secret: str, access_token: str, username: str):
        """
        Initialize the TraktFetcher.
        
        Args:
            client_id: Trakt API client ID
            client_secret: Trakt API client secret
            access_token: Trakt API access token
            username: Trakt username
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.username = username
        self.headers = {
            'Content-Type': 'application/json',
            'trakt-api-version': '2',
            'trakt-api-key': client_id,
            'Authorization': f'Bearer {access_token}'
        }
    
    def _make_trakt_request(self, endpoint: str, error_message: str = "Error in Trakt API request", empty_result: Any = None) -> Any:
        """
        Make a request to the Trakt API with proper error handling.
        
        Args:
            endpoint: API endpoint to call (should start with /)
            error_message: Message to log in case of error
            empty_result: Value to return in case of error
            
        Returns:
            Response data or empty_result in case of error
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"{error_message}: {e}")
            return empty_result
    
    def get_watched_shows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch watched shows for the user.
        
        Args:
            limit: Optional limit on the number of shows to return
            
        Returns:
            List of watched shows with their metadata
        """
        endpoint = f"/users/{self.username}/watched/shows"
        
        logger.info(f"Fetching watched shows for user {self.username}")
        
        watched_shows = self._make_trakt_request(
            endpoint=endpoint,
            error_message=f"Error fetching watched shows for user {self.username}",
            empty_result=[]
        )
        
        if not watched_shows:
            logger.error("Failed to fetch watched shows")
            return []
            
        # Apply limit if specified
        if limit and limit > 0:
            watched_shows = watched_shows[:limit]
        
        logger.info(f"Successfully fetched {len(watched_shows)} watched shows")
        
        # Process and extract relevant show information
        processed_shows = self._process_shows(watched_shows)
        
        return processed_shows
    
    def get_episode_watch_history(self, show_id: str) -> List[Dict[str, Any]]:
        """
        Fetch episode-level watch history for a specific show.
        
        Args:
            show_id: Trakt ID of the show
            
        Returns:
            List of episode watch records with timestamps
        """
        endpoint = f"/users/{self.username}/history/shows/{show_id}"
        
        episode_history = self._make_trakt_request(
            endpoint=endpoint,
            error_message=f"Error fetching episode history for show ID {show_id}",
            empty_result=[]
        )
        
        if episode_history:
            logger.debug(f"Fetched {len(episode_history)} episode watch records for show ID {show_id}")
            
        return episode_history
    
    def get_all_episode_history(self, show_ids: List[str], max_shows: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch episode watch history for multiple shows.
        
        Args:
            show_ids: List of Trakt show IDs
            max_shows: Maximum number of shows to fetch history for (0 means no limit)
            
        Returns:
            Dictionary mapping show IDs to their episode watch histories
        """
        logger.info(f"Fetching episode watch history for {len(show_ids)} shows (max: {max_shows if max_shows > 0 else 'no limit'})")
        
        # Apply limit only if max_shows is greater than 0
        limited_show_ids = show_ids
        if max_shows > 0:
            limited_show_ids = show_ids[:max_shows]
        
        all_history = {}
        
        for show_id in limited_show_ids:
            try:
                episode_history = self.get_episode_watch_history(show_id)
                if episode_history:
                    all_history[show_id] = episode_history
            except Exception as e:
                logger.error(f"Error fetching episode history for show ID {show_id}: {e}")
                continue
        
        logger.info(f"Successfully fetched episode history for {len(all_history)} shows")
        return all_history
    
    def get_ratings(self) -> List[Dict[str, Any]]:
        """
        Fetch user ratings for shows.
        
        Returns:
            List of show ratings
        """
        endpoint = f"/users/{self.username}/ratings/shows"
        
        ratings = self._make_trakt_request(
            endpoint=endpoint,
            error_message=f"Error fetching show ratings for user {self.username}",
            empty_result=[]
        )
        
        if ratings:
            logger.info(f"Successfully fetched {len(ratings)} show ratings")
        
        return ratings
    
    def _process_shows(self, watched_shows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the watched shows data to extract relevant information.
        
        Args:
            watched_shows: Raw show data from Trakt API
            
        Returns:
            Processed show data with relevant fields
        """
        processed_shows = []
        
        for show in watched_shows:
            try:
                processed_show = {
                    'trakt_id': show['show']['ids']['trakt'],
                    'tmdb_id': show['show']['ids'].get('tmdb'),
                    'imdb_id': show['show']['ids'].get('imdb'),
                    'title': show['show']['title'],
                    'year': show['show'].get('year'),
                    'overview': show.get('overview', ''),
                    'plays': show.get('plays', 0),  # Number of plays/watches
                    'watched_episodes': show.get('watched_episodes', 0),
                    'last_watched_at': show.get('last_watched_at'),
                }
                
                processed_shows.append(processed_show)
                
            except KeyError as e:
                logger.warning(f"Missing key when processing show: {e}")
        
        return processed_shows 