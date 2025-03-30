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
    
    def get_watched_shows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch watched shows for the user.
        
        Args:
            limit: Optional limit on the number of shows to return
            
        Returns:
            List of watched shows with their metadata
        """
        endpoint = f"/users/{self.username}/watched/shows"
        url = f"{self.BASE_URL}{endpoint}"
        
        logger.info(f"Fetching watched shows for user {self.username}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            watched_shows = response.json()
            
            # Apply limit if specified
            if limit and limit > 0:
                watched_shows = watched_shows[:limit]
            
            logger.info(f"Successfully fetched {len(watched_shows)} watched shows")
            
            # Process and extract relevant show information
            processed_shows = self._process_shows(watched_shows)
            
            return processed_shows
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching watched shows: {e}")
            raise
    
    def get_show_details(self, show_id: str) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific show.
        
        Args:
            show_id: Trakt ID of the show
            
        Returns:
            Detailed show information
        """
        endpoint = f"/shows/{show_id}?extended=full"
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching show details for ID {show_id}: {e}")
            raise
    
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
    
    def get_ratings(self) -> List[Dict[str, Any]]:
        """
        Fetch user ratings for shows.
        
        Returns:
            List of show ratings
        """
        endpoint = f"/users/{self.username}/ratings/shows"
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching show ratings: {e}")
            raise 