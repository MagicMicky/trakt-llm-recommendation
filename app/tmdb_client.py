"""
Client for interacting with The Movie Database (TMDB) API.
"""

import os
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)

class TMDBClient:
    """Client for interacting with The Movie Database (TMDB) API."""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the TMDB client.
        
        Args:
            api_key: TMDB API key. If not provided, will use TMDB_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            logger.warning("No TMDB API key provided or found in environment variables")
    
    def discover_shows_by_genres(self, genres: List[str], watched_ids: Set[int], limit: int = 30) -> List[Dict[str, Any]]:
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
         
        # Get shows for each genre (up to 3 genres to avoid too many API calls)
        for genre_id in genre_ids[:3]:
            params = {
                "with_genres": genre_id,
                "sort_by": "popularity.desc",
                "page": 1,
                "language": "en-US"
            }
            
            # Make the API request
            genre_shows = self._make_request(
                endpoint="/discover/tv",
                params=params,
                watched_ids=watched_ids,
                limit=limit - len(discovered_shows)
            )
            
            discovered_shows.extend(genre_shows)
            
            # Add delay to avoid rate limiting
            time.sleep(0.25)
            
            # Break if we have enough shows
            if len(discovered_shows) >= limit:
                break
        
        return discovered_shows[:limit]
    
    def discover_shows_with_params(self, params: Dict[str, Any], watched_ids: Set[int], limit: int = 20) -> List[Dict[str, Any]]:
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
        
        # Prepare API parameters
        api_params = {
            "sort_by": "popularity.desc",
            "page": 1,
            "language": "en-US"
        }
        
        # Add custom params
        for key, value in params.items():
            api_params[key] = value
        
        # Make the API request
        return self._make_request(
            endpoint="/discover/tv",
            params=api_params,
            watched_ids=watched_ids,
            limit=limit
        )
    
    def get_similar_shows(self, tmdb_id: int, watched_ids: Set[int], limit: int = 10) -> List[Dict[str, Any]]:
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
        
        # Prepare API parameters
        params = {
            "language": "en-US",
            "page": 1
        }
        
        # Make the API request
        return self._make_request(
            endpoint=f"/tv/{tmdb_id}/similar",
            params=params,
            watched_ids=watched_ids,
            limit=limit
        )
    
    def get_trending_shows(self, watched_ids: Set[int], limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get trending TV shows.
        
        Args:
            watched_ids: Set of watched show IDs to exclude
            limit: Maximum number of recommendations to return
            
        Returns:
            List of trending shows
        """
        logger.info(f"Getting trending TV shows (limit: {limit})")
        
        results = []
        page = 1
        
        # TMDB typically returns 20 results per page, so calculate how many pages we need
        # to reach our limit (considering we might need to exclude some shows)
        max_pages = (limit // 15) + 2  # Add buffer for watched shows filtering
        
        while len(results) < limit and page <= max_pages:
            # Prepare API parameters
            params = {
                "language": "en-US",
                "page": page
            }
            
            # Make the API request for this page
            page_results = self._make_request_single_page(
                endpoint="/trending/tv/week",
                params=params,
                watched_ids=watched_ids
            )
            
            # Break if no more results
            if not page_results:
                break
                
            results.extend(page_results)
            
            # Increment page number
            page += 1
            
            # Add delay to avoid rate limiting
            time.sleep(0.25)
        
        # Limit to requested number
        return results[:limit]
    
    def _make_request_single_page(self, endpoint: str, params: Dict[str, Any], watched_ids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
        """
        Make a request for a single page to the TMDB API and process results.
        
        Args:
            endpoint: TMDB API endpoint to call
            params: Query parameters for the API request
            watched_ids: Set of show IDs to exclude from results
            
        Returns:
            List of processed show data for a single page
        """
        results = []
        
        try:
            # Ensure API key is in parameters
            if "api_key" not in params:
                params["api_key"] = self.api_key
            
            # Build the full URL
            url = f"{self.BASE_URL}{endpoint}"
            
            # Make the API request
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                show_results = data.get("results", [])
                
                for show in show_results:
                    # Skip shows the user has already watched if watched_ids provided
                    if watched_ids is not None and show.get("id") in watched_ids:
                        continue
                        
                    # Process the show data
                    processed_show = self._process_show(show)
                    results.append(processed_show)
            else:
                logger.error(f"TMDB API request failed with status code {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error making TMDB API request to {endpoint}: {e}")
        
        return results
    
    def _make_request(self, endpoint: str, params: Dict[str, Any], watched_ids: Optional[Set[int]] = None, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Make a request to the TMDB API and process results.
        
        Args:
            endpoint: TMDB API endpoint to call
            params: Query parameters for the API request
            watched_ids: Set of show IDs to exclude from results
            limit: Maximum number of results to return (0 for no limit)
            
        Returns:
            List of processed show data
        """
        results = []
        
        try:
            # Ensure API key is in parameters
            if "api_key" not in params:
                params["api_key"] = self.api_key
            
            # Build the full URL
            url = f"{self.BASE_URL}{endpoint}"
            
            # Make the API request
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                show_results = data.get("results", [])
                
                for show in show_results:
                    # Skip shows the user has already watched if watched_ids provided
                    if watched_ids is not None and show.get("id") in watched_ids:
                        continue
                        
                    # Process the show data
                    processed_show = self._process_show(show)
                    results.append(processed_show)
                    
                    # Break if we have reached the limit
                    if limit > 0 and len(results) >= limit:
                        break
            else:
                logger.error(f"TMDB API request failed with status code {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error making TMDB API request to {endpoint}: {e}")
        
        # Apply limit if specified
        if limit > 0:
            return results[:limit]
        
        return results
    
    def _process_show(self, show: Dict[str, Any]) -> Dict[str, Any]:
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