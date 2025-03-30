"""
Module for fetching candidate shows for recommendations.
"""

import logging
from typing import List, Dict, Any, Set

from app.tmdb_enricher import TMDBEnricher

logger = logging.getLogger(__name__)

class CandidateFetcher:
    """Class to fetch candidate shows for recommendations."""
    
    def __init__(self, tmdb_enricher: TMDBEnricher):
        """
        Initialize the CandidateFetcher.
        
        Args:
            tmdb_enricher: TMDBEnricher instance for fetching TMDB data
        """
        self.tmdb_enricher = tmdb_enricher
    
    def get_candidates(self, watched_shows: List[Dict[str, Any]], max_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Get candidate shows for recommendations.
        
        Args:
            watched_shows: List of shows the user has already watched
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate shows for recommendations
        """
        logger.info("Fetching candidate shows for recommendations")
        
        # Get watched show IDs to avoid recommending shows the user has already seen
        watched_tmdb_ids = self._get_watched_tmdb_ids(watched_shows)
        
        # Get trending shows
        trending_shows = self._get_trending_shows(watched_tmdb_ids)
        logger.info(f"Fetched {len(trending_shows)} trending shows")
        
        # Get top-rated shows
        discover_shows = self._get_discover_shows(watched_tmdb_ids)
        logger.info(f"Fetched {len(discover_shows)} discover shows")
        
        # Get similar shows based on the user's top-rated watched shows
        similar_shows = self._get_similar_shows(watched_shows, watched_tmdb_ids)
        logger.info(f"Fetched {len(similar_shows)} similar shows")
        
        # Combine candidates, prioritizing trending shows
        all_candidates = trending_shows + similar_shows + discover_shows
        
        # Remove duplicates while preserving order
        unique_candidates = self._remove_duplicate_candidates(all_candidates)
        
        # Limit the number of candidates
        candidates = unique_candidates[:max_candidates]
        
        logger.info(f"Selected {len(candidates)} candidate shows for recommendations")
        
        return candidates
    
    def _get_watched_tmdb_ids(self, watched_shows: List[Dict[str, Any]]) -> Set[int]:
        """
        Get the TMDB IDs of shows the user has already watched.
        
        Args:
            watched_shows: List of watched shows
            
        Returns:
            Set of TMDB IDs
        """
        tmdb_ids = set()
        
        for show in watched_shows:
            tmdb_id = show.get('tmdb_id')
            if tmdb_id:
                tmdb_ids.add(tmdb_id)
        
        return tmdb_ids
    
    def _get_trending_shows(self, watched_tmdb_ids: Set[int], limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get trending shows that the user hasn't watched.
        
        Args:
            watched_tmdb_ids: Set of TMDB IDs the user has already watched
            limit: Maximum number of trending shows to fetch
            
        Returns:
            List of trending shows
        """
        # Get trending shows for the week
        trending_week = self.tmdb_enricher.get_trending_shows(time_window='week', page=1)
        
        # Get trending shows for the day
        trending_day = self.tmdb_enricher.get_trending_shows(time_window='day', page=1)
        
        # Combine trending shows, prioritizing day trends
        all_trending = trending_day + trending_week
        
        # Remove shows the user has already watched
        filtered_trending = [
            show for show in all_trending 
            if show.get('tmdb_id') and show.get('tmdb_id') not in watched_tmdb_ids
        ]
        
        # Remove duplicates
        unique_trending = self._remove_duplicate_candidates(filtered_trending)
        
        # Return limited number of trending shows
        return unique_trending[:limit]
    
    def _get_discover_shows(self, watched_tmdb_ids: Set[int], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top-rated shows that the user hasn't watched.
        
        Args:
            watched_tmdb_ids: Set of TMDB IDs the user has already watched
            limit: Maximum number of top-rated shows to fetch
            
        Returns:
            List of top-rated shows
        """
        # For now, we'll use a simple search for top-rated shows
        # In a future version, this could be expanded to use TMDB's discover API more extensively
        discover_shows = []
        
        # Get some top-rated shows
        top_rated_shows = self.tmdb_enricher.search_shows('top rated tv shows')
        discover_shows.extend(top_rated_shows)
        
        # Get some recent shows
        recent_shows = self.tmdb_enricher.search_shows('new tv shows 2023')
        discover_shows.extend(recent_shows)
        
        # Remove shows the user has already watched
        filtered_discover = [
            show for show in discover_shows 
            if show.get('tmdb_id') and show.get('tmdb_id') not in watched_tmdb_ids
        ]
        
        # Remove duplicates
        unique_discover = self._remove_duplicate_candidates(filtered_discover)
        
        # Return limited number of discover shows
        return unique_discover[:limit]
    
    def _get_similar_shows(self, watched_shows: List[Dict[str, Any]], watched_tmdb_ids: Set[int], limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get shows similar to the user's top-rated watched shows.
        
        Args:
            watched_shows: List of watched shows
            watched_tmdb_ids: Set of TMDB IDs the user has already watched
            limit: Maximum number of similar shows to fetch
            
        Returns:
            List of similar shows
        """
        similar_shows = []
        
        # Get the user's top 5 watched shows (based on TMDB rating)
        top_watched = sorted(
            [show for show in watched_shows if show.get('tmdb_data', {}).get('vote_average')],
            key=lambda x: x.get('tmdb_data', {}).get('vote_average', 0),
            reverse=True
        )[:5]
        
        # For each top watched show, get similar shows
        for show in top_watched:
            tmdb_data = show.get('tmdb_data', {})
            if 'similar_shows' in tmdb_data:
                # Extract similar shows from the enriched data
                similar = tmdb_data['similar_shows']
                for s in similar:
                    # Format similar shows to match our standard format
                    formatted_show = {
                        'tmdb_id': s.get('id'),
                        'title': s.get('name'),
                        'overview': s.get('overview'),
                        'first_air_date': s.get('first_air_date'),
                        'poster_path': s.get('poster_path'),
                    }
                    
                    # Add poster URL if available
                    if formatted_show.get('poster_path'):
                        formatted_show['poster_url'] = f"https://image.tmdb.org/t/p/w500{formatted_show['poster_path']}"
                    
                    similar_shows.append(formatted_show)
        
        # Remove shows the user has already watched
        filtered_similar = [
            show for show in similar_shows 
            if show.get('tmdb_id') and show.get('tmdb_id') not in watched_tmdb_ids
        ]
        
        # Remove duplicates
        unique_similar = self._remove_duplicate_candidates(filtered_similar)
        
        # Return limited number of similar shows
        return unique_similar[:limit]
    
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
            tmdb_id = candidate.get('tmdb_id')
            if tmdb_id and tmdb_id not in seen_ids:
                seen_ids.add(tmdb_id)
                unique_candidates.append(candidate)
        
        return unique_candidates 