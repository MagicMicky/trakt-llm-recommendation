"""
Module for fetching candidate shows for recommendations.
"""

import logging
from typing import List, Dict, Any, Set
import datetime
import tmdbsimple as tmdb

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
        
        # Initialize empty lists for each source
        trending_shows = []
        discover_shows = []
        similar_shows = []
        
        # Get trending shows with error handling
        try:
            trending_shows = self._get_trending_shows(watched_tmdb_ids)
            logger.info(f"Fetched {len(trending_shows)} trending shows")
        except Exception as e:
            logger.error(f"Error fetching trending shows: {e}")
        
        # Get top-rated shows with error handling
        try:
            discover_shows = self._get_discover_shows(watched_tmdb_ids)
            logger.info(f"Fetched {len(discover_shows)} discover shows")
        except Exception as e:
            logger.error(f"Error fetching discover shows: {e}")
        
        # Get similar shows with error handling
        try:
            similar_shows = self._get_similar_shows(watched_shows, watched_tmdb_ids)
            logger.info(f"Fetched {len(similar_shows)} similar shows")
        except Exception as e:
            logger.error(f"Error fetching similar shows: {e}")
        
        # Combine candidates, prioritizing trending shows
        all_candidates = trending_shows + similar_shows + discover_shows
        
        # If we don't have any candidates, just return an empty list
        if not all_candidates:
            logger.warning("No candidates found from API. Returning empty list.")
            return []
        
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
        try:
            # Get trending shows for the week
            trending_week = self.tmdb_enricher.get_trending_shows(time_window='week', page=1)
            
            # Get trending shows for the day (or use a second page of weekly trends as fallback)
            if trending_week:
                trending_day = self.tmdb_enricher.get_trending_shows(time_window='day', page=1)
            else:
                # Fallback: if the first call didn't work, try with a different page
                trending_day = self.tmdb_enricher.get_trending_shows(time_window='week', page=2)
            
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
        except Exception as e:
            logger.error(f"Error getting trending shows: {e}")
            return []
    
    def _get_discover_shows(self, watched_tmdb_ids: Set[int], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top-rated shows that the user hasn't watched.
        
        Args:
            watched_tmdb_ids: Set of TMDB IDs the user has already watched
            limit: Maximum number of top-rated shows to fetch
            
        Returns:
            List of top-rated shows
        """
        try:
            # For now, we'll use a combination of search and discovery
            discover_shows = []
            
            # Try direct discover API first for top-rated shows
            try:
                discover = tmdb.Discover()
                response = discover.tv(
                    sort_by='vote_average.desc',
                    vote_count_gte=100,  # At least 100 votes for quality
                    page=1
                )
                
                top_rated_results = response.get('results', [])
                for show in top_rated_results:
                    processed_show = {
                        'tmdb_id': show.get('id'),
                        'title': show.get('name'),
                        'overview': show.get('overview'),
                        'first_air_date': show.get('first_air_date'),
                        'popularity': show.get('popularity'),
                        'vote_average': show.get('vote_average'),
                        'vote_count': show.get('vote_count'),
                        'poster_path': show.get('poster_path')
                    }
                    
                    # Add poster URL if available
                    if processed_show.get('poster_path'):
                        processed_show['poster_url'] = f"https://image.tmdb.org/t/p/w500{processed_show['poster_path']}"
                    
                    discover_shows.append(processed_show)
            except Exception as e:
                logger.warning(f"Error getting top-rated shows from discover API: {e}")
                # Fallback to search
                top_rated_shows = self.tmdb_enricher.search_shows('top rated tv shows')
                discover_shows.extend(top_rated_shows)
            
            # Get some recent shows
            try:
                current_year = datetime.datetime.now().year
                discover = tmdb.Discover()
                response = discover.tv(
                    first_air_date_year=current_year,
                    sort_by='popularity.desc',
                    page=1
                )
                
                recent_results = response.get('results', [])
                for show in recent_results:
                    processed_show = {
                        'tmdb_id': show.get('id'),
                        'title': show.get('name'),
                        'overview': show.get('overview'),
                        'first_air_date': show.get('first_air_date'),
                        'popularity': show.get('popularity'),
                        'vote_average': show.get('vote_average'),
                        'vote_count': show.get('vote_count'),
                        'poster_path': show.get('poster_path')
                    }
                    
                    # Add poster URL if available
                    if processed_show.get('poster_path'):
                        processed_show['poster_url'] = f"https://image.tmdb.org/t/p/w500{processed_show['poster_path']}"
                    
                    discover_shows.append(processed_show)
            except Exception as e:
                logger.warning(f"Error getting recent shows from discover API: {e}")
                # Fallback to search
                recent_shows = self.tmdb_enricher.search_shows(f'new tv shows {current_year}')
                discover_shows.extend(recent_shows)
            
            # If discover_shows is still empty, use search as a fallback
            if not discover_shows:
                top_rated_shows = self.tmdb_enricher.search_shows('top rated tv shows')
                discover_shows.extend(top_rated_shows)
                
                current_year = datetime.datetime.now().year
                recent_shows = self.tmdb_enricher.search_shows(f'new tv shows {current_year}')
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
        except Exception as e:
            logger.error(f"Error getting discover shows: {e}")
            return []
    
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