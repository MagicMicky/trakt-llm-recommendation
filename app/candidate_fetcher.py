"""
Module for fetching candidate shows for recommendations.
"""

import logging
from typing import List, Dict, Any, Set
import datetime
import tmdbsimple as tmdb
import time

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
    
    def get_candidates(self, watched_shows: List[Dict[str, Any]], max_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Get candidate shows for recommendations.
        
        Args:
            watched_shows: List of shows the user has already watched
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate shows for recommendations
        """
        logger.info(f"Fetching up to {max_candidates} trending candidate shows for recommendations")
        
        # Get watched show IDs to avoid recommending shows the user has already seen
        watched_tmdb_ids = self._get_watched_tmdb_ids(watched_shows)
        
        # Get trending shows with error handling
        try:
            trending_shows = self._fetch_trending_shows(num_pages=10)  # Fetch more pages of trending shows
            logger.info(f"Fetched {len(trending_shows)} trending shows as candidates")
            
            # Supplement with recent shows from current year if needed
            if len(trending_shows) < max_candidates:
                logger.info(f"Supplementing with recent shows from current year")
                current_year = datetime.datetime.now().year
                recent_shows = self._fetch_shows_for_year(current_year, num_pages=3)
                trending_shows.extend(recent_shows)
                logger.info(f"Added {len(recent_shows)} recent shows from {current_year}")
        except Exception as e:
            logger.error(f"Error fetching trending shows: {e}")
            trending_shows = []
        
        # If we don't have any candidates, just return an empty list
        if not trending_shows:
            logger.warning("No candidates found from API. Returning empty list.")
            return []
            
        # Remove shows the user has already watched
        filtered_shows = [
            show for show in trending_shows 
            if show.get('tmdb_id') and show.get('tmdb_id') not in watched_tmdb_ids
        ]
        
        # Remove duplicates while preserving order
        unique_shows = self._remove_duplicate_candidates(filtered_shows)
        
        # Limit to max_candidates
        candidates = unique_shows[:max_candidates]
        
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
    
    def _get_recent_shows(self, watched_tmdb_ids: Set[int], limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent shows from TMDB that the user hasn't watched.
        
        Args:
            watched_tmdb_ids: Set of TMDB IDs the user has already watched
            limit: Maximum number of recent shows to fetch
            
        Returns:
            List of recent shows
        """
        all_recent_shows = []
        current_year = datetime.datetime.now().year
        years_to_fetch = [current_year, current_year - 1]  # Current year and previous year
        
        # Number of pages to fetch (20 results per page, up to 25 pages max)
        pages_per_year = min(25, (limit // len(years_to_fetch)) // 20 + 1)
        
        logger.info(f"Fetching {pages_per_year} pages of shows for each of {len(years_to_fetch)} years")
        
        # Fetch shows for current and previous year
        for year in years_to_fetch:
            shows_for_year = self._fetch_shows_for_year(year, pages_per_year)
            all_recent_shows.extend(shows_for_year)
            logger.info(f"Fetched {len(shows_for_year)} shows for year {year}")
            
            # Break early if we have enough candidates
            if len(all_recent_shows) >= limit:
                break
        
        # Fetch additional trending shows if needed
        if len(all_recent_shows) < limit:
            logger.info(f"Not enough recent shows ({len(all_recent_shows)}), fetching trending shows to reach {limit}")
            trending = self._fetch_trending_shows(5)  # Fetch 5 pages of trending shows
            all_recent_shows.extend(trending)
            logger.info(f"Added {len(trending)} trending shows to candidates")
        
        # Remove shows the user has already watched
        filtered_shows = [
            show for show in all_recent_shows 
            if show.get('tmdb_id') and show.get('tmdb_id') not in watched_tmdb_ids
        ]
        
        # Remove duplicates while preserving order
        unique_shows = self._remove_duplicate_candidates(filtered_shows)
        
        # Limit the number of shows
        return unique_shows[:limit]
    
    def _fetch_shows_for_year(self, year: int, num_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch shows for a specific year.
        
        Args:
            year: Year to fetch shows for
            num_pages: Number of pages to fetch
            
        Returns:
            List of shows for the year
        """
        shows = []
        
        try:
            for page in range(1, num_pages + 1):
                discover = tmdb.Discover()
                response = discover.tv(
                    first_air_date_year=year,
                    sort_by='popularity.desc',
                    page=page,
                    include_adult=False
                )
                
                results = response.get('results', [])
                if not results:
                    logger.info(f"No more results for year {year} after page {page}")
                    break
                
                for show in results:
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
                    
                    shows.append(processed_show)
                
                # Add a short delay to prevent rate limiting
                time.sleep(0.25)
        
        except Exception as e:
            logger.error(f"Error fetching shows for year {year}: {e}")
        
        return shows
    
    def _fetch_trending_shows(self, num_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch trending shows from TMDB.
        
        Args:
            num_pages: Number of pages to fetch
            
        Returns:
            List of trending shows
        """
        trending_shows = []
        
        try:
            for page in range(1, num_pages + 1):
                discover = tmdb.Discover()
                response = discover.tv(
                    sort_by='popularity.desc',
                    page=page,
                    include_adult=False
                )
                
                results = response.get('results', [])
                if not results:
                    break
                
                for show in results:
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
                    
                    trending_shows.append(processed_show)
                
                # Add a short delay to prevent rate limiting
                time.sleep(0.25)
        
        except Exception as e:
            logger.error(f"Error fetching trending shows: {e}")
        
        return trending_shows
    
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