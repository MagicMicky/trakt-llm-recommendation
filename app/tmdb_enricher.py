"""
Module for enriching shows with metadata from TMDB API.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import tmdbsimple as tmdb

logger = logging.getLogger(__name__)

class TMDBEnricher:
    """Class to enrich show data with additional metadata from TMDB."""
    
    def __init__(self, api_key: str):
        """
        Initialize the TMDBEnricher.
        
        Args:
            api_key: TMDB API key
        """
        self.api_key = api_key
        tmdb.API_KEY = api_key
    
    def enrich_shows(self, shows: List[Dict[str, Any]], rate_limit_delay: float = 0.25) -> List[Dict[str, Any]]:
        """
        Enrich a list of shows with TMDB metadata.
        
        Args:
            shows: List of shows to enrich
            rate_limit_delay: Delay between API calls to avoid rate limiting
            
        Returns:
            List of enriched shows
        """
        enriched_shows = []
        
        logger.info(f"Enriching {len(shows)} shows with TMDB data")
        
        for i, show in enumerate(shows):
            try:
                # Check if we have a TMDB ID
                tmdb_id = show.get('tmdb_id')
                
                if not tmdb_id:
                    logger.warning(f"No TMDB ID for show {show.get('title')}, skipping enrichment")
                    enriched_shows.append(show)
                    continue
                
                logger.debug(f"Enriching show {i+1}/{len(shows)}: {show.get('title')}")
                
                # Get TMDB data
                enriched_show = self._get_tmdb_data(show, tmdb_id)
                enriched_shows.append(enriched_show)
                
                # Add delay to avoid rate limiting
                time.sleep(rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error enriching show {show.get('title')}: {e}")
                # Still add the original show to the list
                enriched_shows.append(show)
        
        logger.info(f"Enriched {len(enriched_shows)} shows with TMDB data")
        
        return enriched_shows
    
    def _get_tmdb_data(self, show: Dict[str, Any], tmdb_id: int) -> Dict[str, Any]:
        """
        Get TMDB data for a show and add it to the show data.
        
        Args:
            show: Show data to enrich
            tmdb_id: TMDB ID of the show
            
        Returns:
            Enriched show data
        """
        enriched_show = show.copy()
        
        try:
            # Get TV show details
            tv = tmdb.TV(tmdb_id)
            details = tv.info(append_to_response='credits,keywords,content_ratings,external_ids,images,recommendations,similar')
            
            # Add TMDB metadata to the show
            enriched_show.update({
                'tmdb_data': {
                    'id': details.get('id'),
                    'overview': details.get('overview'),
                    'first_air_date': details.get('first_air_date'),
                    'last_air_date': details.get('last_air_date'),
                    'status': details.get('status'),
                    'number_of_seasons': details.get('number_of_seasons'),
                    'number_of_episodes': details.get('number_of_episodes'),
                    'genres': [genre.get('name') for genre in details.get('genres', [])],
                    'networks': [network.get('name') for network in details.get('networks', [])],
                    'popularity': details.get('popularity'),
                    'vote_average': details.get('vote_average'),
                    'vote_count': details.get('vote_count'),
                    'poster_path': details.get('poster_path'),
                    'backdrop_path': details.get('backdrop_path'),
                }
            })
            
            # Add cast information
            if 'credits' in details and 'cast' in details['credits']:
                cast = details['credits']['cast'][:10]  # Limit to top 10 cast members
                enriched_show['tmdb_data']['cast'] = [
                    {
                        'id': actor.get('id'),
                        'name': actor.get('name'),
                        'character': actor.get('character'),
                        'profile_path': actor.get('profile_path')
                    } for actor in cast
                ]
            
            # Add crew information (focus on creators)
            if 'created_by' in details:
                creators = details['created_by']
                enriched_show['tmdb_data']['creators'] = [
                    {
                        'id': creator.get('id'),
                        'name': creator.get('name')
                    } for creator in creators
                ]
            
            # Add keywords/tags
            if 'keywords' in details and 'results' in details['keywords']:
                enriched_show['tmdb_data']['keywords'] = [
                    keyword.get('name') for keyword in details['keywords']['results']
                ]
            
            # Add content ratings
            if 'content_ratings' in details and 'results' in details['content_ratings']:
                # Try to find US rating
                us_rating = next((rating.get('rating') for rating in details['content_ratings']['results'] 
                                  if rating.get('iso_3166_1') == 'US'), None)
                enriched_show['tmdb_data']['content_rating'] = us_rating
            
            # Add similar shows
            if 'similar' in details and 'results' in details['similar']:
                similar_shows = details['similar']['results'][:5]  # Limit to 5 similar shows
                enriched_show['tmdb_data']['similar_shows'] = [
                    {
                        'id': s.get('id'),
                        'name': s.get('name'),
                        'overview': s.get('overview'),
                        'first_air_date': s.get('first_air_date'),
                        'poster_path': s.get('poster_path')
                    } for s in similar_shows
                ]
            
            # Calculate image URLs
            if enriched_show['tmdb_data'].get('poster_path'):
                enriched_show['tmdb_data']['poster_url'] = f"https://image.tmdb.org/t/p/w500{enriched_show['tmdb_data']['poster_path']}"
            
            if enriched_show['tmdb_data'].get('backdrop_path'):
                enriched_show['tmdb_data']['backdrop_url'] = f"https://image.tmdb.org/t/p/w1280{enriched_show['tmdb_data']['backdrop_path']}"
            
        except Exception as e:
            logger.error(f"Error getting TMDB data for show ID {tmdb_id}: {e}")
        
        return enriched_show 