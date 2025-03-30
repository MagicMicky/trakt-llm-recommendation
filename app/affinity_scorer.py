"""
Module for calculating user affinity scores for watched shows.

This module analyzes Trakt watch data to infer how much a user liked each show
based on their watching patterns, ratings, and other behavior signals.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class AffinityScorer:
    """
    Class to calculate user affinity scores for TV shows based on watching behavior.
    Analyzes watch patterns to infer user preferences beyond explicit ratings.
    """
    
    def __init__(self):
        """Initialize the AffinityScorer."""
        # Configure default thresholds and weights
        self.config = {
            # Thresholds
            'binge_days_per_episode': 7,  # Was 3, increased to 7 days per episode to consider as "binged"
            'high_rating_threshold': 7,    # Was 8, lowered to 7 as the rating threshold (scale 1-10)
            'completion_threshold': 0.7,   # Was 0.85, lowered to 0.7 as % of episodes to consider "completed"
            'drop_threshold': 0.4,         # Was 0.3, increased to 0.4 as % of episodes below which show is "dropped"
            'min_episodes_for_valid': 2,    # Was 3, lowered to 2 minimum episodes for valid assessment
            
            # Weights for scoring
            'weight_high_rating': 4,        # Was 3, increased to 4 points for high rating
            'weight_completion': 3,         # Was 2, increased to 3 points for completing a show
            'weight_binge': 3,              # Was 2, increased to 3 points for binge-watching
            'weight_rewatch': 3,            # Points for rewatching (unchanged)
            'weight_incomplete': -1,        # Points for leaving incomplete (unchanged)
            'weight_dropped': -2,           # Points for dropping early (unchanged)
            
            # Score ranges
            'min_score': -5,
            'max_score': 10,
            
            # True/False flags thresholds
            'favorite_score_threshold': 5,  # Was 7, lowered to 5 for the score needed to consider a show a favorite
        }
    
    def calculate_affinity_scores(self, 
                                watched_shows: List[Dict[str, Any]], 
                                watch_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                                ratings: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Calculate affinity scores for a list of watched shows.
        
        Args:
            watched_shows: List of shows from Trakt API with basic watch info
            watch_history: Optional detailed episode-level watch history by show
            ratings: Optional list of show ratings from Trakt
            
        Returns:
            List of shows with added affinity scores and flags
        """
        logger.info(f"Calculating affinity scores for {len(watched_shows)} shows")
        
        # Add debug logging for input data
        if watch_history:
            logger.info(f"Watch history provided for {len(watch_history)} shows")
            # Sample a show to check data structure
            if watch_history:
                sample_show_id = next(iter(watch_history))
                sample_history = watch_history[sample_show_id]
                if sample_history:
                    logger.info(f"Sample watch history for show {sample_show_id}: {len(sample_history)} episodes")
                    logger.info(f"Sample episode data: {sample_history[0] if sample_history else 'None'}")
                    
                    # Check episode structure more carefully
                    if sample_history and 'episode' in sample_history[0]:
                        episode_info = sample_history[0]['episode']
                        logger.info(f"Episode structure details: {episode_info}")
        
        if ratings:
            logger.info(f"Ratings provided for {len(ratings)} shows")
        
        # Create a mapping of show IDs to ratings if available
        ratings_map = {}
        if ratings:
            for rating in ratings:
                if 'show' in rating and 'ids' in rating['show'] and 'trakt' in rating['show']['ids']:
                    ratings_map[rating['show']['ids']['trakt']] = rating.get('rating', 0)
            
            logger.info(f"Mapped {len(ratings_map)} ratings to show IDs")
        
        # Process each show
        enriched_shows = []
        
        # Track episode count adjustments for diagnostic purposes
        shows_with_adjusted_episode_count = 0
        total_episode_count_before = 0
        total_episode_count_after = 0
        
        for show in watched_shows:
            # Get trakt ID for lookup
            trakt_id = show.get('trakt_id')
            if not trakt_id:
                # Skip shows without a valid Trakt ID
                enriched_shows.append(show)
                continue
                
            # Get episode history for this show if available
            episode_history = None
            if watch_history and str(trakt_id) in watch_history:
                episode_history = watch_history[str(trakt_id)]
                
            # Keep track of original watched_episodes for diagnostics
            original_watched_episodes = show.get('watched_episodes', 0)
            total_episode_count_before += original_watched_episodes
                
            # Calculate affinity metrics
            affinity_data = self._calculate_show_affinity(
                show, 
                episode_history,
                ratings_map.get(trakt_id)
            )
            
            # Check if episode count was adjusted
            adjusted_watched_episodes = affinity_data.get('metrics', {}).get('watched_episodes', 0)
            if adjusted_watched_episodes != original_watched_episodes:
                shows_with_adjusted_episode_count += 1
            
            total_episode_count_after += adjusted_watched_episodes
            
            # Add affinity data to the show
            enriched_show = {**show, 'affinity': affinity_data}
            enriched_shows.append(enriched_show)
        
        # Add detailed diagnostic logging about episode count adjustments
        logger.info(f"Episode count statistics:")
        logger.info(f"  - Shows with adjusted episode counts: {shows_with_adjusted_episode_count}/{len(watched_shows)}")
        logger.info(f"  - Total episodes before adjustment: {total_episode_count_before}")
        logger.info(f"  - Total episodes after adjustment: {total_episode_count_after}")
        logger.info(f"  - Difference: {total_episode_count_after - total_episode_count_before}")
        
        # Log detailed stats about the affinity scores
        favorite_count = len(self.get_favorites(enriched_shows))
        binged_count = len(self.get_binged_shows(enriched_shows))
        dropped_count = len(self.get_dropped_shows(enriched_shows))
        completed_count = len(self.get_completed_shows(enriched_shows))
        
        # If we have no favorites, binged or dropped shows, log more details for debugging
        if favorite_count == 0 and binged_count == 0 and dropped_count == 0:
            # Log the top 5 scores to see how close they are to thresholds
            scores = [(s.get('title', 'Unknown'), 
                      s.get('affinity', {}).get('score', 0),
                      s.get('affinity', {}).get('flags', {})) 
                     for s in enriched_shows if 'affinity' in s]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 5 affinity scores:")
            for i, (title, score, flags) in enumerate(scores[:5]):
                logger.info(f"  {i+1}. {title}: Score={score}, Flags={flags}")
                
            # Check how many shows have time span data (needed for binge detection)
            shows_with_timespan = sum(1 for s in enriched_shows 
                                     if s.get('affinity', {}).get('metrics', {}).get('watch_time_span_days') is not None)
            logger.info(f"Only {shows_with_timespan} out of {len(enriched_shows)} shows have watch time span data")
        
        logger.info(f"Affinity scores calculated for {len(enriched_shows)} shows")
        return enriched_shows
    
    def _calculate_show_affinity(self, 
                               show: Dict[str, Any], 
                               episode_history: Optional[List[Dict[str, Any]]] = None,
                               rating: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate affinity metrics for a single show.
        
        Args:
            show: Show data with basic watch info
            episode_history: Detailed episode-level watch history
            rating: User's rating for the show (1-10)
            
        Returns:
            Dictionary with affinity metrics and scores
        """
        # Extract basic data
        watched_episodes = show.get('watched_episodes', 0)
        
        # Get total_episodes with better error handling and logging
        total_episodes = 0
        tmdb_data = show.get('tmdb_data', {})
        if 'number_of_episodes' in tmdb_data:
            total_episodes = tmdb_data.get('number_of_episodes', 0)
            if total_episodes == 0 and tmdb_data.get('status') != 'Canceled':
                # TMDB sometimes returns 0 for still-running shows
                logger.debug(f"TMDB reports 0 episodes for '{show.get('title')}' with status '{tmdb_data.get('status')}'")
        
        # If TMDB doesn't have episode count data (or has invalid data),
        # try to infer from episode history if available
        if total_episodes <= 0 and episode_history:
            # Try to count unique episodes
            unique_episodes = set()
            for record in episode_history:
                if 'episode' in record:
                    # Either use episode ID if available
                    if 'ids' in record['episode'] and 'trakt' in record['episode']['ids']:
                        unique_episodes.add(record['episode']['ids']['trakt'])
                    # Or use season/episode number combination
                    elif 'season' in record['episode'] and 'number' in record['episode']:
                        unique_episodes.add(f"S{record['episode']['season']}E{record['episode']['number']}")
            
            # If we found unique episodes and we have more than what TMDB reports,
            # use our count as the total (at minimum)
            unique_episode_count = len(unique_episodes)
            if unique_episode_count > 0:
                # Add a small buffer to account for potential unwatched episodes
                inferred_total = max(unique_episode_count, int(unique_episode_count * 1.1))
                logger.debug(f"Inferred total episodes for '{show.get('title')}': {inferred_total} (from {unique_episode_count} unique watched)")
                total_episodes = inferred_total
        
        plays = show.get('plays', 0)
        last_watched_at = show.get('last_watched_at')
        
        # Override watched_episodes count with actual data from episode_history if available
        if episode_history:
            # Count unique episodes (episode ID or episode+season combination)
            unique_episodes = set()
            for record in episode_history:
                if 'episode' in record:
                    # Either use episode ID if available
                    if 'ids' in record['episode'] and 'trakt' in record['episode']['ids']:
                        unique_episodes.add(record['episode']['ids']['trakt'])
                    # Or use season/episode number combination
                    elif 'season' in record['episode'] and 'number' in record['episode']:
                        unique_episodes.add(f"S{record['episode']['season']}E{record['episode']['number']}")
            
            # Update watched_episodes count
            watched_episodes_from_history = len(unique_episodes)
            if watched_episodes_from_history > 0:
                # Log for diagnostic purposes
                logger.debug(f"Updating watched episodes for '{show.get('title')}': {watched_episodes} → {watched_episodes_from_history}")
                watched_episodes = watched_episodes_from_history
        
        # Calculate basic metrics
        completion_ratio = self._calculate_completion_ratio(watched_episodes, total_episodes)
        time_span = self._calculate_watch_time_span(episode_history)
        binge_score = self._calculate_binge_score(time_span, watched_episodes)
        
        # Fix for is_rewatch calculation
        # Original: is_rewatch = plays > watched_episodes if watched_episodes > 0 else False
        # This may be too strict as it requires plays to exceed watched episodes
        # Instead, we'll consider a show rewatched if it has multiple plays per episode
        avg_plays_per_episode = plays / watched_episodes if watched_episodes > 0 else 0
        is_rewatch = avg_plays_per_episode >= 1.2  # Consider rewatched if 20% of episodes have been watched twice
        
        # Log some debug info for high-play shows
        if plays > 5 and watched_episodes > 0:
            logger.debug(f"Show with high play count: watched={watched_episodes}, plays={plays}, "
                        f"avg_plays={avg_plays_per_episode:.2f}, is_rewatch={is_rewatch}")
        
        # Calculate derived flags
        is_completed = completion_ratio >= self.config['completion_threshold']
        is_dropped = (not is_completed and 
                     watched_episodes >= self.config['min_episodes_for_valid'] and 
                     completion_ratio <= self.config['drop_threshold'])
        was_binged = binge_score > 0
        
        # Calculate the affinity score
        score = 0
        
        # Only apply scoring if we have enough episodes to make a judgment
        if watched_episodes >= self.config['min_episodes_for_valid'] or (rating is not None):
            # Rating-based score
            if rating is not None and rating >= self.config['high_rating_threshold']:
                score += self.config['weight_high_rating']
                
            # Completion-based score
            if is_completed:
                score += self.config['weight_completion']
            elif is_dropped:
                score += self.config['weight_dropped']
            elif watched_episodes > 0 and not is_completed:
                score += self.config['weight_incomplete']
                
            # Binge-based score
            if was_binged:
                score += self.config['weight_binge']
                
            # Rewatch-based score
            if is_rewatch:
                score += self.config['weight_rewatch']
        
        # Clamp score to min/max range
        score = max(self.config['min_score'], min(self.config['max_score'], score))
        
        # Determine if this is a favorite
        is_favorite = score >= self.config['favorite_score_threshold']
        
        # Compile final affinity data
        affinity_data = {
            'score': score,
            'metrics': {
                'completion_ratio': completion_ratio,
                'watch_time_span_days': time_span.days if time_span else None,
                'plays': plays,
                'explicit_rating': rating,
                'watched_episodes': watched_episodes,
                'total_episodes': total_episodes,
            },
            'flags': {
                'is_favorite': is_favorite,
                'is_completed': is_completed,
                'is_dropped': is_dropped,
                'was_binged': was_binged,
                'is_rewatch': is_rewatch,
                'has_rating': rating is not None
            }
        }
        
        return affinity_data
    
    def _calculate_completion_ratio(self, watched_episodes: int, total_episodes: int) -> float:
        """
        Calculate the ratio of watched episodes to total episodes.
        Handles cases where total_episodes might be unknown.
        
        Args:
            watched_episodes: Number of episodes watched
            total_episodes: Total episodes in the show
            
        Returns:
            Completion ratio (0.0-1.0)
        """
        if not total_episodes or total_episodes <= 0:
            # If we don't know the total, assume whatever they watched is what exists
            return 1.0
            
        # Account for potential reporting flukes - if they watched all episodes except 1-2
        # and the show is reasonably sized, consider it complete
        if total_episodes > 5 and watched_episodes >= total_episodes - 2:
            return 1.0
            
        return min(1.0, watched_episodes / total_episodes)
    
    def _calculate_watch_time_span(self, episode_history: Optional[List[Dict[str, Any]]]) -> Optional[timedelta]:
        """
        Calculate the time span between first and last episode watched.
        
        Args:
            episode_history: List of episode watch records with timestamps
            
        Returns:
            Time delta between first and last episode watched, or None if not available
        """
        if not episode_history:
            return None
            
        try:
            # Extract and parse timestamps
            timestamps = []
            for record in episode_history:
                if 'watched_at' in record:
                    try:
                        # Parse ISO format timestamp
                        timestamp = datetime.fromisoformat(record['watched_at'].replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing timestamp '{record.get('watched_at')}': {e}")
            
            if len(timestamps) <= 1:
                return None
                
            # Sort and calculate time span
            timestamps.sort()
            time_span = timestamps[-1] - timestamps[0]
            
            # Log if the time span is very small or very large
            if time_span.days < 1:
                logger.debug(f"Very short time span for episode history: {time_span}")
            
            return time_span
        except Exception as e:
            logger.warning(f"Error calculating watch time span: {e}")
            return None
    
    def _calculate_binge_score(self, time_span: Optional[timedelta], episode_count: int) -> float:
        """
        Calculate a binge score based on how quickly episodes were watched.
        
        Args:
            time_span: Time between first and last episode watched
            episode_count: Number of episodes watched
            
        Returns:
            Binge score (0.0-1.0) where higher means more intense binging
        """
        if not time_span or episode_count <= 1:
            return 0.0
            
        # Calculate days per episode
        days_per_episode = time_span.total_seconds() / (86400 * max(1, episode_count - 1))
        
        # If days per episode is less than threshold, it's a binge
        if days_per_episode <= self.config['binge_days_per_episode']:
            # Calculate intensity, where 0 days/episode is 1.0 and threshold days/episode is 0.0
            intensity = 1.0 - (days_per_episode / self.config['binge_days_per_episode'])
            return max(0.0, min(1.0, intensity))
            
        return 0.0
    
    def get_top_shows(self, 
                     shows_with_affinity: List[Dict[str, Any]], 
                     min_score: Optional[float] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top shows based on affinity scores.
        
        Args:
            shows_with_affinity: Shows with calculated affinity data
            min_score: Optional minimum score threshold
            limit: Maximum number of shows to return
            
        Returns:
            List of top shows by affinity score
        """
        # Filter by minimum score if specified
        filtered_shows = shows_with_affinity
        if min_score is not None:
            filtered_shows = [show for show in shows_with_affinity 
                             if 'affinity' in show and show['affinity'].get('score', 0) >= min_score]
        
        # Sort by affinity score (descending)
        sorted_shows = sorted(filtered_shows, 
                             key=lambda x: x.get('affinity', {}).get('score', 0),
                             reverse=True)
        
        # Return limited result
        return sorted_shows[:limit]
    
    def get_favorites(self, shows_with_affinity: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get shows marked as favorites based on affinity scores.
        
        Args:
            shows_with_affinity: Shows with calculated affinity data
            
        Returns:
            List of favorite shows
        """
        return [show for show in shows_with_affinity 
                if 'affinity' in show and 
                show['affinity'].get('flags', {}).get('is_favorite', False)]
    
    def get_binged_shows(self, shows_with_affinity: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get shows that were binged.
        
        Args:
            shows_with_affinity: Shows with calculated affinity data
            
        Returns:
            List of binged shows
        """
        return [show for show in shows_with_affinity 
                if 'affinity' in show and 
                show['affinity'].get('flags', {}).get('was_binged', False)]
    
    def get_dropped_shows(self, shows_with_affinity: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get shows that were dropped.
        
        Args:
            shows_with_affinity: Shows with calculated affinity data
            
        Returns:
            List of dropped shows
        """
        return [show for show in shows_with_affinity 
                if 'affinity' in show and 
                show['affinity'].get('flags', {}).get('is_dropped', False)]
    
    def get_potential_completionist_shows(self, shows_with_affinity: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get shows that are good candidates for a completionist viewer.
        This is a more relaxed version of favorite detection, useful for users who tend to complete shows.
        
        Args:
            shows_with_affinity: Shows with calculated affinity data
            
        Returns:
            List of shows that are nearly complete with good progress
        """
        completionist_shows = []
        
        for show in shows_with_affinity:
            if 'affinity' not in show:
                continue
                
            metrics = show['affinity'].get('metrics', {})
            flags = show['affinity'].get('flags', {})
            
            # Check for shows with substantial progress but not marked as favorites or completed
            watched = metrics.get('watched_episodes', 0)
            total = metrics.get('total_episodes', 0)
            
            # Conditions for a potential completionist show:
            # 1. Has watched a decent number of episodes (at least 5)
            # 2. Has good progress (at least 60%) but not completed
            # 3. Not already marked as a favorite or dropped
            is_potential = (
                watched >= 5 and
                total > 0 and
                0.6 <= (watched / total) < self.config['completion_threshold'] and
                not flags.get('is_favorite', False) and
                not flags.get('is_dropped', False)
            )
            
            if is_potential:
                completionist_shows.append(show)
                
        return completionist_shows
    
    def get_completed_shows(self, shows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get shows that were completed by the user.
        
        Args:
            shows: List of shows with affinity data
            
        Returns:
            List of completed shows
        """
        return [
            show for show in shows 
            if 'affinity' in show and 
            show['affinity'].get('flags', {}).get('is_completed', False)
        ] 