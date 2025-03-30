"""
Module for rendering the recommendations to a web page.
"""

import logging
from typing import Dict, Any
from flask import Flask, render_template
from datetime import datetime

from app.utils import format_date, truncate_text

logger = logging.getLogger(__name__)

def create_app(recommendations: Dict[str, Any]) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        recommendations: Recommendations data
        
    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__, 
        template_folder='../templates',
        static_folder='../static'
    )
    
    # Register routes
    @app.route('/')
    def home():
        """Render the home page with recommendations."""
        recs = recommendations.get('recommendations', [])
        explanation = recommendations.get('overall_explanation', '')
        
        # Format dates and process text for display
        for rec in recs:
            if 'first_air_date' in rec:
                rec['formatted_date'] = format_date(rec['first_air_date'])
            if 'overview' in rec:
                rec['truncated_overview'] = truncate_text(rec['overview'], 150)
        
        # Get the current date for the page
        current_date = datetime.now().strftime("%B %d, %Y")
        
        return render_template(
            'recommendations.html',
            recommendations=recs,
            overall_explanation=explanation,
            current_date=current_date
        )
    
    @app.route('/about')
    def about():
        """Render the about page."""
        return render_template('about.html')
    
    # Simple error handlers that don't use templates
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors with a simple message."""
        return "Page not found. <a href='/'>Go to home page</a>", 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors with a simple message."""
        logger.error(f"Server error: {e}")
        return "Server error. Please try again later. <a href='/'>Go to home page</a>", 500
    
    return app 