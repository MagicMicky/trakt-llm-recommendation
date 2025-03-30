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
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        logger.error(f"Server error: {e}")
        return render_template('500.html'), 500
    
    return app

def create_templates():
    """Create HTML templates if they don't exist."""
    import os
    from pathlib import Path
    
    logger.info("Checking for HTML templates")
    
    templates_dir = Path('../templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create base template if it doesn't exist
    base_template_path = templates_dir / 'base.html'
    if not base_template_path.exists():
        with open(base_template_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}TV Show Recommendations{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #e1e1e1;
        }
        .navbar {
            background-color: #1f1f1f;
        }
        .card {
            background-color: #1f1f1f;
            border: none;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card-img-top {
            height: 300px;
            object-fit: cover;
        }
        .card-title {
            font-weight: bold;
            color: #ffffff;
        }
        .explanation-card {
            background-color: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .recommendation-date {
            color: #b3b3b3;
            font-style: italic;
            margin-bottom: 20px;
        }
        .unique-appeal {
            font-style: italic;
            background-color: rgba(0, 123, 255, 0.1);
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        footer {
            background-color: #1f1f1f;
            color: #b3b3b3;
            padding: 20px 0;
            margin-top: 50px;
        }
    </style>
    {% block additional_head %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark mb-4">
        <div class="container">
            <a class="navbar-brand" href="/">TV Show Recommender</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/about">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container">
        {% block content %}{% endblock %}
    </div>

    <footer class="text-center">
        <div class="container">
            <p>&copy; {% now year %} TV Show Recommender</p>
            <p>Powered by AI with Trakt and TMDB data</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>""")
        logger.info("Created base.html template")
    
    # Create recommendations template if it doesn't exist
    recs_template_path = templates_dir / 'recommendations.html'
    if not recs_template_path.exists():
        with open(recs_template_path, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}Your Personalized TV Show Recommendations{% endblock %}

{% block content %}
<h1 class="mb-4 text-center">Your Personalized TV Show Recommendations</h1>
<p class="recommendation-date text-center">Generated on {{ current_date }}</p>

{% if overall_explanation %}
<div class="explanation-card">
    <h4>Recommendation Strategy</h4>
    <p>{{ overall_explanation }}</p>
</div>
{% endif %}

{% if recommendations %}
    <div class="row">
        {% for rec in recommendations %}
            <div class="col-md-6 mb-4">
                <div class="card h-100">
                    {% if rec.poster_url %}
                        <img src="{{ rec.poster_url }}" class="card-img-top" alt="{{ rec.title }} poster">
                    {% else %}
                        <div class="card-img-top bg-secondary d-flex align-items-center justify-content-center">
                            <span class="text-white">No poster available</span>
                        </div>
                    {% endif %}
                    <div class="card-body">
                        <h5 class="card-title">{{ rec.title }}</h5>
                        {% if rec.formatted_date %}
                            <p class="card-text text-muted">{{ rec.formatted_date }}</p>
                        {% endif %}
                        {% if rec.truncated_overview %}
                            <p class="card-text">{{ rec.truncated_overview }}</p>
                        {% endif %}
                        <div class="mt-3">
                            <h6>Why you'll like it:</h6>
                            <p>{{ rec.explanation }}</p>
                        </div>
                        {% if rec.unique_appeal %}
                            <div class="unique-appeal">
                                <strong>What's fresh:</strong> {{ rec.unique_appeal }}
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        {% endfor %}
    </div>
{% else %}
    <div class="alert alert-warning text-center">
        <h4>No recommendations available</h4>
        <p>We couldn't generate recommendations at this time. Please try again later.</p>
    </div>
{% endif %}
{% endblock %}""")
        logger.info("Created recommendations.html template")
    
    # Create 404 template if it doesn't exist
    not_found_template_path = templates_dir / '404.html'
    if not not_found_template_path.exists():
        with open(not_found_template_path, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}Page Not Found{% endblock %}

{% block content %}
<div class="text-center my-5">
    <h1 class="display-1">404</h1>
    <h2 class="mb-4">Page Not Found</h2>
    <p class="lead">The page you're looking for doesn't exist or has been moved.</p>
    <a href="/" class="btn btn-primary mt-3">Go Home</a>
</div>
{% endblock %}""")
        logger.info("Created 404.html template")
    
    # Create 500 template if it doesn't exist
    error_template_path = templates_dir / '500.html'
    if not error_template_path.exists():
        with open(error_template_path, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}Server Error{% endblock %}

{% block content %}
<div class="text-center my-5">
    <h1 class="display-1">500</h1>
    <h2 class="mb-4">Server Error</h2>
    <p class="lead">Something went wrong on our end. Please try again later.</p>
    <a href="/" class="btn btn-primary mt-3">Go Home</a>
</div>
{% endblock %}""")
        logger.info("Created 500.html template")
    
    # Create about template if it doesn't exist
    about_template_path = templates_dir / 'about.html'
    if not about_template_path.exists():
        with open(about_template_path, 'w', encoding='utf-8') as f:
            f.write("""{% extends "base.html" %}

{% block title %}About TV Show Recommender{% endblock %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-lg-8">
        <h1 class="mb-4">About TV Show Recommender</h1>
        
        <div class="card mb-4">
            <div class="card-body">
                <h4>How It Works</h4>
                <p>TV Show Recommender is a personalized recommendation system that analyzes your watching history from Trakt.tv and generates custom recommendations specifically tailored to your taste.</p>
                
                <h5 class="mt-4">The recommendation process:</h5>
                <ol>
                    <li>We fetch your watch history from Trakt.tv</li>
                    <li>We enrich that data with additional information from The Movie Database (TMDB)</li>
                    <li>We analyze your viewing patterns to build a detailed taste profile</li>
                    <li>We identify trending and well-rated shows that you haven't seen yet</li>
                    <li>Our AI system compares your taste profile with potential new shows</li>
                    <li>We generate a curated list of recommendations with personalized explanations</li>
                </ol>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-body">
                <h4>Data Sources</h4>
                <p>We use the following data sources:</p>
                <ul>
                    <li><a href="https://trakt.tv" target="_blank">Trakt.tv</a> - For your watching history</li>
                    <li><a href="https://www.themoviedb.org" target="_blank">The Movie Database (TMDB)</a> - For show metadata and images</li>
                </ul>
                <p>We prioritize your privacy and only process the data necessary to generate recommendations.</p>
            </div>
        </div>
        
        <div class="card">
            <div class="card-body">
                <h4>Technology</h4>
                <p>This application is built with:</p>
                <ul>
                    <li>Python and Flask for the backend</li>
                    <li>OpenAI's GPT models for intelligent recommendation generation</li>
                    <li>Bootstrap for the frontend design</li>
                </ul>
                <p>It's an open-source project designed to help TV enthusiasts discover their next favorite show.</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}""")
        logger.info("Created about.html template")
    
    logger.info("HTML templates created successfully") 