# TV Show Recommender

A self-hosted TV show recommendation system that analyzes your Trakt.tv watch history, enriches it with TMDB metadata, builds a taste profile, and uses OpenAI to generate personalized TV show recommendations.

## Features

- Fetches your watch history from Trakt.tv
- Enriches show data with metadata from TMDB (The Movie Database)
- Builds a detailed taste profile based on your viewing preferences
- Retrieves trending and unseen shows as recommendation candidates
- Uses OpenAI's API to generate personalized recommendations
- Displays results on a beautiful web interface

## Requirements

- Python 3.11+
- Trakt.tv account and API credentials
- TMDB API key
- OpenAI API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tv-recommender.git
cd tv-recommender
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables by copying the example file:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your API keys:
```
TRAKT_CLIENT_ID=your_trakt_client_id
TRAKT_CLIENT_SECRET=your_trakt_client_secret
TRAKT_ACCESS_TOKEN=your_trakt_access_token
TRAKT_USERNAME=your_trakt_username
TMDB_API_KEY=your_tmdb_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Getting API Keys

### Trakt.tv
1. Create an account at [Trakt.tv](https://trakt.tv)
2. Go to [https://trakt.tv/oauth/applications](https://trakt.tv/oauth/applications)
3. Click "New Application"
4. Fill out the form (redirect URI can be `urn:ietf:wg:oauth:2.0:oob`)
5. After creating your app, note the Client ID and Client Secret
6. Use the [Trakt API Docs](https://trakt.docs.apiary.io/) to generate an access token

### TMDB
1. Create an account at [TMDB](https://www.themoviedb.org)
2. Go to [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)
3. Follow the instructions to request an API key

### OpenAI
1. Create an account at [OpenAI](https://platform.openai.com/)
2. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
3. Create a new API key

## Usage

### Running Locally

Run the application with:

```bash
python main.py
```

The web interface will be available at http://localhost:5000

### Docker

Build and run with Docker:

```bash
docker build -t tv-recommender .
docker run -p 5000:5000 --env-file .env tv-recommender
```

Or use docker-compose:

```bash
docker-compose up
```

## How It Works

1. **Fetch Watch History**: The app connects to Trakt.tv and fetches your watch history.
2. **Enrich with Metadata**: It then enriches each show with additional metadata from TMDB (genres, cast, keywords, etc.).
3. **Build Taste Profile**: The app analyzes your watching patterns to build a detailed taste profile.
4. **Get Candidate Shows**: It fetches trending shows and shows similar to ones you've highly rated.
5. **Generate Recommendations**: OpenAI's API compares your taste profile with candidate shows to generate personalized recommendations.
6. **Display Results**: The recommendations are displayed on a web page with explanations.

## Project Structure

```
tv-recommender/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                 # API keys, tokens
│
├── main.py              # Entry point
├── app/                 
│   ├── trakt_fetcher.py         # Fetches watch history
│   ├── tmdb_enricher.py         # Adds metadata to shows
│   ├── profile_builder.py       # Builds taste profile from watch history
│   ├── candidate_fetcher.py     # Gets trending + new shows
│   ├── recommender.py           # Uses LLM to select and explain shows
│   ├── output_web.py            # Renders HTML 
│   └── utils.py                 # Common helpers
│
└── templates/           # HTML templates for the output page
```

## License

MIT

## Disclaimer

This project is not affiliated with Trakt.tv, TMDB, or OpenAI. It is a personal project for educational purposes. 