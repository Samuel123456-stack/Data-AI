import aiohttp
import asyncio
from typing import Any

from agent.config import Config

Config.validate()

TMDB_KEY = Config.TMDB_API_KEY
TMDB_API_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_URL = 'https://image.tmdb.org/t/p/w500'


def _build_poster_url(path: str | None) -> str | None:
    if not path:
        return None
    return f'{TMDB_IMAGE_URL}{path}'


def _extract_director(credits: dict[str, Any]) -> str | None:
    crew = credits.get('crew') or []
    for person in crew:
        if person.get('job') == 'Director':
            return person.get('name')
    return None


def _extract_cast(credits: dict[str, Any]) -> list[dict[str, str | None]]:
    cast = credits.get('cast') or []
    return [
        {
            'name': person.get('name'),
            'character': person.get('character'),
        }
        for person in cast[:5]
        if person.get('name')
    ]


def _extract_streaming_platforms(providers: dict[str, Any], region: str) -> list[str]:
    region_data = (providers.get('results') or {}).get(region) or {}
    platforms: list[str] = []

    for group in ('flatrate', 'rent', 'buy'):
        for provider in region_data.get(group) or []:
            name = provider.get('provider_name')
            if name and name not in platforms:
                platforms.append(name)

    return platforms


def _normalize_movie(data: dict[str, Any], region: str) -> dict[str, Any]:
    release_date = data.get('release_date') or ''
    runtime = data.get('runtime')
    credits = data.get('credits') or {}
    providers = data.get('watch/providers') or {}

    return {
        'source': 'TMDB',
        'tmdb_id': data.get('id'),
        'title': data.get('title') or data.get('original_title'),
        'release_year': int(release_date[:4]) if release_date[:4].isdigit() else None,
        'director': _extract_director(credits),
        'genres': [genre.get('name') for genre in data.get('genres') or [] if genre.get('name')],
        'imdb_rating': round(float(data.get('vote_average') or 0), 1),
        'duration_minutes': runtime,
        'primary_language': data.get('original_language'),
        'synopsis': data.get('overview'),
        'age_rating': None,
        'content_warnings': None,
        'cast': _extract_cast(credits),
        'poster_url': _build_poster_url(data.get('poster_path')),
        'streaming_platforms': _extract_streaming_platforms(providers, region),
    }


def _pick_best_result(results: list[dict[str, Any]], release_year: int | None = None) -> dict[str, Any] | None:
    if not results:
        return None

    if release_year:
        same_year = [
            movie for movie in results
            if (movie.get('release_date') or '').startswith(str(release_year))
        ]
        with_poster = [movie for movie in same_year if movie.get('poster_path')]
        if with_poster:
            return with_poster[0]
        if same_year:
            return same_year[0]

    with_poster = [movie for movie in results if movie.get('poster_path')]
    return with_poster[0] if with_poster else results[0]


async def search_movie(
    title: str,
    api: str = TMDB_KEY,
    language: str = 'pt-BR',
    region: str = 'BR',
    release_year: int | None = None,
) -> dict[str, Any] | str:
    if not api:
        raise ValueError('TMDB_API_KEY was not found')

    timeout = aiohttp.ClientTimeout(total=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(
                f'{TMDB_API_URL}/search/movie',
                params={
                    'api_key': api,
                    'query': title,
                    'language': language,
                    'include_adult': 'false',
                    'page': 1,
                    **({'primary_release_year': release_year} if release_year else {}),
                },
            ) as response:
                if response.status != 200:
                    return f'Error while searching movie on TMDB. Status {response.status}'

                search_data = await response.json()
                results = search_data.get('results') or []
                if not results:
                    return f'Movie not found on TMDB: {title}'

                selected_movie = _pick_best_result(results, release_year)
                movie_id = selected_movie.get('id') if selected_movie else None
                if not movie_id:
                    return f'Movie found without TMDB id: {title}'

            async with session.get(
                f'{TMDB_API_URL}/movie/{movie_id}',
                params={
                    'api_key': api,
                    'language': language,
                    'append_to_response': 'credits,watch/providers',
                },
            ) as response:
                if response.status != 200:
                    return f'Error while getting movie details on TMDB. Status {response.status}'

                detail_data = await response.json()
                return _normalize_movie(detail_data, region)

        except asyncio.TimeoutError:
            return f'Timeout for searching the movie on TMDB: {title}'
        except aiohttp.ClientError as e:
            return f'Connection error for searching on TMDB: "{title}": {e}'


async def enrich_recommendation_posters(recommendation: Any) -> Any:
    """Garante poster_url buscando cada filme diretamente no TMDB após o agente responder."""
    for movie in getattr(recommendation, 'movies', []):
        result = await search_movie(movie.title, release_year=movie.release_year)
        if isinstance(result, dict) and result.get('poster_url'):
            movie.poster_url = result['poster_url']
            if result.get('streaming_platforms') and not movie.streaming_platforms:
                movie.streaming_platforms = result['streaming_platforms']

    return recommendation
