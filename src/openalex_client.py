# openalex_client.py
# Client for interacting with the OpenAlex API, with requests-cache.

import requests
import requests_cache
import logging
from config import OPENALEX_API_BASE_URL, OPENALEX_USER_EMAIL

# Install a global cache for all requests. Responses will be cached for 1 day.
# The cache will be stored in a file named 'api_cache.sqlite'.
requests_cache.install_cache('api_cache', backend='sqlite', expire_after=86400)

class OpenAlexClient:
    """
    Handles all interactions with the OpenAlex API.
    Caching is handled automatically by requests-cache.
    """
    def __init__(self):
        self.headers = {'User-Agent': f'SciPathBench/1.0 (mailto:{OPENALEX_USER_EMAIL})'}

    def _normalize_id(self, identifier: str) -> str:
        """
        Normalize an OpenAlex work identifier to just the OpenAlex ID (e.g., 'W123...').
        Accepts full URLs like 'https://openalex.org/W123' or already-normalized IDs.
        """
        if not identifier:
            return identifier
        # split by slash and take last non-empty segment
        parts = [p for p in identifier.split('/') if p]
        return parts[-1] if parts else identifier

    def _make_request(self, endpoint, params=None):
        """Internal method to handle API requests."""
        try:
            if params is None:
                params = {}
            params['mailto'] = OPENALEX_USER_EMAIL

            response = requests.get(f"{OPENALEX_API_BASE_URL}{endpoint}", params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"API HTTP Error: {e} - URL: {e.response.url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"API Request Failed: {e}")
            return None

    def get_paper_by_id(self, openalex_id: str):
        """
        Retrieves a single paper's metadata. The request will be cached automatically.
        Accepts either a bare OpenAlex ID (W...) or a full URL.
        """
        norm = self._normalize_id(openalex_id)
        return self._make_request(f"/works/{norm}")

    def get_neighbors(self, openalex_id: str):
        """
        Gets all papers that a given paper cites (outgoing references).
        This is a forward-only search.
        Returns a list of normalized OpenAlex IDs (W...).
        """
        norm = self._normalize_id(openalex_id)
        work = self._make_request(f"/works/{norm}")
        if not work:
            return []

        refs = work.get('referenced_works', [])[:25]  # Limit to first 25 references
        # Normalize each neighbor id to 'W...'
        return [self._normalize_id(r) for r in refs]

    def get_many_papers(self, ids: list[str]) -> dict:
        """
        Fetch multiple works' metadata, leveraging cache. 
        Returns a mapping id -> JSON or None.
        """
        results = {}
        for pid in ids:
            norm = self._normalize_id(pid)
            results[norm] = self.get_paper_by_id(norm)
        return results
