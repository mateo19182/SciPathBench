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

    def _make_request(self, endpoint, params=None):
        """Internal method to handle API requests and count them."""
        try:
            if params is None:
                params = {}
            params['mailto'] = OPENALEX_USER_EMAIL

            response = requests.get(f"{OPENALEX_API_BASE_URL}{endpoint}", params=params, headers=self.headers)
            
            # Log whether the response came from the cache
            # if response.from_cache:
            #     # logging.info(f"CACHE HIT: Response for {response.url} loaded from cache.")
            #     # We don't count cached responses as new API calls
            #     self.api_call_count -= 1
            # else:
            #     logging.info(f"CACHE MISS: Making live API call to {response.url}")

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
        """
        return self._make_request(f"/works/{openalex_id}")

    def get_neighbors(self, openalex_id: str):
        """
        Gets all papers that a given paper cites (outgoing).
        This is a forward-only search.
        """
        work = self._make_request(f"/works/{openalex_id}")
        if not work:
            return []
        
        # Return only outgoing citations (referenced_works)
        return work.get('referenced_works', [])
