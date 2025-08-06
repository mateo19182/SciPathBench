# openalex_client.py
# Client for interacting with the OpenAlex API, with requests-cache.

import requests
import requests_cache
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OPENALEX_API_BASE_URL, OPENALEX_USER_EMAIL, OPENCITATIONS_API_KEY

class OpenAlexClient:
    """
    Handles all interactions with the OpenAlex API.
    Caching is handled automatically by requests-cache.
    """
    def __init__(self):
        self.headers = {'User-Agent': f'SciPathBench/1.0 (mailto:{OPENALEX_USER_EMAIL})'}
        self.session = requests_cache.CachedSession(
            'api_cache',
            # backend='memory',
            expire_after=8640000,
            allowable_codes=[200, 404],
            allowable_methods=['GET'],
        )

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

            response = self.session.get(f"{OPENALEX_API_BASE_URL}{endpoint}", params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"API HTTP Error: {e} - URL: {e.response.url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"API Request Failed: {e}")
            return None
    
    def _make_open_citations_request(self, id, params=None):
        """
        Internal method to handle OpenCitations API requests.
        """
        try:
            if params is None:
                params = {}
            # Ensure endpoint doesn't already contain 'doi:' prefix
            clean_id = id.replace('doi:', '') if id.startswith('doi:') else id
            url = f"https://api.opencitations.net/index/v2/references/doi:{clean_id}"
            response = self.session.get(
                url,
                headers={"authorization": OPENCITATIONS_API_KEY},
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"OpenCitations API HTTP Error: {e} - URL: {e.response.url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"OpenCitations API Request Failed: {e}")
            return None
        
    def get_paper_by_id(self, openalex_id: str):
        """
        Retrieves a single paper's metadata. The request will be cached automatically.
        Accepts either a bare OpenAlex ID (W...) or a full URL.
        """
        norm = self._normalize_id(openalex_id)
        return self._make_request(f"/works/{norm}")

    def get_neighbors(self, id: str = None, doi: str = None):
        """
        Gets all papers that a given paper cites (outgoing references).
        This is a forward-only search.
        Returns a list of normalized OpenAlex IDs (W...).
        Source can be 'opencitations' or 'openalex'.
        """
        
        if doi:
            # Get citations from OpenCitations API
            citations = self._make_open_citations_request(doi)
            if not citations:
                return []
            
            openalex_ids = []
            for citation in citations:
                cited_field = citation.get('citing', '')
                # Extract openalex:W... from the cited field
                if 'openalex:' in cited_field:
                    parts = cited_field.split()
                    for part in parts:
                        if part.startswith('openalex:W'):
                            openalex_ids.append(part.replace('openalex:', ''))
            
            return openalex_ids[:25]  # Limit to first 25 references
            
        elif id:
            # Get citations from OpenAlex API
            norm = self._normalize_id(id)

            work = self._make_request(f"/works/{norm}")
            if not work:
                return []

            refs = work.get('referenced_works', [])[:25]  # Limit to first 25 references
            # Normalize each neighbor id to 'W...'
            return [self._normalize_id(r) for r in refs]
        else:
            logging.error(f"Invalid or missing id/doi.")
            return []
        
    def get_many_papers(self, ids: list[str], max_workers: int = 10) -> dict:
        """
        Fetch multiple works' metadata in parallel, leveraging cache. 
        Returns a mapping id -> JSON or None.
        """
        if not ids:
            return {}
            
        results = {}
        normalized_ids = [self._normalize_id(pid) for pid in ids]
        
        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_id = {
                executor.submit(self.get_paper_by_id, norm_id): norm_id 
                for norm_id in normalized_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                norm_id = future_to_id[future]
                try:
                    result = future.result()
                    results[norm_id] = result
                except Exception as e:
                    logging.error(f"Failed to fetch paper {norm_id}: {e}")
                    results[norm_id] = None
                    
        return results
