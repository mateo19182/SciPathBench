# openalex_client.py
# Client for interacting with the OpenAlex API, with requests-cache.

import os
import time
import random
import re
import requests
import requests_cache
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import (
    OPENALEX_API_BASE_URL,
    OPENALEX_USER_EMAIL,
    OPENCITATIONS_API_KEY,
    OPENALEX_CACHE_BACKEND,
    OPENALEX_CACHE_NAME,
    OPENALEX_CACHE_EXPIRE_SECONDS,
    OPENALEX_MAX_RETRIES,
    OPENALEX_RETRY_BACKOFF_SECONDS,
    OPENALEX_MAX_WORKERS,
)

class OpenAlexClient:
    """
    Handles all interactions with the OpenAlex API.
    Caching is handled automatically by requests-cache.
    """
    def __init__(self):
        self.headers = {'User-Agent': f'SciPathBench/1.0 (mailto:{OPENALEX_USER_EMAIL})'}
        # Ensure cache directory exists when using file-based cache
        if OPENALEX_CACHE_BACKEND in {"sqlite", "filesystem"}:
            cache_dir = os.path.dirname(OPENALEX_CACHE_NAME)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

        self.session = requests_cache.CachedSession(
            OPENALEX_CACHE_NAME,
            backend=OPENALEX_CACHE_BACKEND,
            expire_after=OPENALEX_CACHE_EXPIRE_SECONDS,
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
        """Internal method to handle API requests with basic 429 retry/backoff."""
        if params is None:
            params = {}
        params['mailto'] = OPENALEX_USER_EMAIL

        url = f"{OPENALEX_API_BASE_URL}{endpoint}"
        attempts = 0
        while True:
            try:
                response = self.session.get(url, params=params, headers=self.headers)

                # Fast path: success
                if response.status_code == 200:
                    return response.json()

                # Respect 404 without retry
                if response.status_code == 404:
                    logging.warning(f"OpenAlex 404 Not Found: {response.url}")
                    return None

                # Handle rate limiting with Retry-After header
                if response.status_code == 429:
                    attempts += 1
                    if attempts > OPENALEX_MAX_RETRIES:
                        logging.error(f"OpenAlex 429 Too Many Requests after {OPENALEX_MAX_RETRIES} retries: {response.url}")
                        return None
                    retry_after = response.headers.get('Retry-After')
                    try:
                        wait_seconds = float(retry_after) if retry_after is not None else OPENALEX_RETRY_BACKOFF_SECONDS
                    except ValueError:
                        wait_seconds = OPENALEX_RETRY_BACKOFF_SECONDS
                    # Add small jitter to avoid thundering herd
                    wait_seconds += random.uniform(0, 0.25 * OPENALEX_RETRY_BACKOFF_SECONDS)
                    logging.warning(f"OpenAlex 429 received. Backing off for {wait_seconds:.2f}s (attempt {attempts}/{OPENALEX_MAX_RETRIES}) -> {response.url}")
                    time.sleep(wait_seconds)
                    continue

                # Retry on transient 5xx
                if 500 <= response.status_code < 600:
                    attempts += 1
                    if attempts > OPENALEX_MAX_RETRIES:
                        logging.error(f"OpenAlex {response.status_code} after {OPENALEX_MAX_RETRIES} retries: {response.url}")
                        return None
                    wait_seconds = OPENALEX_RETRY_BACKOFF_SECONDS * (2 ** (attempts - 1))
                    wait_seconds += random.uniform(0, 0.25 * OPENALEX_RETRY_BACKOFF_SECONDS)
                    logging.warning(f"OpenAlex {response.status_code}. Retrying in {wait_seconds:.2f}s (attempt {attempts}/{OPENALEX_MAX_RETRIES}) -> {response.url}")
                    time.sleep(wait_seconds)
                    continue

                # Other client errors: no retry
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                logging.error(f"API HTTP Error: {e} - URL: {getattr(e.response, 'url', url)}")
                return None
            except requests.exceptions.RequestException as e:
                attempts += 1
                if attempts > OPENALEX_MAX_RETRIES:
                    logging.error(f"API Request Failed after retries: {e} - URL: {url}")
                    return None
                wait_seconds = OPENALEX_RETRY_BACKOFF_SECONDS * (2 ** (attempts - 1))
                wait_seconds += random.uniform(0, 0.25 * OPENALEX_RETRY_BACKOFF_SECONDS)
                logging.warning(f"API Request error '{e}'. Retrying in {wait_seconds:.2f}s (attempt {attempts}/{OPENALEX_MAX_RETRIES}) -> {url}")
                time.sleep(wait_seconds)
    
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
        
    def _extract_openalex_ids_from_opencitations(self, oc_items: list[dict]) -> list[str]:
        """
        Attempt to extract OpenAlex work IDs from OpenCitations records when present.
        Returns a list of 'W...' IDs.
        """
        if not oc_items:
            return []
        results: list[str] = []
        pattern = re.compile(r"openalex:(W\d+)")
        for item in oc_items:
            for key in ("cited", "citing"):
                value = item.get(key)
                if not value or not isinstance(value, str):
                    continue
                for match in pattern.findall(value):
                    if match not in results:
                        results.append(match)
        return results

    def _is_doi(self, identifier: str) -> bool:
        if not identifier:
            return False
        if identifier.lower().startswith('doi:'):
            return True
        if identifier.lower().startswith('https://doi.org/'):
            return True
        # Basic DOI pattern: starts with 10. and contains a slash
        return bool(re.match(r"^10\.\d{4,9}/\S+", identifier))

    def _clean_doi(self, identifier: str) -> str:
        doi = identifier.strip()
        doi = doi[len('doi:'):] if doi.lower().startswith('doi:') else doi
        doi = doi[len('https://doi.org/'):] if doi.lower().startswith('https://doi.org/') else doi
        return doi

    def get_paper_by_id(self, openalex_id: str):
        """
        Retrieves a single paper's metadata. The request will be cached automatically.
        Accepts either a bare OpenAlex ID (W...) or a full URL.
        """
        if self._is_doi(openalex_id):
            clean_doi = self._clean_doi(openalex_id)
            return self._make_request(f"/works/doi:{clean_doi}")
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
            # Get citations from OpenCitations API (preferred when DOI is available)
            citations = self._make_open_citations_request(doi)
            if not citations:
                return []
            openalex_ids = self._extract_openalex_ids_from_opencitations(citations)
            return openalex_ids[:25] if openalex_ids else []
            
        elif id:
            # Get citations from OpenAlex API
            # If the provided id is actually a DOI, route to DOI branch
            if self._is_doi(id):
                doi_clean = self._clean_doi(id)
                return self.get_neighbors(doi=doi_clean)

            norm = self._normalize_id(id)

            work = self._make_request(f"/works/{norm}")
            if not work:
                return []

            # Prefer OpenCitations if DOI is present to reduce OpenAlex graph load
            doi_value = (work.get('ids') or {}).get('doi')
            if doi_value:
                oc_items = self._make_open_citations_request(doi_value)
                if oc_items:
                    oc_openalex_ids = self._extract_openalex_ids_from_opencitations(oc_items)
                    if oc_openalex_ids:
                        return oc_openalex_ids[:25]

            refs = work.get('referenced_works', [])[:25]  # Limit to first 25 references
            # Normalize each neighbor id to 'W...'
            return [self._normalize_id(r) for r in refs]
        else:
            logging.error("Invalid or missing id/doi.")
            return []
        
    def get_many_papers(self, ids: list[str], max_workers: int = OPENALEX_MAX_WORKERS) -> dict:
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

    def get_top_papers(self, limit: int, since_year: int | None = None, concept_id: str | None = None) -> list[dict]:
        """
        Retrieve top-cited papers from OpenAlex.

        Args:
            limit: Number of papers to return (must be between 1 and 200).
            since_year: Optional lower bound publication year (e.g., 2015).
            concept_id: Optional OpenAlex concept ID (e.g., C41008148 for Machine Learning).

        Returns:
            A list of work JSON objects from OpenAlex, sorted by citations desc.
        """
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")

        filters = ["has_doi:true"]
        if since_year is not None:
            filters.append(f"from_publication_date:{since_year}-01-01")
        if concept_id is not None and concept_id.strip():
            filters.append(f"concepts.id:{concept_id}")

        params = {
            "sort": "cited_by_count:desc",
            "per_page": limit,
            "filter": ",".join(filters) if filters else None,
        }
        # Remove None params to avoid sending 'filter=None'
        params = {k: v for k, v in params.items() if v is not None}

        data = self._make_request("/works", params=params)
        if not data:
            return []
        return data.get("results", []) or []
