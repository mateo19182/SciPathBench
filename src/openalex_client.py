# openalex_client.py
# Client for interacting with the OpenAlex API.

import requests
import logging
from config import OPENALEX_API_BASE_URL, OPENALEX_USER_EMAIL


class OpenAlexClient:
    """
    Handles all interactions with the OpenAlex API.
    This class is responsible for fetching paper data and tracking API calls.
    """

    def __init__(self):
        self.api_call_count = 0
        self.headers = {
            "User-Agent": f"SciPathBench/1.0 (mailto:{OPENALEX_USER_EMAIL})"
        }

    def _make_request(self, endpoint, params=None):
        """Internal method to handle API requests and count them."""
        self.api_call_count += 1
        try:
            if params is None:
                params = {}
            params["mailto"] = OPENALEX_USER_EMAIL

            response = requests.get(
                f"{OPENALEX_API_BASE_URL}{endpoint}",
                params=params,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"API HTTP Error: {e} - URL: {e.response.url}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"API Request Failed: {e}")
            return None

    def get_paper_by_id(self, openalex_id: str):
        """Retrieves a single paper's full metadata by its OpenAlex ID."""
        logging.info(
            f"API CALL {self.api_call_count + 1}: Getting details for paper ID {openalex_id}"
        )
        return self._make_request(f"/works/{openalex_id}")

    def get_neighbors(self, openalex_id: str):
        """
        Gets all papers that a given paper cites (outgoing) and that cite it (incoming).
        This treats the graph as undirected. Returns a list of neighbor IDs.
        This costs 2 API calls.
        """
        logging.info(
            f"API CALLS ~{self.api_call_count + 1}-{self.api_call_count + 2}: Getting all neighbors for {openalex_id}"
        )

        # This first call is counted by get_paper_by_id
        work = self.get_paper_by_id(openalex_id)
        if not work:
            return []

        # Get outgoing citations (already part of the 'work' object)
        citations = work.get("referenced_works", [])

        # Get incoming citations (this is the second API call)
        references_data = self._make_request(
            "/works", params={"filter": f"cites:{openalex_id}", "select": "id"}
        )
        references = (
            [item["id"].split("/")[-1] for item in references_data.get("results", [])]
            if references_data
            else []
        )

        return list(set(citations + references))

    def reset_api_call_count(self):
        """Resets the counter for a new run."""
        self.api_call_count = 0

    def get_api_call_count(self):
        """Returns the total number of API calls made."""
        return self.api_call_count
