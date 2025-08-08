# dataset.py
# A curated list of landmark academic papers (using their OpenAlex IDs)
# to be used as a pool for generating benchmark pairs.

from src.services.openalex_client import OpenAlexClient
from src.config import LANDMARK_DATA_FILE, LANDMARK_ID_PREFERENCE
import logging
import json
from pathlib import Path

def _load_landmark_papers_from_file(file_path: str) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        logging.warning(f"LANDMARK_DATA_FILE not found at {file_path}; falling back to empty list")
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            items = json.load(f)
        ids: list[str] = []
        if LANDMARK_ID_PREFERENCE == "doi":
            for item in items:
                doi = item.get("doi")
                if doi:
                    ids.append(str(doi))
                elif item.get("openalex_id"):
                    ids.append(str(item.get("openalex_id")))
        else:
            for item in items:
                if item.get("openalex_id"):
                    ids.append(str(item.get("openalex_id")))
                elif item.get("doi"):
                    ids.append(str(item.get("doi")))
        return ids
    except Exception as e:
        logging.error(f"Failed to load LANDMARK_DATA_FILE {file_path}: {e}")
        return []


LANDMARK_PAPERS = _load_landmark_papers_from_file(LANDMARK_DATA_FILE)

DOI_PAPERS = [
    # "10.1038/nature12373",
    "10.1186/1756-8722-6-59",
    
]


# Function to check if a OpenAlex ID is valid
def is_valid_openalex_id(client, paper_id: str) -> bool:
    """
    Validates if the given OpenAlex ID is in the LANDMARK_PAPERS list.
    """
    try:
        paper = client.get_paper_by_id(paper_id)
        if paper is not None:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error validating OpenAlex ID {paper_id}: {e}")
        logging.error(f"Error validating OpenAlex ID {paper_id}: {e}")
        return False


def test_dataset():
    client = OpenAlexClient()
    for paper_id in LANDMARK_PAPERS:
        if is_valid_openalex_id(client, paper_id):
            print(f"Paper ID {paper_id} is valid.")
        else:
            print(f"Paper ID {paper_id} is NOT valid.")
