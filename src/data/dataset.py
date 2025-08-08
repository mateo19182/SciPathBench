# dataset.py
# A curated list of landmark academic papers (using their OpenAlex IDs)
# to be used as a pool for generating benchmark pairs.

from src.services.openalex_client import OpenAlexClient
import logging

LANDMARK_PAPERS = [
    "W4298289240",
    "W4385245566",
    "W2618530766",
    "W1995875735",
    "W2001771035",
    "W2949650786",
    "W4300912893",
    "W2173213060",
    "W2185907055",
    "W4239510810",
    "W2064675550",
    "W2046533349",
    "W2169528473",
    "W2036265926",
    "W4299608875",
    "W4206489165",
    "W2126466006",
    "W2028590532",
]

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
