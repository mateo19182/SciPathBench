# generate_benchmark_data.py
# A script to pre-calculate the shortest citation paths using the Inciteful.xyz API
# and save them to a file, ordered by difficulty.

import logging
import json
import itertools
import requests
import time
from tqdm import tqdm
from dataset import LANDMARK_PAPERS
from utils import setup_logging

# This is the correct API endpoint that returns the path data.
INCITEFUL_CONNECTOR_API_URL = "https://api.inciteful.xyz/connector"


def get_path_from_inciteful(start_id: str, end_id: str):
    """
    Fetches the shortest citation path and paper details from the Inciteful.xyz API.
    Returns a tuple: (path_ids, paper_details_list)
    """
    params = {"from": start_id, "to": end_id, "extend": "0"}

    try:
        response = requests.get(INCITEFUL_CONNECTOR_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        paths = data.get("paths", [])
        papers_details = data.get("papers", [])

        if paths and len(paths) > 0:
            return paths[0], papers_details
        return None, None
    except requests.exceptions.RequestException as e:
        logging.error(
            f"API request to Inciteful failed for {start_id} -> {end_id}: {e}"
        )
        return None, None
    except json.JSONDecodeError:
        logging.error(
            f"Failed to parse JSON response from Inciteful for {start_id} -> {end_id}"
        )
        return None, None


def generate_data():
    """
    Finds shortest paths for unique pairs from LANDMARK_PAPERS, ensuring start/end
    papers are not reused, and saves the simplified results to a JSON file.
    """
    setup_logging("benchmark_generator.log")
    logging.info(
        "Starting benchmark data generation using Inciteful.xyz connector API."
    )

    benchmark_pairs = []
    used_start_end_ids = set()  # Tracks start/end papers included in a saved path.

    paper_pairs = list(itertools.combinations(LANDMARK_PAPERS, 2))

    logging.info(f"Generated {len(paper_pairs)} unique pairs to process.")

    for start_id, end_id in tqdm(paper_pairs, desc="Processing pairs"):
        # Skip this pair if either paper has already been used as a start/end point
        if start_id in used_start_end_ids or end_id in used_start_end_ids:
            continue

        # Make the API call to Inciteful
        path_ids, path_details_list = get_path_from_inciteful(start_id, end_id)

        if path_ids:
            path_length = len(path_ids) - 1
            if path_length > 0:
                logging.info(
                    f"Found path of length {path_length} for {start_id} -> {end_id}. Adding to dataset."
                )

                # Create a simple mapping of ID to Title for easy lookup
                id_to_title_map = {
                    p["id"]: p.get("title", "Title Not Found")
                    for p in path_details_list
                }
                # Create a list of titles in the correct path order
                path_titles = [id_to_title_map.get(pid) for pid in path_ids]

                benchmark_pairs.append(
                    {
                        "difficulty": path_length,
                        "start_id": start_id,
                        "end_id": end_id,
                        "path_ids": path_ids,
                        "path_titles": path_titles,
                    }
                )
                # Add the start and end papers of this successful path to the used set
                used_start_end_ids.add(start_id)
                used_start_end_ids.add(end_id)
            else:
                logging.warning(
                    f"Path found for {start_id} -> {end_id} has length 0. Skipping."
                )
        else:
            logging.warning(f"No path found between {start_id} and {end_id}.")

        # Add a small delay to be respectful to the API
        time.sleep(1)

    # Sort the results by difficulty
    benchmark_pairs.sort(key=lambda x: x["difficulty"])

    output_filename = "benchmark_pairs.json"
    with open(output_filename, "w") as f:
        json.dump(benchmark_pairs, f, indent=4)

    logging.info(
        f"Successfully generated {len(benchmark_pairs)} unique benchmark pairs."
    )
    logging.info(f"Data saved to {output_filename}.")


if __name__ == "__main__":
    generate_data()
