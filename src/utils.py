# utils.py
# Utility functions for logging and data processing.

import logging


def setup_logging(log_file):
    """Configures logging to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def reconstruct_abstract(inverted_index: dict) -> str:
    """
    Reconstructs the abstract text from OpenAlex's inverted index format.
    """
    if not inverted_index:
        return "Abstract not available."

    try:
        # Find the maximum index to determine the length of the list
        max_len = max(
            max(positions) for positions in inverted_index.values() if positions
        )
        abstract_list = [""] * (max_len + 1)

        for word, positions in inverted_index.items():
            for pos in positions:
                abstract_list[pos] = word

        return " ".join(filter(None, abstract_list))
    except (ValueError, TypeError):
        # Handle cases where the inverted_index is empty or malformed
        return "Abstract could not be reconstructed."
