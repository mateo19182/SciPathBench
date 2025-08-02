# utils.py
# Utility functions for logging and data processing.

import logging


def setup_logging(log_file):
    """
    Configures logging to both console and a file in a robust way.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


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
