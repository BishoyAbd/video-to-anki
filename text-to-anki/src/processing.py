# --- Helper Functions ---
import json
import logging  # Added for diagnostics
from typing import Generator, Optional, List, Dict  # Added imports

logger = logging.getLogger(__name__)  # Added for diagnostics

def split_into_chunks(text: str, word_limit: int) -> Generator[str, None, None]:
    """
    Splits text into chunks based on a strict word limit.

    Args:
        text: The full text to split.
        word_limit: The exact number of words desired per chunk (last chunk may be smaller).

    Yields:
        String chunks of the text.
    """
    words = text.split()
    # Diagnostic log:
    logger.info(f"split_into_chunks called with word_limit={word_limit}, processing text with {len(words)} words.")

    if not words:
        return

    current_chunk_words = []
    current_word_count = 0

    for word in words:
        current_chunk_words.append(word)
        current_word_count += 1

        if current_word_count == word_limit:
            yield " ".join(current_chunk_words)
            current_chunk_words = []
            current_word_count = 0

    # Yield any remaining words in the last chunk
    if current_chunk_words:
        yield " ".join(current_chunk_words)


def extract_json_from_response_text(text_response: str) -> Optional[List[Dict]]:
    """
    Extracts JSON data from a text response.

    Args:
        text_response: The text response containing JSON data.

    Returns:
        A list of dictionaries parsed from the JSON, or None if parsing fails.
    """
    try:
        return json.loads(text_response)
    except json.JSONDecodeError:
        return None