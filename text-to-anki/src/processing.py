import requests
import json
import time # Import time for potential delays between requests
# Import Dict from typing for compatibility with Python < 3.9
from typing import List, Dict, Optional, Tuple, Generator

# Use absolute import from the package root (src)
from .config import settings
# Import Anki models and functions
from .models import ProcessedSentencePair, AnkiNotePayload, AnkiAddResult
from .anki import add_notes_to_anki

LLM_API_URL = settings.LLM_API_URL
LLM_API_KEY = settings.LLM_API_KEY # Optional API key
LLM_MODEL_NAME = settings.LLM_MODEL_NAME
CHUNK_WORD_COUNT = settings.CHUNK_WORD_COUNT

# --- Helper Functions ---

def split_into_chunks(text: str, word_limit: int) -> Generator[str, None, None]:
    """
    Splits text into chunks based on an approximate word limit.
    Tries to split at sentence boundaries where possible near the limit.

    Args:
        text: The full text to split.
        word_limit: The desired approximate number of words per chunk.

    Yields:
        String chunks of the text.
    """
    words = text.split()
    if not words:
        return

    current_chunk_words = []
    current_word_count = 0
    sentence_enders = {'.', '?', '!'}

    for word in words:
        current_chunk_words.append(word)
        current_word_count += 1

        # Check if we've reached the limit and are at a potential sentence end
        if current_word_count >= word_limit and any(word.endswith(p) for p in sentence_enders):
            yield " ".join(current_chunk_words)
            current_chunk_words = []
            current_word_count = 0

    # Yield any remaining words in the last chunk
    if current_chunk_words:
        yield " ".join(current_chunk_words)


def extract_json_from_response_text(text_response: str) -> Optional[List[Dict]]:
    """
    Attempts to extract a JSON list from the LLM's generated text.
    Handles potential markdown code blocks or surrounding text.
    (Same logic as before)
    """
    if not text_response:
        print("Warning: LLM generated empty text response.")
        return None
    try:
        json_start = text_response.find('[')
        json_end = text_response.rfind(']')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = text_response[json_start : json_end + 1]
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                return parsed_json
            else:
                 print(f"Warning: Extracted JSON is not a list: {type(parsed_json)}")
                 return None
        else:
            try:
                parsed_json = json.loads(text_response.strip())
                if isinstance(parsed_json, list):
                    print("Directly parsed LLM response as JSON list.")
                    return parsed_json
                else:
                    print(f"Warning: Directly parsed response is not a list: {type(parsed_json)}")
                    return None
            except json.JSONDecodeError:
                 print("Warning: Could not find JSON list brackets and direct parsing failed.")
                 print(f"LLM Response Text was:\n---\n{text_response}\n---")
                 return None

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from extracted LLM text: {e}")
        print(f"Extracted Text was:\n---\n{json_str}\n---") # Use json_str if defined
        return None
    except Exception as e:
        print(f"Unexpected error during JSON extraction: {e}")
        return None

def _call_llm_api_for_chunk(text_chunk: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Internal function to call the LLM API for a single chunk of text.

    Args:
        text_chunk: The text chunk to process.

    Returns:
        A tuple containing:
        - list: The parsed JSON list from the LLM response if successful.
        - str: An error message string if the API call failed, otherwise None.
    """
    if not LLM_API_URL or not LLM_MODEL_NAME:
         return None, "LLM API URL or Model Name is not configured."

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    # Format the prompt using the template from settings
    prompt = settings.LLM_PROCESS_PROMPT_TEMPLATE.format(text_chunk=text_chunk)

    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "top_p": settings.LLM_TOP_P,
        "stream": False
    }

    print(f"  Sending chunk to LLM (approx {len(text_chunk.split())} words)...")
    # print(f"  Payload: {json.dumps(payload)}") # Uncomment for deep debugging

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=60) # Keep timeout reasonable per chunk
        response.raise_for_status()

        try:
            # Try parsing the JSON response
            llm_response_data = response.json()
        except json.JSONDecodeError as jde: # Specific catch for JSON parsing errors
            print(f"  Error: Could not decode JSON response from LLM server for chunk. JSONDecodeError: {jde}")
            raw_text = response.text if hasattr(response, 'text') else "N/A"
            print(f"  Raw LLM Response Text that failed parsing:\n---\n{raw_text}\n---")
            return None, f"Invalid JSON response received from LLM server: {jde}"
        except requests.exceptions.RequestException as rex: # Catch potential requests-level JSON issues (less common)
             print(f"  Error: RequestException during JSON decoding: {rex}")
             return None, f"RequestException during JSON decoding: {rex}"


        try:
            # Validate the structure before accessing deeply nested keys
            if not isinstance(llm_response_data, dict):
                print(f"  Error: LLM response is not a dictionary. Got: {type(llm_response_data)}")
                print(f"  LLM Response Data: {llm_response_data}")
                return None, "LLM server response was not a dictionary."

            if "choices" not in llm_response_data or not isinstance(llm_response_data["choices"], list) or not llm_response_data["choices"]:
                 print(f"  Error: 'choices' key missing, not a list, or empty in LLM response.")
                 print(f"  LLM Response Data: {llm_response_data}")
                 return None, "LLM server response missing 'choices' list or list is empty."

            first_choice = llm_response_data["choices"][0]
            if not isinstance(first_choice, dict) or "text" not in first_choice:
                 print(f"  Error: First choice is not a dict or 'text' key missing.")
                 print(f"  First Choice Data: {first_choice}")
                 return None, "LLM server response choice missing 'text' field."

            generated_text = first_choice["text"]
            if not isinstance(generated_text, str):
                 print(f"  Error: Expected 'text' field to be a string, but got {type(generated_text)}.")
                 print(f"  Generated Text Data: {generated_text}")
                 return None, "LLM server response 'text' field was not a string."

            generated_text = generated_text.strip()

        except (KeyError, IndexError, TypeError) as e: # Catch errors during structure traversal
            print(f"  Error: Could not extract text from LLM response structure for chunk: {e}")
            print(f"  LLM Response Data: {llm_response_data}") # Log the structure that caused the error
            return None, f"LLM server response did not match expected format (choices[0].text): {e}"

        # Now parse the JSON *within* the generated text
        json_list = extract_json_from_response_text(generated_text)

        if json_list is None:
             # extract_json_from_response_text already prints details
             return None, "Failed to parse structured JSON list from LLM's generated text for chunk."

        return json_list, None # Success for this chunk

    except requests.exceptions.RequestException as e:
        error_message = f"Error communicating with LLM server for chunk: {e}"
        print(f"  Error: {error_message}")
        # Log details if available
        if hasattr(e, 'response') and e.response is not None:
             print(f"  LLM Server Response Status: {e.response.status_code}")
             try:
                 print(f"  LLM Server Response Body: {e.response.json()}")
             except json.JSONDecodeError:
                 print(f"  LLM Server Response Body (non-JSON): {e.response.text}")
        return None, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred processing chunk: {e}"
        print(f"  Error: {error_message}")
        return None, error_message


# --- Main Processing Function ---

def get_sentence_pairs_and_add_to_anki(
    full_text: str,
    deck_name: str,
    model_name: str,
    tags: List[str]
) -> Tuple[AnkiAddResult, Optional[str]]:
    """
    Splits text into chunks, calls LLM for each, processes the result,
    formats notes, and adds them to Anki immediately per chunk.
    Aggregates Anki results and processing errors.

    Args:
        full_text: The complete input text (e.g., transcript).
        deck_name: The target Anki deck name.
        model_name: The Anki note model name.
        tags: A list of tags to add to the Anki notes.

    Returns:
        A tuple containing:
        - AnkiAddResult: Aggregated results of notes added/failed across all chunks.
        - str: An aggregated error message string containing LLM processing errors
               and Anki connection errors, or None if no errors occurred.
    """
    if not LLM_API_URL or not LLM_MODEL_NAME:
        return AnkiAddResult(notes_added=0, notes_failed=0), "LLM API URL or Model Name is not configured."
    if not full_text:
        return AnkiAddResult(notes_added=0, notes_failed=0), "Input text is empty."

    aggregated_anki_result = AnkiAddResult(notes_added=0, notes_failed=0)
    all_errors: List[str] = []
    total_chunks = 0

    print(f"\nSplitting text into chunks of ~{CHUNK_WORD_COUNT} words and processing...")

    for i, chunk in enumerate(split_into_chunks(full_text, CHUNK_WORD_COUNT)):
        total_chunks += 1
        print(f"\nProcessing Chunk {i+1}...")
        json_list, llm_error = _call_llm_api_for_chunk(chunk)

        if llm_error:
            error_msg = f"Chunk {i+1} LLM Error: {llm_error}"
            print(f"  -> {error_msg}")
            all_errors.append(error_msg)
            continue # Skip to the next chunk

        if json_list is None:
            print(f"  -> Warning: Chunk {i+1} processed but resulted in no valid JSON data.")
            continue # Skip to next chunk

        # Process pairs from this chunk
        chunk_pairs: List[ProcessedSentencePair] = []
        for item_index, item in enumerate(json_list): # Add index for logging
            if isinstance(item, dict) and 'german' in item and 'english' in item:
                try:
                    german_text = item['german'] # Access key
                    english_text = item['english'] # Access key

                    # Check if the retrieved values are strings before creating Pydantic model
                    if not isinstance(german_text, str):
                        print(f"  Warning: Skipping item {item_index} in chunk {i+1} due to non-string value for 'german': {german_text}")
                        continue
                    if not isinstance(english_text, str):
                        print(f"  Warning: Skipping item {item_index} in chunk {i+1} due to non-string value for 'english': {english_text}")
                        continue

                    # Try creating the Pydantic model (validation happens here)
                    chunk_pairs.append(ProcessedSentencePair(german=german_text, english=english_text))

                except KeyError as ke:
                    print(f"  Error: KeyError accessing german/english in item {item_index} of chunk {i+1}: {item} - Error: {ke}")
                except Exception as e:
                     print(f"  Warning: Skipping item {item_index} in chunk {i+1} due to validation/other error: {item} - Error: {e}")
            else:
                print(f"  Warning: Skipping invalid item format in chunk {i+1}, item {item_index}: {item}")

        if not chunk_pairs:
            print(f"  -> Successfully processed chunk {i+1}, but no valid sentence pairs were generated.")
            continue # Skip to next chunk

        print(f"  -> Successfully processed chunk {i+1}, generated {len(chunk_pairs)} pairs. Adding to Anki...")

        # Format notes for Anki for this chunk
        anki_notes_chunk: List[AnkiNotePayload] = []
        for pair in chunk_pairs:
             try:
                 anki_note = AnkiNotePayload(
                     deckName=deck_name,
                     modelName=model_name,
                     fields={"Front": pair.german, "Back": pair.english},
                     tags=tags
                 )
                 anki_notes_chunk.append(anki_note)
             except Exception as format_e:
                 print(f"  Error formatting pair for Anki in chunk {i+1}: {pair} - Error: {format_e}")

        if not anki_notes_chunk:
             print(f"  -> No notes formatted for Anki from chunk {i+1} (check formatting errors).")
             continue

        # Add notes to Anki for this chunk
        anki_result_list, anki_connect_error_msg = add_notes_to_anki(anki_notes_chunk)

        if anki_connect_error_msg:
            error_msg = f"Chunk {i+1} AnkiConnect Error: {anki_connect_error_msg}"
            print(f"  -> {error_msg}")
            all_errors.append(error_msg)
            aggregated_anki_result.notes_failed += len(anki_notes_chunk)
        elif anki_result_list is not None:
            added_count = sum(1 for res in anki_result_list if res is not None)
            failed_count = len(anki_result_list) - added_count
            aggregated_anki_result.notes_added += added_count
            aggregated_anki_result.notes_failed += failed_count
            print(f"  -> Anki Add Result for Chunk {i+1}: Added={added_count}, Failed/Duplicates={failed_count}")
        else:
            error_msg = f"Chunk {i+1}: Unknown communication error with AnkiConnect during addNotes."
            print(f"  -> {error_msg}")
            all_errors.append(error_msg)
            aggregated_anki_result.notes_failed += len(anki_notes_chunk)

    print(f"\nFinished processing {total_chunks} chunks.")

    final_error_message = "; ".join(all_errors) if all_errors else None

    if final_error_message:
        print(f"Warning: Processing completed with errors: {final_error_message}")
    elif total_chunks == 0:
         print("Input text resulted in zero chunks.")
    elif aggregated_anki_result.notes_added == 0 and aggregated_anki_result.notes_failed == 0:
         print("Processing complete, but no notes were generated or added from any chunk.")
    else:
        print(f"Processing complete. Aggregated Anki Results: Added={aggregated_anki_result.notes_added}, Failed/Duplicates={aggregated_anki_result.notes_failed}")

    return aggregated_anki_result, final_error_message

