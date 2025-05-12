import requests
import json
import time # Import time for potential delays between requests
import os # Import os for environment variable access
# Import Dict from typing for compatibility with Python < 3.9
from typing import List, Dict, Optional, Tuple, Generator
import openai  # Add openai import

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
def split_into_chunks(text: str, word_limit: int, overlap: int = 10) -> Generator[str, None, None]:
    """
    Splits text into chunks with overlapping words to prevent sentence truncation.
    
    Args:
        text: The full text to split.
        word_limit: The number of words for each chunk.
        overlap: Number of words to overlap between consecutive chunks (default 10).
                This helps prevent sentences from being cut off at chunk boundaries.
                
    Yields:
        String chunks of the text with specified overlap.
    """
    if word_limit <= 0:
        stripped_text = text.strip()
        if stripped_text:  # Only yield if there's actual content
            yield stripped_text
        return

    words = text.split()
    if not words:
        return
    
    # Special case - if text is shorter than the limit, just yield it
    if len(words) <= word_limit:
        yield " ".join(words)
        return
        
    # For longer texts, create overlapping chunks
    start_index = 0
    
    while start_index < len(words):
        # Calculate the end index for this chunk (not exceeding word array length)
        end_index = min(start_index + word_limit, len(words))
        
        # Extract the chunk and yield it
        chunk = " ".join(words[start_index:end_index])
        yield chunk
        
        # For next iteration, move forward by (word_limit - overlap) words
        # This creates the desired overlap between chunks
        # But don't move backward if we're at the end
        if end_index == len(words):
            break
            
        start_index += max(1, word_limit - overlap)  # Ensure we always move forward


def extract_json_from_response_text(text: str) -> Optional[List[Dict]]:
    """
    Extract a JSON list from LLM response text, handling various formats.
    
    This function is simpler now that we're using the OpenAI client which returns cleaner JSON,
    but we keep it for backward compatibility and potential fallback scenarios.
    
    Args:
        text: The raw text response from the LLM.
        
    Returns:
        The parsed JSON list if found, or None if parsing failed.
    """
    if not text:
        print("  Error: Empty text received from LLM")
        return None
        
    # First try to parse the whole text as JSON
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, list):
            return parsed_json
        print(f"  Error: Parsed JSON is not a list but {type(parsed_json)}")
        return None
    except json.JSONDecodeError:
        # Text includes other content, try to extract JSON parts
        pass
    
    # Look for JSON between brackets
    import re
    json_pattern = r'\[(.*)\]'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        try:
            json_str = f"[{json_match.group(1)}]"
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                return parsed_json
        except json.JSONDecodeError:
            pass
    
    print(f"  Error: Could not extract valid JSON list from LLM response")
    print(f"  Raw response text: {text[:300]}...")  # Print first 300 chars
    return None

def _call_llm_api_for_chunk(text_chunk: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Internal function to call the LLM API for a single chunk of text.
    Uses the same approach as the working TextProcessor.process_german_text() 
    from notebooks/lm_mistral_server.py.

    Args:
        text_chunk: The text chunk to process.

    Returns:
        A tuple containing:
        - list: The parsed JSON list from the LLM response if successful.
        - str: An error message string if the API call failed, otherwise None.
    """
    global LLM_MODEL_NAME  # Properly declare the global variable
    
    if not LLM_API_URL or not LLM_MODEL_NAME:
         return None, "LLM API URL or Model Name is not configured."

    # Use a local variable for the model name to avoid global issues
    current_model = LLM_MODEL_NAME

    # Format the prompt using the template from settings
    prompt = settings.LLM_PROCESS_PROMPT_TEMPLATE.format(text_chunk=text_chunk)

    # Normalize API URL
    api_base_url = LLM_API_URL
    if api_base_url.endswith("/completions"):
        api_base_url = api_base_url[:-len("/completions")]
    elif api_base_url.endswith("/chat/completions"):
        api_base_url = api_base_url[:-len("/chat/completions")]
        
    if not api_base_url.endswith("/v1"):
        if api_base_url.endswith("/"): 
            api_base_url += "v1"
        else: 
            api_base_url += "/v1"

    # Create schema for structured output that uses "deutsch" field name
    output_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "deutsch": {
                    "type": "string",
                    "description": "The original German full sentence."
                },
                "english": {
                    "type": "string",
                    "description": "The translated English full sentence."
                }
            },
            "required": ["deutsch", "english"]
        },
        "description": "A list of translated full sentence pairs."
    }

    user_prompt = f"""Segment and translate this German transcript:

TRANSCRIPT:
{text_chunk}

RULES:
1. Find ALL LOGICAL BREAKS in the text (pauses, topic changes, sentence endings)
2. Create complete, meaningful sentences - as many as naturally exist
3. Each sentence should express one complete thought
4. Add capital letters and punctuation
5. Translate each sentence to English

EXAMPLE:
"das ist ein wichtiges Thema KI ist überall maschinelles Lernen verändert wie wir arbeiten neue Tools müssen verstanden werden"

GOOD OUTPUT:
[
  {{
    "deutsch": "Das ist ein wichtiges Thema.",
    "english": "This is an important topic."
  }},
  {{
    "deutsch": "KI ist überall.",
    "english": "AI is everywhere."
  }},
  {{
    "deutsch": "Maschinelles Lernen verändert, wie wir arbeiten.",
    "english": "Machine learning changes how we work."
  }},
  {{
    "deutsch": "Neue Tools müssen verstanden werden.",
    "english": "New tools need to be understood."
  }}
]

BAD OUTPUT (don't do this - fragments, not complete thoughts):
[
  {{
    "deutsch": "Das ist ein wichtiges.",
    "english": "This is an important."
  }},
  {{
    "deutsch": "KI ist.",
    "english": "AI is."
  }}
]

BAD OUTPUT (don't do this - everything in one sentence):
[
  {{
    "deutsch": "Das ist ein wichtiges Thema KI ist überall maschinelles Lernen verändert wie wir arbeiten neue Tools müssen verstanden werden.",
    "english": "This is an important topic AI is everywhere machine learning changes how we work new tools need to be understood."
  }}
]

Extract ALL natural sentences, each expressing ONE complete thought.
"""

    print(f"  Sending chunk to LLM (approx {len(text_chunk.split())} words)...")

    try:
        # Create OpenAI client with our API base URL
        client = openai.OpenAI(base_url=api_base_url, api_key=LLM_API_KEY or "dummy_key")
        
        # Add retry logic for reliability
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": "You reconstruct complete German sentences from transcript lines, then translate them to English. Respond in JSON format."},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    top_p=settings.LLM_TOP_P,
                    seed=42,  # For more consistent results
                    extra_body={
                        "guided_json": output_schema,
                        "enable_thinking": False
                    }
                )
                
                generated_text = completion.choices[0].message.content
                
                # Parse the JSON directly since OpenAI client returns properly structured response
                try:
                    # First try to parse the entire content
                    json_list = json.loads(generated_text)
                    
                    # Check if we got an array directly
                    if isinstance(json_list, list):
                        return json_list, None  # Success!
                    
                    # If we got an object with a top-level array property, try to extract it
                    elif isinstance(json_list, dict):
                        # Look for any array values in the dict (sometimes models return {"pairs": [...]} etc.)
                        for key, value in json_list.items():
                            if isinstance(value, list) and len(value) > 0:
                                print(f"  Found array under key '{key}' instead of direct list")
                                return value, None  # Success with array from dict
                        
                        error_msg = f"LLM returned JSON object but no array field found: {list(json_list.keys())}"
                        print(f"  Error: {error_msg}")
                    else:
                        error_msg = f"LLM returned non-list JSON: {type(json_list)}"
                        print(f"  Error: {error_msg}")
                    # Fall through to retry
                except json.JSONDecodeError as jde:
                    error_msg = f"Could not parse JSON from LLM response: {jde}"
                    print(f"  Error: {error_msg}")
                    print(f"  Raw text: {generated_text[:200]}...")
                    # Fall through to retry
                    
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    
            except Exception as e:
                error_str = str(e)
                if ("The model" in error_str and "does not exist" in error_str) and attempt < max_retries - 1:
                    # Model not found error - try with known working model
                    fallback_model = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
                    print(f"  Model '{current_model}' not found. Trying with fallback model '{fallback_model}'")
                    current_model = fallback_model
                    LLM_MODEL_NAME = fallback_model  # Update the global for future chunks
                    time.sleep(retry_delay)
                elif attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed with error: {e}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

        # If we got here, all retries failed
        return None, "All retries failed to get valid JSON response from LLM"

    except openai.APIConnectionError as e:
        error_message = f"CONNECTION ERROR: {e}. Check URL and if vLLM server is running."
        print(f"  Error: {error_message}")
        return None, error_message
    except json.JSONDecodeError as e:
        error_message = f"JSON PARSING ERROR: {e}. Model output might not be valid JSON."
        print(f"  Error: {error_message}")
        return None, error_message
    except Exception as e:
        error_message = f"UNEXPECTED ERROR: {e}"
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
            try:
                # Check for either "german" or "deutsch" field
                german_text = None
                if "german" in item:
                    german_text = item["german"]
                elif "deutsch" in item:
                    german_text = item["deutsch"]
                else:
                    print(f"  Warning: Item {item_index} has neither 'german' nor 'deutsch' field: {item}")
                    continue
                    
                english_text = item.get("english")
                
                if not english_text:
                    print(f"  Warning: Item {item_index} missing 'english' field: {item}")
                    continue

                # Check if the retrieved values are strings before creating Pydantic model
                if not isinstance(german_text, str):
                    print(f"  Warning: Skipping item {item_index} in chunk {i+1} due to non-string value for german text: {german_text}")
                    continue
                if not isinstance(english_text, str):
                    print(f"  Warning: Skipping item {item_index} in chunk {i+1} due to non-string value for 'english': {english_text}")
                    continue

                # Try creating the Pydantic model (validation happens here)
                chunk_pairs.append(ProcessedSentencePair(german=german_text, english=english_text))

            except KeyError as ke:
                print(f"  Error: KeyError accessing fields in item {item_index} of chunk {i+1}: {item} - Error: {ke}")
            except Exception as e:
                print(f"  Warning: Skipping item {item_index} in chunk {i+1} due to validation/other error: {item} - Error: {e}")

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

