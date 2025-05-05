import requests
import json
# Import Dict from typing for compatibility with Python < 3.9
from typing import List, Dict, Optional, Tuple

# Use absolute import from the package root (src)
from .config import settings
from .models import AnkiNotePayload

ANKICONNECT_URL = settings.ANKICONNECT_URL

# --- Private Helper ---
def _invoke_anki_connect(action: str, **params) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Internal helper to send requests to the AnkiConnect API and handle common errors.
    Includes logging of request and response.

    Args:
        action: The AnkiConnect action to perform.
        **params: The parameters for the action.

    Returns:
        A tuple containing:
        - dict: The JSON response dictionary from AnkiConnect if successful and JSON.
        - str: An error message string if an error occurred, otherwise None.
    """
    # Construct the final payload to be sent
    request_payload = {"action": action, "version": 6, "params": params}

    # --- LOGGING: Print the exact payload being sent ---
    try:
        print(f"\n--- Sending Request to AnkiConnect ---")
        print(f"URL: {ANKICONNECT_URL}")
        print(f"Action: {action}")
        # Use indent for readability, ensure_ascii=False for non-ASCII chars
        print(f"Payload:\n{json.dumps(request_payload, indent=2, ensure_ascii=False)}")
        print("---------------------------------------")
    except Exception as log_e:
        print(f"Error trying to print AnkiConnect request payload: {log_e}")
    # --- END LOGGING ---

    error_message = None
    response_json = None
    response_status = None
    response_text = None

    try:
        response = requests.post(ANKICONNECT_URL, json=request_payload, timeout=15)
        response_status = response.status_code
        response_text = response.text # Store raw text for logging in case of JSON error

        # --- LOGGING: Print received response status and body ---
        print(f"\n--- Received Response from AnkiConnect ---")
        print(f"Status Code: {response_status}")
        print(f"Raw Response Body:\n{response_text}")
        print("-----------------------------------------")
        # --- END LOGGING ---

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = response.json() # Try parsing JSON *after* logging raw text

        # Check for application-level errors within the JSON response
        if response_json.get('error') is not None:
            error_message = f"AnkiConnect API Error in Response: {response_json['error']}"
            print(f"Error Detail: {error_message}") # Log the specific Anki error
            response_json = None # Clear response if there's an Anki error

    except requests.exceptions.Timeout:
        error_message = f"Timeout connecting to AnkiConnect at {ANKICONNECT_URL}"
        print(f"Error Detail: {error_message}")
    except requests.exceptions.ConnectionError:
        error_message = f"Connection refused by AnkiConnect at {ANKICONNECT_URL}. Is Anki running?"
        print(f"Error Detail: {error_message}")
    except requests.exceptions.RequestException as e:
        # Includes HTTPError from raise_for_status()
        error_message = f"Connection/HTTP error to AnkiConnect: {e}"
        # Log details if response was received before error
        if response_status is not None:
             error_message += f" (Status Code: {response_status})"
        print(f"Error Detail: {error_message}")
    except json.JSONDecodeError:
        # This happens if status code was OK (e.g., 200) but body wasn't valid JSON
        error_message = "Invalid JSON received from AnkiConnect despite OK status."
        print(f"Error Detail: {error_message}") # Raw body was already logged
    except Exception as e:
        error_message = f"An unexpected error occurred during AnkiConnect communication: {e}"
        print(f"Error Detail: {error_message}")

    return response_json, error_message

# --- Public Functions ---

def add_notes_to_anki(notes: List[AnkiNotePayload]) -> Tuple[Optional[List[Optional[int]]], Optional[str]]:
    """
    Adds multiple notes to Anki using the 'addNotes' action.

    Args:
        notes: A list of AnkiNotePayload objects representing the notes to add.

    Returns:
        A tuple containing:
        - list: The result array from AnkiConnect (list of note IDs or nulls).
        - str: An error message string if the API call failed, otherwise None.
    """
    if not notes:
        return [], "No notes provided to add." # Return empty list and message

    # Convert Pydantic models to dictionaries for the JSON payload
    # Also set the deckName in duplicateScopeOptions dynamically
    notes_dict_list = []
    for note in notes:
        # Use model_dump for Pydantic v2+, exclude unset to avoid sending None options unless explicitly set
        note_dict = note.model_dump(exclude_unset=True)
        if "options" not in note_dict:
             note_dict["options"] = {}
        if "duplicateScopeOptions" not in note_dict["options"]:
             note_dict["options"]["duplicateScopeOptions"] = {}

        note_dict["options"]["duplicateScopeOptions"]["deckName"] = note.deckName
        if "allowDuplicate" not in note_dict["options"]:
             note_dict["options"]["allowDuplicate"] = False # Default if not set

        notes_dict_list.append(note_dict)


    # Call the helper which now includes detailed logging
    response_data, error = _invoke_anki_connect('addNotes', notes=notes_dict_list)

    if error:
        # Error details already logged by _invoke_anki_connect
        return None, error # Communication or Anki API error occurred

    if response_data is None:
         # Should be caught by error handling, but as a safeguard
         return None, "Unknown error occurred during AnkiConnect 'addNotes' call (response was None)."

    # Result should be a list of note IDs (integers) or nulls
    result_list = response_data.get('result')
    if not isinstance(result_list, list):
         error_message = f"Unexpected result format from AnkiConnect 'addNotes': {result_list}"
         print(error_message)
         return None, error_message # Return the specific format error

    # Success case
    return result_list, None


def check_connection() -> bool:
    """Checks if AnkiConnect is reachable by fetching its version."""
    print(f"Checking connection to AnkiConnect at {ANKICONNECT_URL}...")
    response, error = _invoke_anki_connect('version') # Uses the logging helper
    if error is None and response and response.get('result') is not None:
        print(f"AnkiConnect connection successful (Version: {response.get('result')}).")
        return True
    else:
        # Error message already printed by _invoke_anki_connect
        print("AnkiConnect connection failed.")
        return False
