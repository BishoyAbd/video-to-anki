import json
import requests

# --- Configuration ---
ANKICONNECT_URL = "http://localhost:8765"
DECK_NAME = "Tech Deutsch"
MODEL_NAME = "Basic (and reversed card)" # Standard Anki type for reverse cards
GERMAN_FIELD = "Front" # Corresponds to the 'Front' field in the Basic model
ENGLISH_FIELD = "Back"  # Corresponds to the 'Back' field in the Basic model
TAGS = ["LLM_Business", "Tech_Deutsch"] # Optional tags to add

# --- Content: German sentences about LLM business side & English translation ---
sentence_pairs = [
    {
        "de": "Die Implementierung von LLMs kann die Kundeninteraktion automatisieren und die Betriebskosten senken.",
        "en": "Implementing LLMs can automate customer interaction and reduce operational costs."
    },
    {
        "de": "Unternehmen nutzen LLMs zur Analyse von Markttrends und zur Verbesserung von Geschäftsstrategien.",
        "en": "Companies use LLMs to analyze market trends and improve business strategies."
    },
    {
        "de": "Die Monetarisierung von LLM-basierten Diensten ist ein wachsender Bereich für Technologieunternehmen.",
        "en": "Monetizing LLM-based services is a growing area for technology companies."
    },
    {
        "de": "Die Skalierbarkeit von LLM-Lösungen ist entscheidend für den Einsatz in großen Unternehmen.",
        "en": "The scalability of LLM solutions is crucial for deployment in large enterprises."
    },
    {
        "de": "Ethische Überlegungen und Datenschutz sind zentrale Herausforderungen bei der geschäftlichen Nutzung von LLMs.",
        "en": "Ethical considerations and data privacy are key challenges in the business use of LLMs."
    },
    {
        "de": "Der ROI für LLM-Investitionen muss sorgfältig bewertet werden.",
        "en": "The ROI for LLM investments needs to be carefully evaluated."
    },
     {
        "de": "LLMs ermöglichen personalisierte Marketingkampagnen in großem Maßstab.",
        "en": "LLMs enable personalized marketing campaigns at scale."
    }
]

# --- Helper Function for AnkiConnect API Calls ---
def invoke(action, **params):
    """Sends a request to AnkiConnect"""
    payload = {"action": action, "version": 6, "params": params}
    try:
        response = requests.post(ANKICONNECT_URL, json=payload, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()
        if 'error' in result and result['error'] is not None:
            print(f"AnkiConnect API Error: {result['error']}")
            return None
        return result.get('result')
    except requests.exceptions.RequestException as e:
        print(f"Connection Error to AnkiConnect ({ANKICONNECT_URL}): {e}")
        print("Is Anki running with AnkiConnect installed and enabled?")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from AnkiConnect.")
        return None


# --- Main Logic ---

# 1. Check AnkiConnect connection
print("Checking AnkiConnect connection...")
version = invoke('version')
if version is None:
    print("Exiting script.")
    exit()
print(f"AnkiConnect version: {version}. Connection successful.")

# 2. Prepare notes for bulk adding
notes_to_add = []
print(f"\nPreparing {len(sentence_pairs)} notes for deck '{DECK_NAME}' using model '{MODEL_NAME}'...")
for pair in sentence_pairs:
    note = {
        "deckName": DECK_NAME,
        "modelName": MODEL_NAME,
        "fields": {
            GERMAN_FIELD: pair["de"],
            ENGLISH_FIELD: pair["en"]
        },
        "options": {
            "allowDuplicate": False, # Set to True if you want duplicates, otherwise it prevents adding identical cards
            "duplicateScope": "deck",
             "duplicateScopeOptions": {
                "deckName": DECK_NAME,
                "checkChildren": False,
                "checkAllModels": False
            }
        },
        "tags": TAGS
    }
    notes_to_add.append(note)

# 3. Add notes to Anki via AnkiConnect
if notes_to_add:
    print(f"\nAttempting to add {len(notes_to_add)} notes to Anki...")
    result = invoke('addNotes', notes=notes_to_add)

    if result is not None:
        added_count = sum(1 for note_id in result if note_id is not None)
        failed_count = len(result) - added_count
        print(f"\nFinished.")
        print(f"  Successfully added: {added_count} notes.")
        if failed_count > 0:
             print(f"  Failed/Duplicates: {failed_count} notes (Note: Failures can be due to duplicates if allowDuplicate=false).")
             # You could potentially analyze 'result' which contains note IDs or null for failures.
             # For example: failed_indices = [i for i, note_id in enumerate(result) if note_id is None]
        print(f"\nPlease check the '{DECK_NAME}' deck in Anki.")
    else:
        print("\nFailed to add notes. AnkiConnect returned an error or no result.")
else:
    print("\nNo notes were prepared to be added.")