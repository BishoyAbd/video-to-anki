import os
from dotenv import load_dotenv
from pathlib import Path
# Import Optional for type hinting compatibility with Python < 3.10
from typing import List, Optional

from  .prompts import LLM_PROCESS_PROMPT_TEMPLATE
# Determine the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
dotenv_path = BASE_DIR / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from: {dotenv_path}")
else:
    print("No .env file found, relying on system environment variables.")


class Settings:
    """Loads and stores application settings from environment variables."""
    # AnkiConnect Settings
    ANKICONNECT_URL: str = os.getenv("ANKICONNECT_URL", "http://localhost:8765")
    ANKI_DECK_NAME: str = os.getenv("ANKI_DECK_NAME", "Test Deutsch")
    ANKI_MODEL_NAME: str = os.getenv("ANKI_MODEL_NAME", "Basic (and reversed card)")
    ANKI_TAGS: List[str] = [
        tag.strip() for tag in os.getenv("ANKI_TAGS", "").split(',') if tag.strip()
    ]

    # LLM Settings (Your Custom Server - OpenAI compatible)
    LLM_API_URL: Optional[str] = os.getenv("LLM_API_URL")
    LLM_MODEL_NAME: Optional[str] = os.getenv("LLM_MODEL_NAME") # Model identifier for the API
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") # Optional API key

    # Default Prompt - expects {text_chunk} to be formatted in
    # Import the prompt template from prompts.py
    # Make sure to add this import at the top of the config.py file:
    # from .prompts import PROCESS_PROMPT_TEMPLATE
    LLM_PROCESS_PROMPT_TEMPLATE: str = LLM_PROCESS_PROMPT_TEMPLATE

    # LLM Generation Parameters
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1024))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 0.9))

    # Chunking Settings
    CHUNK_WORD_COUNT: int = int(os.getenv("CHUNK_WORD_COUNT", 30)) # Added chunk size setting


    # --- Validations ---
    if not LLM_API_URL:
        print("Warning: LLM_API_URL environment variable is not set!")
    if not LLM_MODEL_NAME:
         print("Warning: LLM_MODEL_NAME environment variable is not set!")
    if CHUNK_WORD_COUNT <= 0:
        print("Warning: CHUNK_WORD_COUNT must be positive, defaulting to 30.")
        CHUNK_WORD_COUNT = 30


# Create a single instance of settings to be imported elsewhere
settings = Settings()
