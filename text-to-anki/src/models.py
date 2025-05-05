from pydantic import BaseModel, Field, HttpUrl
# Import Dict from typing for compatibility with Python < 3.9
from typing import List, Optional, Dict

# --- API Request Models ---

class ProcessTextRequest(BaseModel):
    """Request model for the /api/process endpoint (direct text input)."""
    text: str = Field(..., description="The chunk of text to process.")
    # Optional overrides for configuration per request
    deck_name: Optional[str] = Field(None, description="Override the default Anki deck name.")
    model_name: Optional[str] = Field(None, description="Override the default Anki model name.")
    tags: Optional[List[str]] = Field(None, description="Override the default Anki tags.")

class ProcessYoutubeRequest(BaseModel):
    """Request model for the /api/youtube/process endpoint."""
    youtube_url: HttpUrl = Field(..., description="The URL of the YouTube video.")
    preferred_language: str = Field('en', description="Preferred language code for the transcript (e.g., 'en', 'de').")
    # Optional overrides for configuration per request
    deck_name: Optional[str] = Field(None, description="Override the default Anki deck name.")
    model_name: Optional[str] = Field(None, description="Override the default Anki model name.")
    tags: Optional[List[str]] = Field(None, description="Override the default Anki tags.")


# --- API Response Models ---

class ProcessedSentencePair(BaseModel):
    """Represents a single generated German/English sentence pair."""
    german: str
    english: str

class AnkiAddResult(BaseModel):
    """Details about the result of adding notes to Anki."""
    notes_added: int
    notes_failed: int # Includes duplicates if allowDuplicate=false
    anki_connect_error: Optional[str] = None # Error message from AnkiConnect if any

class BaseProcessResponse(BaseModel):
    """Base structure for processing responses."""
    message: str
    processed_pairs: List[ProcessedSentencePair] = []
    anki_result: Optional[AnkiAddResult] = None
    error: Optional[str] = None # For general processing, LLM, or YouTube errors

class ProcessTextResponse(BaseProcessResponse):
    """Response model for the /api/process endpoint."""
    original_text: str

class ProcessYoutubeResponse(BaseProcessResponse):
    """Response model for the /api/youtube/process endpoint."""
    youtube_url: HttpUrl
    video_id: Optional[str] = None
    transcript_language_fetched: Optional[str] = None # e.g., 'en', 'de'


# --- Internal Models (used between modules) ---

class AnkiNotePayload(BaseModel):
    """Structure for a single note to be sent to AnkiConnect."""
    deckName: str
    modelName: str
    # Use typing.Dict for Python < 3.9 compatibility
    fields: Dict[str, str] # e.g., {"Front": "...", "Back": "..."}
    options: dict = { # Default options, can be overridden
        "allowDuplicate": False,
        "duplicateScope": "deck",
        "duplicateScopeOptions": {
             "deckName": None, # Will be set dynamically
             "checkChildren": False,
             "checkAllModels": False
         }
    }
    tags: List[str] = []
