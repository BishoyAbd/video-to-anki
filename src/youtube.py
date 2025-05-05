import sys
from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple, List

# Attempt to import the library, handle if missing
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print("Warning: 'youtube-transcript-api' library not found. YouTube processing will be disabled.")
    # Define dummy classes/exceptions if needed for type hinting, though not strictly necessary here
    class YouTubeTranscriptApi: pass
    class NoTranscriptFound(Exception): pass
    class TranscriptsDisabled(Exception): pass
    class VideoUnavailable(Exception): pass
    YOUTUBE_API_AVAILABLE = False


def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts the YouTube video ID from various URL formats.

    Args:
        url: The YouTube video URL.

    Returns:
        The extracted video ID string, or None if parsing fails.
    """
    if not url:
        return None

    parsed_url = urlparse(url)
    video_id = None

    # Handle www.youtube.com, m.youtube.com, youtube.com
    if parsed_url.netloc.endswith("youtube.com"):
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            video_id = query_params.get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
             video_id = parsed_url.path.split("/embed/")[1].split("?")[0]
        elif parsed_url.path.startswith("/shorts/"):
            video_id = parsed_url.path.split("/shorts/")[1].split("?")[0]
        elif parsed_url.path.startswith("/live/"):
            video_id = parsed_url.path.split("/live/")[1].split("?")[0]

    # Handle youtu.be
    elif parsed_url.netloc == "youtu.be":
        video_id = parsed_url.path[1:].split("?")[0] # Remove leading '/'

    if not video_id:
        print(f"Warning: Could not parse video ID from URL structure: {url}")

    return video_id


def format_transcript(transcript_data_object) -> str:
    """
    Formats the fetched transcript data (list/iterable of snippets)
    into a single text string.

    Args:
        transcript_data_object: The iterable object returned by transcript.fetch().

    Returns:
        A single string containing the concatenated transcript text.
        Returns an error string if formatting fails.
    """
    lines = []
    try:
        # Ensure the main object is iterable
        iter(transcript_data_object)

        for i, piece in enumerate(transcript_data_object):
            # Check if the yielded 'piece' has a 'text' attribute
            if hasattr(piece, 'text'):
                lines.append(str(piece.text))
            else:
                print(f"Warning: Transcript piece #{i} does not have expected 'text' attribute: {type(piece)} - {piece}")
                lines.append("[Missing Text Attribute]") # Placeholder for this piece

    except TypeError:
        print(f"Error: The fetched transcript data (type: {type(transcript_data_object)}) is not iterable.")
        return "[Error Formatting Transcript - Not Iterable]"
    except Exception as e:
        print(f"An unexpected error occurred during transcript formatting: {e}")
        return "[Error Formatting Transcript - Unexpected]"

    return "\n".join(lines)


def get_transcript_text(video_id: str, preferred_language: str = 'en') -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches and formats the transcript for a given video ID.
    Tries the preferred language first, then falls back to the first available.

    Args:
        video_id: The YouTube video ID.
        preferred_language: The preferred language code (e.g., 'en', 'de').

    Returns:
        A tuple containing:
        - str: The formatted transcript text if successful.
        - str: An error message string if fetching/formatting failed, otherwise None.
    """
    if not YOUTUBE_API_AVAILABLE:
        return None, "youtube-transcript-api library is not installed."

    try:
        print(f"Fetching transcript list for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to find the preferred language transcript
        transcript_to_fetch = None
        try:
            transcript_to_fetch = transcript_list.find_transcript([preferred_language])
            print(f"Found preferred language transcript: {transcript_to_fetch.language} ({transcript_to_fetch.language_code})")
        except NoTranscriptFound:
            print(f"Preferred language '{preferred_language}' not found. Checking available languages.")
            # If preferred not found, try getting the first available one
            available_langs = [t.language_code for t in transcript_list]
            if not available_langs:
                 return None, f"No transcripts found at all for video ID: {video_id}."
            print(f"Available languages: {available_langs}. Attempting to fetch first one: {available_langs[0]}")
            try:
                 # Fetch the first transcript object from the list
                 first_transcript_code = available_langs[0]
                 transcript_to_fetch = transcript_list.find_transcript([first_transcript_code])
            except NoTranscriptFound: # Should not happen if available_langs is populated, but safeguard
                 return None, f"Could not fetch any available transcript for video ID: {video_id}."


        # Fetch and format the chosen transcript
        print(f"Fetching transcript data for language: {transcript_to_fetch.language_code}")
        transcript_data = transcript_to_fetch.fetch()
        formatted_text = format_transcript(transcript_data)

        if formatted_text.startswith("[Error Formatting Transcript"):
            # Error details already printed by format_transcript
            return None, "Failed to format the fetched transcript data."

        print("Transcript fetched and formatted successfully.")
        return formatted_text, None

    except TranscriptsDisabled:
        error_msg = f"Transcripts are disabled for video ID: {video_id}"
        print(f"Error: {error_msg}")
        return None, error_msg
    except VideoUnavailable:
        error_msg = f"Video {video_id} is unavailable (private, deleted, or region-restricted)."
        print(f"Error: {error_msg}")
        return None, error_msg
    except NoTranscriptFound:
         # This might catch cases where list_transcripts itself finds nothing initially
         error_msg = f"No transcripts found for video ID: {video_id} after checking."
         print(f"Error: {error_msg}")
         return None, error_msg
    except Exception as e:
        # Catch-all for other unexpected errors (network issues, API changes etc.)
        error_msg = f"An unexpected error occurred fetching transcript for {video_id}: {e}"
        print(f"Error: {error_msg}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed debug info
        return None, error_msg

