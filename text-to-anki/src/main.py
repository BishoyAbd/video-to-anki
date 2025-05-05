from fastapi import FastAPI, HTTPException, Body, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
# Import necessary types for hinting
from typing import List, Optional, Tuple
import json # Import json for logging dicts
import traceback # Import traceback for detailed stack trace

# Import project modules using absolute paths from src
from .config import settings
from .models import (
    ProcessTextRequest, ProcessTextResponse,
    ProcessYoutubeRequest, ProcessYoutubeResponse, # Added YouTube models
    AnkiNotePayload, AnkiAddResult, ProcessedSentencePair
)
# Import the renamed processing function
from .processing import get_sentence_pairs_and_add_to_anki
from .anki import add_notes_to_anki, check_connection # Keep check_connection
from .youtube import extract_video_id, get_transcript_text # Added YouTube import

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Text-to-Anki Service",
    description="Processes text chunks or YouTube transcripts using an external LLM and adds notes via AnkiConnect.",
    version="0.4.0" # Bump version for workflow change
)

# --- Helper Function for common processing steps ---
# This helper is now much simpler, mainly determining config and calling the processing function
def _initiate_processing_and_adding(
    text_to_process: str,
    deck_name_override: Optional[str],
    model_name_override: Optional[str],
    tags_override: Optional[List[str]]
) -> Tuple[Optional[AnkiAddResult], Optional[str]]:
    """
    Determines configuration and calls the main processing function
    which handles LLM calls and Anki adding per chunk.

    Returns:
        Tuple (aggregated_anki_result, combined_error_message)
    """
    # Determine final Anki config
    deck = deck_name_override or settings.ANKI_DECK_NAME
    model = model_name_override or settings.ANKI_MODEL_NAME
    tags = tags_override or settings.ANKI_TAGS

    print("Calling processing and adding function...") # DEBUG

    # Call the function that now handles both LLM and Anki interaction per chunk
    aggregated_anki_result, combined_error_message = get_sentence_pairs_and_add_to_anki(
        full_text=text_to_process,
        deck_name=deck,
        model_name=model,
        tags=tags
    )

    # --- DEBUGGING for helper return values ---
    print("\n--- Processing Helper Return Values ---")
    print(f"Type of aggregated_anki_result: {type(aggregated_anki_result)}")
    if aggregated_anki_result:
        try:
            print(f"Aggregated Anki Result: {json.dumps(aggregated_anki_result.model_dump(), indent=2)}")
        except Exception as log_e:
            print(f"Error logging anki_result content: {log_e}")
    else:
        print("aggregated_anki_result is None (unexpected)") # Should always return an AnkiAddResult object
    print(f"combined_error_message: {combined_error_message}")
    print("---------------------------------------")
    # --- END DEBUGGING ---

    # Return the results directly from the processing function
    return aggregated_anki_result, combined_error_message


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    return """
    <html>
        <head><title>Text-to-Anki Service</title></head>
        <body>
            <h1>Text-to-Anki Service is Running</h1>
            <p>Use the <code>POST /api/process</code> endpoint for direct text input.</p>
            <p>Use the <code>POST /api/youtube/process</code> endpoint for YouTube URLs.</p>
            <p>Check the <a href="/docs">API documentation (/docs)</a>.</p>
        </body>
    </html>
    """

@app.post("/api/process",
          response_model=ProcessTextResponse,
          summary="Process Direct Text and Add to Anki",
          description="Takes a text chunk, uses an LLM to generate German/English sentence pairs, and attempts to add them to Anki via AnkiConnect.",
          responses={
              status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "AnkiConnect or LLM service unreachable"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal processing error"},
          }
)
async def process_direct_text(
    request: ProcessTextRequest = Body(...)
):
    """
    Processes direct text input:
    1. Calls processing function which chunks text, calls LLM, and adds to Anki per chunk.
    2. Returns aggregated results.
    """
    # --- LOGGING: Log incoming request ---
    request_data = {}
    try:
        request_data = request.model_dump() # Get request data as dict
        print(f"\n--- Received Request (/api/process) ---")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))
        print("---------------------------------------")
    except Exception as log_e:
        print(f"Error logging request data: {log_e}")
    # --- END LOGGING ---

    response_content = None
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR # Default to internal error

    try:
        # Pre-check AnkiConnect
        if not check_connection():
            error_msg = "AnkiConnect is not reachable."
            # Create response directly here for early exit
            response_content = ProcessTextResponse(
                message="Processing failed: AnkiConnect check failed.",
                original_text=request.text,
                error=error_msg,
                # Ensure anki_result is included in the error response
                anki_result=AnkiAddResult(notes_added=0, notes_failed=0, anki_connect_error=error_msg)
            )
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            # Use return for early exit
            return JSONResponse(
                status_code=status_code,
                content=response_content.model_dump(exclude_none=True)
            )

        # Call the simplified helper which calls the main processing/adding function
        anki_result, combined_error = _initiate_processing_and_adding(
            text_to_process=request.text,
            deck_name_override=request.deck_name,
            model_name_override=request.model_name,
            tags_override=request.tags
        )
        # --- Explicitly log helper results before response creation ---
        print(f"--- Helper Results in /api/process ---")
        print(f"anki_result: {anki_result}")
        print(f"combined_error: {combined_error}")
        print(f"--------------------------------------")
        # --- End logging ---

        # Prepare final response based on helper results
        # Note: processed_pairs will be empty as they are not aggregated anymore
        response_content = ProcessTextResponse(
            message="Processing complete.", # Default success message
            original_text=request.text,
            processed_pairs=[], # Pairs are processed and added per chunk, not returned in aggregate
            anki_result=anki_result, # The aggregated result from the processing function
            error=combined_error # Combined errors from LLM and Anki steps
        )

        status_code = status.HTTP_200_OK
        # Adjust final message and status based on outcomes
        if combined_error:
            response_content.message = f"Processing completed with errors (see error field). Check Anki for results."
            # Decide if any error should change status code, e.g., if AnkiConnect was consistently failing
            if "AnkiConnect Error" in combined_error or "LLM Error" in combined_error:
                 # Keep 200 OK to indicate the request was handled, but report errors
                 pass
            else: # Potentially other internal errors
                 status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif anki_result and anki_result.notes_added == 0 and anki_result.notes_failed == 0:
             response_content.message = "Processing complete, but no notes were generated or added (e.g., empty input or no valid pairs from LLM)."
        elif anki_result: # Some notes were likely processed
             response_content.message = f"Processing completed. Aggregated Anki Results: Added={anki_result.notes_added}, Failed/Duplicates={anki_result.notes_failed}."

    except Exception as e:
        # Catch any unexpected errors during endpoint execution
        # --- DETAILED EXCEPTION LOGGING ---
        print(f"!!! UNEXPECTED ERROR in /api/process endpoint !!!")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Value (str): {str(e)}")
        print(f"Exception Representation (repr): {repr(e)}")
        print("--- Traceback ---")
        traceback.print_exc() # Print the full stack trace
        print("-----------------")
        # --- END DETAILED LOGGING ---

        req_text = request.text if request else "N/A"
        response_content = ProcessTextResponse(
            message="An internal server error occurred.",
            original_text=req_text,
            error=f"{type(e).__name__}: {str(e)}", # Include exception type in response error
            processed_pairs=[] # Ensure this is empty on error too
        )
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    finally:
        # --- LOGGING: Log final response ---
        try:
            print(f"\n--- Sending Response ({status_code}) from /api/process ---")
            if response_content:
                 print(json.dumps(response_content.model_dump(exclude_none=True), indent=2, ensure_ascii=False))
            else:
                 print("(No response content object created due to early error)")
            print("-----------------------------------------------------")
        except Exception as log_e:
            print(f"Error logging response data: {log_e}")
        # --- END LOGGING ---

    # Ensure response_content is not None before dumping
    final_content = response_content.model_dump(exclude_none=True) if response_content else {"message": "Error occurred before response generation.", "error": "Internal Server Error"}

    return JSONResponse(
        status_code=status_code,
        content=final_content
    )


@app.post("/api/youtube/process",
          response_model=ProcessYoutubeResponse,
          summary="Process YouTube Transcript and Add to Anki",
           description="Takes a YouTube URL, fetches the transcript, uses an LLM to generate sentence pairs, and attempts to add them to Anki.",
           responses={
              status.HTTP_404_NOT_FOUND: {"description": "Video ID not found or invalid URL / Transcript unavailable"},
              status.HTTP_400_BAD_REQUEST: {"description": "Invalid YouTube URL format"},
              status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "AnkiConnect, LLM, or YouTube service unreachable/unavailable"},
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal processing error"},
          }
)
async def process_youtube_url(
     request: ProcessYoutubeRequest = Body(...)
):
    """
    Processes a YouTube URL:
    1. Extracts Video ID.
    2. Fetches transcript text.
    3. Calls processing function which handles LLM calls and Anki adding per chunk.
    4. Returns aggregated results.
    """
    # --- LOGGING: Log incoming request ---
    request_data = {}
    try:
        request_data = request.model_dump()
        if 'youtube_url' in request_data:
             request_data['youtube_url'] = str(request_data['youtube_url'])
        print(f"\n--- Received Request (/api/youtube/process) ---")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))
        print("---------------------------------------------")
    except Exception as log_e:
        print(f"Error logging request data: {log_e}")
    # --- END LOGGING ---

    response_content = None
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR # Default
    video_id = None # Initialize video_id

    try:
        # --- 1. Extract Video ID ---
        video_id = extract_video_id(str(request.youtube_url))
        if not video_id:
            # Create response directly for early exit via HTTPException
            response_content = ProcessYoutubeResponse(
                    message="Invalid YouTube URL.",
                    youtube_url=request.youtube_url,
                    error="Could not extract Video ID from the provided URL.",
                    processed_pairs=[]
                )
            status_code=status.HTTP_400_BAD_REQUEST
            # Raise HTTPException which will be caught and handled
            raise HTTPException(status_code=status_code, detail="Invalid YouTube URL")

        print(f"Extracted Video ID: {video_id}")

        # --- Pre-check: AnkiConnect ---
        if not check_connection():
            error_msg = "AnkiConnect is not reachable."
            response_content = ProcessYoutubeResponse(
                    message="Processing failed: AnkiConnect check failed.",
                    youtube_url=request.youtube_url,
                    video_id=video_id,
                    error=error_msg,
                    anki_result=AnkiAddResult(notes_added=0, notes_failed=0, anki_connect_error=error_msg),
                    processed_pairs=[]
                )
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            raise HTTPException(status_code=status_code, detail=error_msg)

        # --- 2. Fetch Transcript ---
        transcript_text, transcript_error = get_transcript_text(video_id, request.preferred_language)

        if transcript_error:
            # Determine status code based on transcript error
            if "No transcript" in transcript_error or "Video unavailable" in transcript_error or "disabled" in transcript_error:
                 status_code = status.HTTP_404_NOT_FOUND
            elif "library is not installed" in transcript_error:
                 status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            else: # Other errors (e.g., network)
                 status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            # Create response content for the exception
            response_content = ProcessYoutubeResponse(
                    message="Failed to fetch YouTube transcript.",
                    youtube_url=request.youtube_url,
                    video_id=video_id,
                    error=transcript_error,
                    processed_pairs=[]
                )
            raise HTTPException(status_code=status_code, detail=transcript_error)

        if not transcript_text:
             error_msg = "Fetched transcript content was empty."
             response_content = ProcessYoutubeResponse(
                    message="Transcript found but was empty.",
                    youtube_url=request.youtube_url,
                    video_id=video_id,
                    error=error_msg,
                    processed_pairs=[]
                )
             status_code=status.HTTP_404_NOT_FOUND
             raise HTTPException(status_code=status_code, detail=error_msg)

        # --- 3. Initiate Processing & Adding ---
        anki_result, combined_error = _initiate_processing_and_adding(
            text_to_process=transcript_text,
            deck_name_override=request.deck_name,
            model_name_override=request.model_name,
            tags_override=request.tags
        )
        # --- Explicitly log helper results before response creation ---
        print(f"--- Helper Results in /api/youtube/process ---")
        print(f"anki_result: {anki_result}")
        print(f"combined_error: {combined_error}")
        print(f"---------------------------------------------")
        # --- End logging ---

        # --- 4. Prepare Final Response ---
        response_content = ProcessYoutubeResponse(
            message="Processing complete.", # Default
            youtube_url=request.youtube_url,
            video_id=video_id,
            transcript_language_fetched=request.preferred_language, # Keep this info
            processed_pairs=[], # Pairs are processed and added per chunk
            anki_result=anki_result,
            error=combined_error
        )

        status_code = status.HTTP_200_OK
        # Adjust final message and status based on outcomes
        if combined_error:
            response_content.message = f"Processing completed with errors (see error field). Check Anki for results."
            if "AnkiConnect Error" in combined_error or "LLM Error" in combined_error:
                 pass # Keep 200 OK
            else:
                 status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif anki_result and anki_result.notes_added == 0 and anki_result.notes_failed == 0:
             response_content.message = "Processing complete, but no notes were generated or added."
        elif anki_result:
             response_content.message = f"Processing completed. Aggregated Anki Results: Added={anki_result.notes_added}, Failed/Duplicates={anki_result.notes_failed}."

    except HTTPException as http_exc:
         # Error details already logged by FastAPI/Uvicorn usually
         print(f"HTTP Exception caught in /youtube/process: {http_exc.status_code} - {http_exc.detail}")
         # Ensure response_content is set from before the raise, status_code is http_exc.status_code
         status_code = http_exc.status_code
         # If response_content wasn't set before raising (shouldn't happen with current logic), create a basic one
         if not response_content:
              response_content = ProcessYoutubeResponse(
                   message=f"Processing failed with HTTP {status_code}.",
                   youtube_url=request.youtube_url,
                   video_id=video_id,
                   error=http_exc.detail,
                   processed_pairs=[]
              )
         pass # Proceed to finally block
    except Exception as e:
        # Catch any unexpected errors during endpoint execution
        # --- DETAILED EXCEPTION LOGGING ---
        print(f"!!! UNEXPECTED ERROR in /api/youtube/process endpoint !!!")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Value (str): {str(e)}")
        print(f"Exception Representation (repr): {repr(e)}")
        print("--- Traceback ---")
        traceback.print_exc() # Print the full stack trace
        print("-----------------")
        # --- END DETAILED LOGGING ---

        response_content = ProcessYoutubeResponse(
            message="An internal server error occurred.",
            youtube_url=request.youtube_url,
            video_id=video_id if video_id else None,
            error=f"{type(e).__name__}: {str(e)}", # Include exception type in response error
            processed_pairs=[]
        )
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    finally:
        # --- LOGGING: Log final response ---
        final_response_dict = {}
        try:
            print(f"\n--- Sending Response ({status_code}) from /api/youtube/process ---")
            if response_content:
                 final_response_dict = response_content.model_dump(exclude_none=True)
                 if "youtube_url" in final_response_dict:
                      final_response_dict["youtube_url"] = str(final_response_dict["youtube_url"])
                 print(json.dumps(final_response_dict, indent=2, ensure_ascii=False))
            else:
                 print("(No response content object created due to early error)")
                 # Ensure we have a basic error structure if response_content is None
                 if status_code >= 400:
                     final_response_dict = {"message": "Processing failed due to an early error.", "error": "Internal Server Error or specific HTTP exception"}
                 else: # Should not happen if status code is < 400 but response is None
                     final_response_dict = {"message": "Unexpected state: No response content."}

            print("----------------------------------------------------------")
        except Exception as log_e:
            print(f"Error logging response data: {log_e}")
            if not final_response_dict:
                 final_response_dict = {"message": "Error during response logging.", "error": str(log_e)}
        # --- END LOGGING ---

    # Ensure response_content is not None before dumping
    if response_content:
        final_content = response_content.model_dump(exclude_none=True)
        # Ensure HttpUrl is converted to string for JSON response
        if "youtube_url" in final_content:
            final_content["youtube_url"] = str(final_content["youtube_url"])
    else:
        # Fallback if something went wrong before response_content was created
        final_content = {"message": "Error occurred before response generation.", "error": "Internal Server Error or specific HTTP exception"}
        # Try to add URL/ID if available
        if request and request.youtube_url:
            final_content["youtube_url"] = str(request.youtube_url)
        if video_id:
            final_content["video_id"] = video_id


    return JSONResponse(
        status_code=status_code,
        content=final_content
    )


@app.get("/health",
         response_model=dict,
         summary="Health Check",
         description="Checks connectivity to dependent services (AnkiConnect).")
async def health_check():
    anki_ok = check_connection()
    status_code = status.HTTP_200_OK if anki_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if anki_ok else "error", "anki_connect_reachable": anki_ok}
        )
