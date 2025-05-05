# Project: Text-to-Anki Service

## 1. Introduction

This document outlines the requirements, proposed technology stack, and development plan for a service that processes a given text chunk, extracts relevant sentence pairs (e.g., German/English technical terms), and adds them as flashcards to a user's Anki Desktop application via the AnkiConnect add-on. The service is intended to be easily shareable and runnable via Google Colab and ngrok for demonstration and free use.

## 2. Goals

* Create a web service that accepts text input.
* Process the input text to generate structured data suitable for Anki flashcards (e.g., sentence pairs).
* Interact with the user's local Anki Desktop instance via AnkiConnect to add the generated flashcards.
* Provide a clear setup and usage guide for deployment on a local machine or via Google Colab/ngrok.
* Structure the project following best practices for clarity, maintainability, and sharing on GitHub.

## 3. Functional Requirements

1.  **FR1: Text Input:** The service must provide an HTTP endpoint (e.g., `/api/process`) that accepts a POST request containing a chunk of text data (potentially unstructured or fragmented, like a transcript snippet).
2.  **FR2: Text Processing (Refined v2):**
    * The service receives the input text chunk.
    * The service sends the chunk to a configured LLM (e.g., Gemini API).
    * The LLM is prompted to:
        * Analyze the input chunk.
        * Identify and/or reconstruct one or more complete, coherent, and grammatically correct German sentences based on the content and context of the chunk. Prefer using original wording where possible, completing fragmented sentences.
        * For each generated German sentence, provide its accurate English translation.
        * Return the results as a structured list (e.g., a JSON array of objects, where each object has a "german" and "english" key).
    * Example LLM Output Structure:
        ```json
        [
          {"german": "Sentence 1 in German.", "english": "Sentence 1 in English."},
          {"german": "Sentence 2 in German.", "english": "Sentence 2 in English."}
        ]
        ```
    * The service parses this structured response from the LLM to get the {German Sentence, English Translation} pairs.
3.  **FR3: Anki Card Creation:** The service must format the generated sentence pairs into the structure required by the AnkiConnect `addNotes` action (specifying deck name, model name, fields, tags). The default model should be `"Basic (and reversed card)"`.
4.  **FR4: AnkiConnect Interaction:** The service must send the formatted note data to the user's configured local AnkiConnect endpoint (defaulting to `http://localhost:8765`).
5.  **FR5: Response:** The service endpoint should return a JSON response indicating success or failure, potentially including the number of cards attempted/added and any errors encountered (e.g., AnkiConnect not reachable, LLM errors, processing errors, LLM response parsing errors).
6.  **FR6: Configuration:** Key parameters must be configurable, ideally via environment variables or a `.env` file:
    * `ANKICONNECT_URL` (e.g., `http://localhost:8765`)
    * `ANKI_DECK_NAME` (e.g., `Tech Deutsch`)
    * `ANKI_MODEL_NAME` (e.g., `Basic (and reversed card)`)
    * `LLM_API_KEY` (e.g., Google AI API Key)
    * `LLM_PROCESS_PROMPT` (The single prompt used to generate the structured German/English sentence pairs from the input chunk)
    * `ANKI_TAGS` (Optional comma-separated list of tags)

## 4. Non-Functional Requirements

1.  **NFR1: Usability:** The service should be easy to set up and run, especially via the Colab/ngrok method. A clear `README` is essential.
2.  **NFR2: Reliability:** The service should handle common errors gracefully (e.g., AnkiConnect unavailable, invalid input format, LLM API errors, LLM response parsing failure, no sentences generated) and provide informative feedback.
3.  **NFR3: Maintainability:** The code should be well-structured, commented, and follow Python best practices (e.g., using FastAPI's dependency injection, Pydantic models).
4.  **NFR4: Shareability:** The project should be easily shareable via GitHub, including necessary setup files (`requirements.txt`, `.env.example`).

## 5. Proposed Tech Stack

* **Language:** Python 3.x
* **Web Framework:** FastAPI
* **HTTP Client:** `requests` (for AnkiConnect)
* **LLM Interaction:** Google AI Python SDK (`google-generativeai`)
* **Colab/ngrok:** Google Colab, `pyngrok`
* **Dependency Management:** `pip` and `requirements.txt`
* **Configuration:** `python-dotenv` (for handling `.env` files)
* **Data Validation:** Pydantic (comes with FastAPI)

## 6. High-Level Plan & Project Structure

**Phase 1: Core Backend Setup**
* Set up FastAPI project structure (as proposed).
* Implement configuration loading (`config.py`, `.env`).
* Implement basic API endpoint (`/api/process` in `main.py`) accepting text and returning a placeholder.
* Define Pydantic models for request/response (`models.py`).
* Integrate AnkiConnect interaction logic (`anki.py`).

**Phase 2: Text Processing Integration**
* Implement LLM interaction logic in `processing.py` for generating the structured German/English sentence pairs using a single prompt.
* Implement robust parsing of the LLM's structured response.
* Integrate this logic into the `/api/process` endpoint.
* Handle LLM API key management securely via configuration.
* Refine error handling for LLM calls and response parsing.

**Phase 3: Colab/ngrok Deployment**
* Create a Colab notebook (`notebooks/run_server_colab.ipynb`) demonstrating server setup (install deps, set API key, run FastAPI with uvicorn, launch ngrok).
* Provide clear instructions within the notebook and `README`.

**Phase 4: Documentation & Packaging**
* Write comprehensive `README.md` (setup, configuration, API usage, Colab guide, contributing).
* Finalize `requirements.txt`.
* Add `LICENSE` file (e.g., MIT or Apache 2.0).
* Add `.gitignore`.
* Ensure code is well-commented and structured.
