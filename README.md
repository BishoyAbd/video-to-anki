# YouTube to Anki Agent


Automate the creation of Anki flashcards from YouTube video transcripts. This agent fetches video transcripts, processes the content using LLMs (Mixtral on colab and ngrok tuneling or your own server), and generates flashcards, optionally adding them directly to your Anki collection via the AnkiConnect addon.

## Overview

Learning effectively from educational YouTube videos often requires active recall and spaced repetition. Manually creating flashcards for key concepts can be time-consuming. This project aims to streamline this process by automatically extracting information from video transcripts and formatting it into Anki flashcards, saving you time and enhancing your learning workflow.

## Features

*   **Transcript Fetching:** Retrieves video transcripts using the `youtube-transcript-api`.
*   **Content Processing:** Extracts key information from transcripts (extensible for different processing strategies, e.g., using Large Language Models).
*   **Flexible Output:**
    *   Generates standard CSV files for manual import into Anki.
    *   Creates Anki package files (`.apkg`) for easy sharing and importing.
*   **Direct Anki Integration:** Seamlessly adds generated cards to a specified deck in your local Anki instance using the **AnkiConnect** addon.
*   **Configurable:** Allows setting API keys, output paths, default Anki deck names, and AnkiConnect parameters.

## How it Works

1.  **Input:** Takes a YouTube video URL as input.
2.  **Fetch:** Retrieves the video's transcript.
3.  **Process:** Analyzes the transcript text to identify key points, definitions, questions, or other relevant information suitable for flashcards. (The specific processing logic can be customized).
4.  **Generate/Add:**
    *   Formats the extracted information into flashcards.
    *   Outputs the flashcards as a CSV or `.apkg` file.
    *   *Alternatively*, connects to a running Anki instance via AnkiConnect and adds the cards directly to the specified deck.

## Installation

Follow these steps to set up the project environment:

1.  **Prerequisites:**
    *   Python 3.8 or higher.
    *   Git.
    *   Anki Desktop application (required for using the generated cards or the AnkiConnect integration).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Agent_from_youtube_to_anki.git
    cd Agent_from_youtube_to_anki
    ```
    *(Replace `your-username` with the actual repository owner's username)*

3.  **Set Up a Virtual Environment (Recommended):**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install and Configure AnkiConnect (Optional, for Direct Integration):**
    *   Open your Anki Desktop application.
    *   Navigate to `Tools` > `Add-ons`.
    *   Click `Get Add-ons...`.
    *   Paste the AnkiConnect addon code: `2055492159` (Verify code on the [official AnkiConnect page](https://ankiweb.net/shared/info/2055492159)).
    *   Click `OK`, wait for installation, and restart Anki when prompted.
    *   **Important:** By default, AnkiConnect listens on `http://127.0.0.1:8765`. If you need to change this or configure trusted origins (e.g., if running the script in a container or remote environment), go to `Tools` > `Add-ons`, select `AnkiConnect`, click `Config`, and modify the settings according to the AnkiConnect documentation.

## Configuration

Certain parameters might need configuration depending on your usage:

*   **API Keys:** If the agent uses external services like OpenAI or Anthropic for content processing, you'll need to provide API keys. Configure these typically via:
    *   Environment Variables (e.g., `export OPENAI_API_KEY='your_key_here'`).
    *   A configuration file (e.g., `config.yaml` or `.env`). Refer to the specific implementation details.
*   **AnkiConnect URL:** If you modified the default AnkiConnect address (`http://127.0.0.1:8765`) in Anki's settings, update the corresponding setting in this agent's configuration (e.g., an `ANKICONNECT_URL` variable or config entry).
*   **Default Deck Name:** You might be able to set a default Anki deck name in the configuration to avoid specifying it via command-line arguments every time.

*(Add specific details about where and how to configure these settings based on your project's structure, e.g., "Create a `.env` file in the root directory..." or "Modify the `config.yaml` file...")*

## Usage

Run the agent from your terminal within the activated virtual environment.

**Command-Line Examples:**

1.  **Generate a CSV file:**
    ```bash
    python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output my_cards.csv
    ```
    *   *Import Manually:* In Anki, go to `File` > `Import...`, select `my_cards.csv`, choose the target deck, and map the CSV columns to Anki note fields.

2.  **Generate an Anki Package (`.apkg`) file:**
    ```bash
    python main.py --url "YOUTUBE_VIDEO_URL" --output my_deck.apkg --deck-name "Video Notes"
    ```
    *   *Import Manually:* Double-click the `my_deck.apkg` file or use Anki's `File` > `Import...`.

3.  **Add Cards Directly to Anki via AnkiConnect:**
    *   **Ensure Anki is running with the AnkiConnect addon enabled.**
    ```bash
    python main.py --url "YOUTUBE_VIDEO_URL" --use-ankiconnect --deck-name "YouTube Imports"
    ```
    *   The script will attempt to connect to AnkiConnect (default: `http://127.0.0.1:8765`) and add the generated cards to the "YouTube Imports" deck. The deck will be created if it doesn't exist.

*(Adjust `main.py` and arguments like `--url`, `--output`, `--deck-name`, `--use-ankiconnect` based on your script's actual implementation.)*

## Dependencies

*   Python 3.8+
*   `youtube-transcript-api`: For fetching transcripts.
*   `requests`: For communicating with AnkiConnect.
*   `genanki`: For generating `.apkg` files.
*   *(Add other major dependencies, e.g., LLM client libraries)*
*   **Runtime Dependency (Optional):** Anki Desktop application with the AnkiConnect addon installed and running for direct integration.

See `requirements.txt` for a full list of Python packages.

## Contributing

Contributions are welcome! If you'd like to improve the agent or add features, please follow these steps:

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally (`git clone https://github.com/your-username/Agent_from_youtube_to_anki.git`).
3.  Create a **new branch** for your feature or bug fix (`git checkout -b feature/your-new-feature`).
4.  Make your **changes**.
5.  **Commit** your changes (`git commit -am 'Add some amazing feature'`).
6.  **Push** the branch to your fork (`git push origin feature/your-new-feature`).
7.  Open a **Pull Request** from your fork's branch to the main repository's `main` branch.

Please ensure your code follows existing style conventions and includes tests for new functionality where appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. *(Ensure you have a LICENSE file in your repository)*