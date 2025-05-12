# src/prompts.py

# Template for the prompt sent to the LLM for processing text chunks
# It expects the variable '{text_chunk}' to be formatted into it.
# Using plain text without Markdown formatting.
LLM_PROCESS_PROMPT_TEMPLATE = """\
[SYSTEM] You segment German transcripts into logical sentences with proper punctuation, then translate to English. Create complete sentences from unpunctuated text.

Segment and translate this German transcript:

TRANSCRIPT:
{text_chunk}

RULES:
1. Find ALL LOGICAL BREAKS in the text (pauses, topic changes, sentence endings)
2. Create complete, meaningful sentences - as many as naturally exist
3. Each sentence should express one complete thought
4. Add capital letters and punctuation
5. Translate each sentence to English
6. Each sentence must be ≤ 10 words to work as flashcards

EXAMPLE:
"das ist ein wichtiges Thema KI ist überall maschinelles Lernen verändert wie wir arbeiten neue Tools müssen verstanden werden"

GOOD OUTPUT:
[
  {{"german": "Das ist ein wichtiges Thema.", "english": "This is an important topic."}},
  {{"german": "KI ist überall.", "english": "AI is everywhere."}},
  {{"german": "Maschinelles Lernen verändert wie wir arbeiten.", "english": "Machine learning changes how we work."}},
  {{"german": "Neue Tools müssen verstanden werden.", "english": "New tools need to be understood."}}
]

BAD OUTPUT (don't do this - fragments, not complete thoughts):
[
  {{"german": "Das ist ein wichtiges.", "english": "This is an important."}},
  {{"german": "KI ist.", "english": "AI is."}}
]

BAD OUTPUT (don't do this - everything in one sentence):
[
  {{"german": "Das ist ein wichtiges Thema KI ist überall maschinelles Lernen verändert wie wir arbeiten neue Tools müssen verstanden werden.", "english": "This is an important topic AI is everywhere machine learning changes how we work new tools need to be understood."}}
]

Extract ALL natural sentences, each expressing ONE complete thought.
Ignore filler like "[Musik]". Output ONLY the JSON list with the correct field names "german" and "english".
"""
