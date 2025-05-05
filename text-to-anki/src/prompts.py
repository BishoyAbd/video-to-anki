# src/prompts.py

# Template for the prompt sent to the LLM for processing text chunks
# It expects the variable '{text_chunk}' to be formatted into it.
# Using plain text without Markdown formatting.
LLM_PROCESS_PROMPT_TEMPLATE = """\
Role: You are an expert linguist tasked with extracting concise information for language learning flashcards.

Task: Analyze the provided German text chunk. Your goal is to identify key concepts or statements and reconstruct them into short, complete, and grammatically correct German sentences.

Constraints:
1. Each generated German sentence MUST contain 10 words or fewer.
2. Ignore non-content markers like '[Musik]'.
3. Focus on extracting meaningful statements, even if they require slight rephrasing for conciseness and grammatical correctness based on the chunk's context.
4. For EACH generated German sentence, provide its accurate English translation.

Output Format:
Return your response ONLY as a valid JSON list of objects. Do not include any text before or after the JSON list. Each object in the list must have exactly two keys:
"german": The short (<= 10 words) German sentence.
"english": The corresponding English translation.

Example Output:
[
  {{"german": "KI-Agenten sind sehr relevant.", "english": "AI agents are very relevant."}},
  {{"german": "Sie unterscheiden sich von ChatGPT.", "english": "They differ from ChatGPT."}}
]

German Text Chunk to Analyze:
'''
{text_chunk}
'''

JSON Output:
"""

# You would typically import this variable into other modules like:
# from .prompts import LLM_PROCESS_PROMPT_TEMPLATE
