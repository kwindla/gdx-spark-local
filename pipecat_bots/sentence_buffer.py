"""Sentence buffer for accumulating LLM output and extracting at boundaries.

Used by LlamaCppBufferedLLMService to manage text accumulation and
sentence-boundary extraction for optimal TTS integration.
"""

import re
from typing import Optional


class SentenceBuffer:
    """Accumulates LLM output and extracts text at sentence boundaries.

    The buffer tracks both accumulated text and a token count for determining
    when to force extraction at non-sentence boundaries.

    Key behaviors:
    - extract_complete_sentences(): Returns all complete sentences, keeps tail
    - extract_at_boundary(): Force extraction at best boundary (sentence/clause/word)
    - Token count resets on emit (regardless of remainder size)

    Design decision: On emit, we reset token_count = 0 even if buffer has a remainder.
    Rationale:
    - The remainder is always an incomplete sentence tail (small)
    - We're about to generate more tokens anyway
    - Precise tracking of remainder tokens would require tokenization
    - The hard_max limit is generous enough that slight undercounting is safe
    """

    def __init__(self):
        """Initialize empty buffer."""
        self.text: str = ""
        self.token_count: int = 0  # Tokens accumulated since last emit

    def add(self, text: str, tokens: int) -> None:
        """Append text from LLM generation with token count.

        Args:
            text: Text content from generation
            tokens: Number of tokens generated (from tokens_predicted)
        """
        self.text += text
        self.token_count += tokens

    def clear(self) -> None:
        """Reset buffer completely (on interruption or completion)."""
        self.text = ""
        self.token_count = 0

    def reset_token_count(self) -> None:
        """Reset token count after emit (regardless of remainder size).

        Called after emitting text to TTS. The remainder (if any) is always
        an incomplete tail that will be completed in the next generation.
        """
        self.token_count = 0

    def has_content(self) -> bool:
        """Check if buffer has any non-whitespace text."""
        return bool(self.text.strip())

    def extract_complete_sentences(self) -> Optional[str]:
        """Extract all complete sentences, keep incomplete tail.

        Finds the LAST sentence boundary (.!? followed by space) and returns
        all text up to and including that boundary. The incomplete tail
        remains in the buffer for further accumulation.

        The pattern requires trailing whitespace to avoid false positives on
        abbreviations like "Dr." or decimals like "3.14". End-of-response
        is handled by EOS logic in the LLM service.

        Returns:
            All complete sentences joined, or None if no complete sentence found.

        Example:
            Buffer: "Hello! How are you? I hope"
            Returns: "Hello! How are you? "
            Remaining buffer: "I hope"
        """
        # Find LAST sentence boundary (.!? with optional closing quotes/parens)
        # Requires trailing whitespace to avoid false positives
        pattern = r'[.!?]["\'\)]*\s'

        matches = list(re.finditer(pattern, self.text))
        if not matches:
            return None

        # Last match gives us the end of the last complete sentence
        last_match = matches[-1]
        boundary = last_match.end()

        # Use lstrip() only - preserve trailing whitespace for TTS pacing
        # The trailing space after punctuation helps TTS add natural pauses
        sentences = self.text[:boundary].lstrip()
        self.text = self.text[boundary:]  # Keep incomplete tail

        return sentences if sentences else None

    def extract_at_boundary(self) -> str:
        """Force extraction when buffer exceeds token limit.

        Extracts text, finding the best break point using priority:
        1. Last sentence boundary (.!?)
        2. Last clause boundary (", " or "; " or newline)
        3. Last word boundary (space)
        4. Everything (fallback)

        This ensures we never emit partial words and prefer natural
        break points for TTS prosody.

        Returns:
            Extracted text (may be empty if buffer is empty)
        """
        if not self.text:
            return ""

        search_region = self.text

        # Priority 1: Last sentence boundary
        sentence_pattern = r'[.!?]["\'\)]*\s'
        sentence_matches = list(re.finditer(sentence_pattern, search_region))
        if sentence_matches:
            boundary = sentence_matches[-1].end()
            # Preserve trailing whitespace for TTS pacing
            result = self.text[:boundary].lstrip()
            self.text = self.text[boundary:]
            return result

        # Priority 2: Last clause boundary (", " or "; " or "\n")
        # Find all clause boundaries and pick the last one
        comma_idx = search_region.rfind(", ")
        semi_idx = search_region.rfind("; ")
        newline_idx = search_region.rfind("\n")

        # Find the rightmost clause boundary
        clause_idx = max(comma_idx, semi_idx, newline_idx)
        if clause_idx > 0:
            # Determine boundary offset based on which delimiter was found
            if clause_idx == newline_idx:
                boundary = clause_idx + 1  # Newline is 1 char
            else:
                boundary = clause_idx + 2  # ", " and "; " are 2 chars
            result = self.text[:boundary].lstrip()
            self.text = self.text[boundary:]
            return result

        # Priority 3: Last word boundary
        space_idx = search_region.rfind(" ")
        if space_idx > 0:
            # Include the trailing space for TTS pacing
            result = self.text[:space_idx + 1].lstrip()
            self.text = self.text[space_idx + 1:]
            return result

        # Fallback: emit everything
        result = self.text.strip()
        self.text = ""
        return result

    def __repr__(self) -> str:
        """Debug representation showing buffer state."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"SentenceBuffer(tokens={self.token_count}, "
            f"len={len(self.text)}, text={text_preview!r})"
        )
