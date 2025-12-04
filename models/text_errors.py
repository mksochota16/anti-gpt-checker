from typing import Optional

from pydantic import BaseModel


class TextErrors(BaseModel):
    AMERICAN_ENGLISH_STYLE: Optional[int] = 0
    BRITISH_ENGLISH: Optional[int] = 0
    CASING: Optional[int] = 0
    COLLOCATIONS: Optional[int] = 0
    COMPOUNDING: Optional[int] = 0
    CONFUSED_WORDS: Optional[int] = 0
    GRAMMAR: Optional[int] = 0
    MISC: Optional[int] = 0
    MULTITOKEN_SPELLING: Optional[int] = 0
    NONSTANDARD_PHRASES: Optional[int] = 0
    NUMBERS: Optional[int] = 0
    PHONETICS: Optional[int] = 0
    PRAWDOPODOBNE_LITEROWKI: Optional[int] = 0
    PUNCTUATION: Optional[int] = 0
    REDUNDANCY: Optional[int] = 0
    REPETITIONS_STYLE: Optional[int] = 0
    SEMANTICS: Optional[int] = 0
    SPELLING: Optional[int] = 0
    STYLE: Optional[int] = 0
    SYNTAX: Optional[int] = 0
    TYPOGRAPHY: Optional[int] = 0
    TYPOS: Optional[int] = 0
    WORD_ORDER: Optional[int] = 0