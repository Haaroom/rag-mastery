from typing import Dict, Any, Optional
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    MarkdownTextSplitter,
)
from langchain.text_splitter import sentence_splitter  # Corrected from lowercase function

class ChunkerFactory:
    """Factory class to initialize different LangChain text splitters."""
    
    _CHUNKER_MAP = {
        "recursive": RecursiveCharacterTextSplitter,
        "character": CharacterTextSplitter,
        "spacy": SpacyTextSplitter,
        "nltk": NLTKTextSplitter,
        "markdown": MarkdownTextSplitter,
        "sentence": sentence_splitter,
    }

    def __init__(self, chunker_type: str, **default_kwargs: Any):
        self.chunker_type = chunker_type.lower()
        self.default_kwargs = default_kwargs
        
        if self.chunker_type not in self._CHUNKER_MAP:
            raise ValueError(f"Unsupported chunker type: {self.chunker_type}")

    def get_chunker(self, **override_kwargs: Any) -> Any:
        """Returns an initialized text splitter instance."""
        chunker_class = self._CHUNKER_MAP[self.chunker_type]
        
        # Merge default configs with any dynamic overrides
        config = {**self.default_kwargs, **override_kwargs}
        
        try:
            return chunker_class(**config)
        except TypeError as e:
            raise TypeError(
                f"Invalid configuration for {self.chunker_type} chunker: {e}"
            )
