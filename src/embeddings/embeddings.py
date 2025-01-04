from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Embeddings(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Retrieves the embedding for a given word.

        Args:
            text (str): The word to retrieve the embedding for.

        Returns:
            Tuple[Optional[str], Optional[float]]: The document and its score.
        """
        pass

    def get_langchain_embedder(self):
        """
        Returns the langchain embedder.
        """
        pass
