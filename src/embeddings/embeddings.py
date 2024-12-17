from abc import ABC, abstractmethod
from typing import List


class EmbeddingsInterface(ABC):
    @abstractmethod
    def get_embedding(self, word: str) -> List[float]:
        """
        Retrieves the embedding for a given word.

        Args:
            word (str): The word to retrieve the embedding for.

        Returns:
            List[float]: The embedding vector for the word.
        """
        pass