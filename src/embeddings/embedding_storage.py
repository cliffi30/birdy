from abc import ABC, abstractmethod
from typing import Dict, List


class EmbeddingStorage(ABC):
    @abstractmethod
    def save_embeddings(self, embeddings_dict: Dict[str, List[float]]):
        """
        Saves the embeddings to the storage.
        """
        pass

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """
        Retrieves all embeddings from the storage.

        Returns:
            List[float]: The embedding for the given text.
        """
        pass

    def get_embedding(self, text: str) -> List[float]:
        """
        Retrieves the embedding for the given text.

        Args:
            text (str): The text to retrieve the embedding for.

        Returns:
            List[float]: The embedding for the given text.
        """
        pass