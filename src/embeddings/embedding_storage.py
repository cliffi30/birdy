from abc import ABC, abstractmethod
from typing import Dict, List

from langchain.schema import Document


class EmbeddingStorage(ABC):
    @abstractmethod
    def save_embeddings(self, embeddings: List[List[Document]]):
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

    def get_embedding(self, text: str, n_results: int) -> List[str]:
        """
        Retrieves the embedding for the given text.

        Args:
            text (str): The text to retrieve the embedding for.
            n_results (int): The number of results to return.

        Returns:
            List[str]: The embeddings for the given text.
        """
        pass