from typing import Dict, List

from chromadb import Client, Settings

from src.embeddings.embedding_storage import EmbeddingStorage


class ChromaDBEmbeddingStorage(EmbeddingStorage):
    def __init__(
            self,
            collection_name: str = "birds",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            persist_directory: str = "./chroma_db"
    ):
        # Initialize ChromaDB
        self.client = Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def save_embeddings(self, embeddings_dict):
        for text, vector in embeddings_dict.items():
            self.collection.upsert(text, vector)

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        embeddings_dict = {}
        results = self.collection.get(include=['embeddings'])
        for document, embedding in zip(results['ids'], results['embeddings']):
            embeddings_dict[document] = embedding
        return embeddings_dict

    def get_embedding(self, text, n_results=3):
        return self.collection.get(
            query_texts=[text],
            n_results=n_results
        )