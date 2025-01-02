from typing import Dict, List
from uuid import uuid4

from chromadb import PersistentClient
from embeddings.embedding_storage import EmbeddingStorage
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class ChromaDBEmbeddingStorage(EmbeddingStorage):
    def __init__(
            self,
            embedder: Embeddings,
            collection_name: str = "birds",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            persist_directory: str = "chroma_db/",
            storage_name = "BirdsV1_lokal_hf_embedded"
    ):
        self.collection_name = collection_name
        self.storage_name = storage_name
        self.embed_model = embedder

        # Initialize ChromaDB
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Initialize Vector Store
        self.vector_store = Chroma(
            collection_name=collection_name,
            client=self.client,
            embedding_function=self.embed_model
        )

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def save_embeddings(self, embeddings_dict: Dict[str, List[Document]]):
        uuids = [str(uuid4()) for _ in range(len(embeddings_dict))]
        self.vector_store.add_documents(filter_complex_metadata(embeddings_dict.values()), ids=uuids)
        self.client.get_or_create_collection(self.collection_name)

    # Speichert LangChain Documents in der Datenbank als Embeddings ab
    def add_documents(self, documents):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(filter_complex_metadata(documents), ids=uuids)
        collection = self.client.get_or_create_collection(self.collection_name)
        print(collection.peek())

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        embeddings_dict = {}
        results = self.collection.get(include=['embeddings'])
        for document, embedding in zip(results['ids'], results['embeddings']):
            embeddings_dict[document] = embedding
        return embeddings_dict

    def get_embedding(self, text: str, n_results: int) -> list[Document]:
        return self.vector_store.similarity_search(query=text, k=n_results)