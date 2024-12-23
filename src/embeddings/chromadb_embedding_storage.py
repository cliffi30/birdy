from typing import Dict, List
from uuid import uuid4

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings

from src.embeddings.embedding_storage import EmbeddingStorage


class ChromaDBEmbeddingStorage(EmbeddingStorage):
    def __init__(
            self,
            collection_name: str = "birds",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            persist_directory: str = "chroma_db/"
    ):
        self.collection_name = collection_name
        self.embed_model = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")

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

    def save_embeddings(self, embeddings_dict):
        for text, vector in embeddings_dict.items():
            self.collection.upsert(text, vector)

    # Speichert LangChain Documents in der Datenbank als Embeddings ab
    def add_documents(self, documents):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(filter_complex_metadata(documents), ids=uuids)
        collection = self.client.get_or_create_collection(self.collection_name)
        print(collection.peek())

    def query_vectorstore(self, text):
        result = self.vector_store.similarity_search(query=text[0], k=3)
        return result

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
