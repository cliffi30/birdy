from typing import Dict, List

from embeddings.embedding_storage import EmbeddingStorage
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class Neo4jEmbeddingStorage(EmbeddingStorage):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        embedder: Embeddings,
        index_name: str = "bird_vector_index",
        node_label: str = "Bird",
        text_node_properties: list[str] = ["text"],
        embedding_node_property: str = "embedding"
    ):
        """
        Initialize Neo4j Vector store with embeddings.
        """
        self.embedder = embedder
        self.uri = uri
        self.user = user
        self.password = password
        self.index_name = index_name
        self.node_label = node_label
        self.existing_graph = Neo4jVector.from_existing_graph(
            url=self.uri,
            username=self.user,
            password=self.password,
            index_name=self.index_name,
            node_label=self.node_label,
            embedding_node_property=embedding_node_property,
            embedding=self.embedder,
            text_node_properties=text_node_properties,
        )

    def save_embeddings(self, embeddings: List[List[Document]]):
        """
        Save documents and their embeddings to Neo4j using vector index.
        """
        for docs in embeddings:
            self.existing_graph.add_documents(docs)

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """
        Retrieve all stored text->embedding pairs.
        """
        query = f"""
        MATCH (n:{self.existing_graph.node_label})
        RETURN n.{self.existing_graph.text_node_property} as text, 
               n.{self.existing_graph.embedding_node_property} as embedding
        """
        results = self.vector_store.graph.run(query)
        
        embeddings_dict = {}
        for record in results:
            text = record["text"]
            embedding = record["embedding"]
            if text and embedding:
                embeddings_dict[text] = embedding
        
        return embeddings_dict

    def get_embedding(self, text: str, n_results: int) -> list[Document]:
        """
        Get embedding for text using similarity search.
        """
        return self.existing_graph.similarity_search(query=text, k=n_results)

    def close(self):
        """Close Neo4j connection."""
        if hasattr(self.vector_store, "graph"):
            self.vector_store.graph.close()