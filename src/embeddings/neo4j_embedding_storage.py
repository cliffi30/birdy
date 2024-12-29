from typing import List, Dict

import numpy as np
from embeddings.embedding_storage import EmbeddingStorage
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM


class Neo4jEmbeddingStorage(EmbeddingStorage):
    def __init__(self, uri: str, user: str, password: str, use_openai: bool = False):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        if use_openai is True:
            self.embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.llm = OpenAILLM(model_name="gpt-4", model_params={"temperature": 0})
        else:
            self.embedder = LocalEmbeddings(model="path/to/local/llama/embeddings")
            self.llm = LocalLLM(model_name="path/to/local/llama/model", model_params={"temperature": 0})

    def store(self, embeddings: Dict[str, np.ndarray]):
        with self.driver.session() as session:
            for node_id, embedding in embeddings.items():
                session.run(
                    "MERGE (n:Node {id: $node_id}) SET n.embedding = $embedding",
                    node_id=node_id,
                    embedding=embedding.tolist(),
                )

    def load(self, node_ids: List[str]) -> Dict[str, np.ndarray]:
        with self.driver.session() as session:
            embeddings = session.run(
                "MATCH (n:Node) WHERE n.id IN $node_ids RETURN n.id AS id, n.embedding AS embedding",
                node_ids=node_ids,
            )
            return {embedding["id"]: np.array(embedding["embedding"]) for embedding in embeddings}

    def close(self):
        self.driver.close()