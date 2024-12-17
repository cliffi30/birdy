import numpy as np
from openai import OpenAI

from src.embeddings.embeddings import Embeddings


class OpenaiEmbeddings(Embeddings):
    def __init__(self, client: OpenAI, model = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return embeddings.data[0].embedding