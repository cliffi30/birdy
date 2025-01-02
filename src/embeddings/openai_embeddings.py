from embeddings.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI


# generates the embeddings using the OpenAI API
class OpenaiEmbeddings(Embeddings):
    def __init__(self, client: OpenAI, model = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def get_embedding(self, text: str) -> list[float]:
        embeddings = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return embeddings.data[0].embedding

    def get_langchain_embedder(self):
        return OpenAIEmbeddings(model=self.model)