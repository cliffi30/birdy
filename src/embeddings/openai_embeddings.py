from embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


# generates the embeddings using the OpenAI API
class OpenaiEmbeddings(Embeddings):
    def __init__(self, client: OpenAI, model = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self.dimensions = 768 # reducing from 1536 to 768 to align with the other embeddings (e.g. Hugging Face)

    def get_embedding(self, text: str) -> list[float]:
        embeddings = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
            dimensions=self.dimensions
        )
        return embeddings.data[0].embedding

    def get_langchain_embedder(self):
        return OpenAIEmbeddings(model=self.model, dimensions=self.dimensions)