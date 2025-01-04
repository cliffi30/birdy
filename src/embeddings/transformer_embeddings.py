from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel

from embeddings.embeddings import Embeddings


# generates the embeddings using a model from the Hugging Face Transformers library
# (after download the processing is done on the local machine)
class TransformerEmbeddings(Embeddings):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

    def get_langchain_embedder(self):
        return HuggingFaceEmbeddings(model_name=self.model_name)
