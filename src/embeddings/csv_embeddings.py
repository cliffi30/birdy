import os

import pandas as pd

from src.embeddings.embeddings import Embeddings


class CsvEmbedding(Embeddings):
    def __init__(self, embeddings_file='./test_embeddings.csv'):
        self.embeddings_file = embeddings_file
        self.embeddings = self.load_embeddings()

    def write_embeddings(self, texts, embeddings):
        df = pd.DataFrame({'text': texts, 'embedding': embeddings})
        df.to_csv(self.embeddings_file, index=False)

    def load_embeddings(self):
        if not os.path.exists(self.embeddings_file) or os.path.getsize(self.embeddings_file) == 0:
            return pd.DataFrame(columns=['text', 'embedding'])
        return pd.read_csv(self.embeddings_file)

    def get_embedding(self, word):
        embedding_row = self.embeddings[self.embeddings['text'] == word]
        if not embedding_row.empty:
            return embedding_row['embedding'].values[0]
        return None