import csv
from typing import Dict, List

from src.embeddings.embedding_storage import EmbeddingStorage


# Stores the embeddings in a CSV file.
class CSVEmbeddingStorage(EmbeddingStorage):
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.embeddings = self._load_embeddings()

    def _load_embeddings(self) -> Dict[str, List[float]]:
        embeddings = {}
        try:
            with open(self.csv_file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    text = row[0]
                    vector = list(map(float, row[1:]))
                    embeddings[text] = vector
        except FileNotFoundError:
            pass  # File doesn't exist yet
        return embeddings

    def save_embeddings(self, embeddings_dict: Dict[str, List[float]]):
        with open(self.csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for text, vector in embeddings_dict.items():
                writer.writerow([text] + vector)

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        return list(self.embeddings)

    def get_embedding(self, text: str) -> List[float]:
        return self.embeddings.get(text, [])