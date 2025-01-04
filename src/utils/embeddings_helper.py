from data.birds_csv_loader import load_birds_csv
from embeddings.embedding_storage import EmbeddingStorage
from embeddings.embeddings import Embeddings
from utils.text_splitter import LangChainTextSplitter


def build_birds_embeddings(
        embedding_storage: EmbeddingStorage,
        csv_path: str = "data/raw/birds_datalist_de.csv",
):
    """
    Builds embeddings from a birds CSV and stores them via the given embedding storage.
    Instead of passing Document objects, this function embeds text chunks
    and saves them as text->embedding pairs, which is compatible with both
    ChromaDB and Neo4j backend storages.
    """
    df = load_birds_csv(csv_path).head(10)

    splitter = LangChainTextSplitter()

    text_embedding_map = {}
    for index, row in df.iterrows():
        # For each bird, split into smaller segments
        docs = splitter.get_recursive_character_chunks_from_text_file(bird_name=row["Vogelname"])
        text_embedding_map[index] = docs

    # Let the embedding storage handle saving
    embedding_storage.save_embeddings(text_embedding_map)
