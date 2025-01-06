from data.birds_csv_loader import load_birds_csv
from embeddings.embedding_storage import EmbeddingStorage
from utils.birds_encyclopedia_extractor import extract_structured_content
from utils.text_splitter import LangChainTextSplitter


def build_birds_embeddings(
        embedding_storage: EmbeddingStorage,
        csv_path: str = "data/raw/birds_datalist_de.csv",
        include_encyclopedia_embeddings: bool = False,
        encyclopedias_path: [str] = ["data/external/Illustrated Encyclopedia of Birds Part 1.pdf",
                                     "data/external/Illustrated Encyclopedia of Birds Part 2.pdf"]
):
    """
    Builds embeddings from a birds CSV and stores them via the given embedding storage.
    Instead of passing Document objects, this function embeds text chunks
    and saves them as text->embedding pairs, which is compatible with both
    ChromaDB and Neo4j backend storages.
    """
    print("Building embeddings...")

    df = load_birds_csv(csv_path)

    splitter = LangChainTextSplitter()

    embeddings = []
    for index, row in df.iterrows():
        # For each bird, split into smaller segments
        docs = splitter.get_recursive_character_chunks_from_text_file(bird_name=row["Vogelname"])
        embeddings.append(docs)

    if include_encyclopedia_embeddings:
        for path in encyclopedias_path:
            embeddings.append(extract_structured_content(path))

    embedding_storage.save_embeddings(embeddings)
    print("Embeddings saved.")
