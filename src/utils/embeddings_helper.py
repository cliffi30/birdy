from data.birds_csv_loader import load_birds_csv
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from utils.text_splitter import LangChainTextSplitter

chroma_embedding_storage_name = "BirdsV1_lokal_hf_embedded"

def build_birds_embeddings(use_chrome_embeddings: bool = True):
    # load the birds dataset
    df = load_birds_csv("data/raw/birds_datalist_de.csv")

    # create the embeddings
    birds = df.copy()

    # Initialize text splitter
    splitter = LangChainTextSplitter()

    documents = []

    # Textsplitter ausführen
    for index, row in birds.iterrows():
        document = splitter.get_recursive_character_chunks_from_text_file(bird_name=row['Vogelname'])
        documents.append(document)

    if use_chrome_embeddings is True:
        # Create the embedding storage - ChromaDB
        embedding_storage = ChromaDBEmbeddingStorage(chroma_embedding_storage_name)

        # Save the embeddings
        for document in documents:
            # Save the embeddings in given storage
            embedding_storage.add_documents(documents=document)