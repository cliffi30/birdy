from data.birds_csv_loader import load_birds_csv
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from utils.text_splitter import LangChainTextSplitter

chroma_embedding_storage_name = "BirdsV1_lokal_hf_embeeded"

# Pipelin welche die verschiedenen Textdateien zu jedem Vogel aus dem Datenordner liest und
# daraus dann Embeddings produziert. Die Embeddings werden in der ChromaDb mit der Erweiterung von langChain
# direkt abgespeichert.
# Muss nur 1x ausgeführt werden. Danach kann mit Main_Query.py direkt die Abfrage gemacht werden.
def main():
    # load the birds dataset
    df = load_birds_csv("data/raw/birds_datalist_de.csv")

    # create the embeddings
    birds = df.copy()

    # Initialisieren vom TextSplitter
    splitter = LangChainTextSplitter()

    documents = []

    # Textsplitter ausführen
    for index, row in birds.iterrows():
        document = splitter.get_recursive_character_chunks_from_text_file(bird_name=row['Vogelname'])
        documents.append(document)

    # Create the embedding storage - ChromaDB
    embedding_storage = ChromaDBEmbeddingStorage(chroma_embedding_storage_name)

    # Embeddings in der Db Speichern
    for document in documents:
        # Save the embeddings in given storage
        embedding_storage.add_documents(documents=document)


if __name__ == "__main__":
    main()