from openai import OpenAI

from config import Config
from data.birds_csv_loader import load_birds_csv
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from utils import text_splitter as txSplitter


# Pipelin welche die verschiedenen Textdateien zu jedem Vogel aus dem Datenordner liest und
# daraus dann Embeddings produziert. Die Embeddings werden in der ChromaDb mit der Erweiterung von langChain
# direkt abgespeichert.
# Muss nur 1x ausgeführt werden. Danach kann mit Main_Query.py direkt die Abfrage gemacht werden.

def main():
    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)

    # load the birds dataset
    df = load_birds_csv("data/raw/birds_datalist_de.csv")

    # create the embeddings
    birds = df.copy()

    # Initialisieren vom TextSplitter
    splitter = txSplitter.LangChainTextSplitter()

    documents = []

    # Textsplitter ausführen
    for index, row in birds.iterrows():
        document = splitter.get_recursive_character_chunks_from_text_file(bird_name=row['Vogelname'])
        documents.append(document)

    # Variante mit OpenAI Embedding
    
    # # Create the embeddings
    # openai_embeddings = OpenaiEmbeddings(client)
    # embeddings = []

    # # Create the embedding storage (CSV or ChromaDB)
    # embedding_storage = ChromaDBEmbeddingStorage("BirdsV1")

    # for document in documents:
    #     embeddings_dict = {}
    #     for idx, chunk in enumerate(document):
    #         embedding = openai_embeddings.get_embedding(chunk.page_content)
    #         embeddings_dict[chunk.metadata["Bird"] + str(idx)] = embedding

    #     # Save the embeddings in given storage
    #     embedding_storage.save_embeddings(embeddings_dict)

    # # Use the embeddings for RAG
    # # Load embeddings from CSV
    # embeddings = embedding_storage.get_all_embeddings()
    # stored_embeddings = list(embeddings.values())
    # stored_texts = list(embeddings.keys())

    # Variante mit loaken Embedding

    # Create the embedding storage - ChromaDB
    embedding_storage = ChromaDBEmbeddingStorage("BirdsV1_lokal_hf_embeeded")

    # Embeddings in der Db Speichern
    for document in documents:
        # Save the embeddings in given storage
        embedding_storage.add_documents(documents=document)


if __name__ == "__main__":
    main()