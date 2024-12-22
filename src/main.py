import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from data.birds_csv_loader import load_birds_csv
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from embeddings.openai_embeddings import OpenaiEmbeddings
from models.openai_completions import OpenaiCompletions


def main():
    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)

    # load the birds dataset
    df = load_birds_csv().head(10)

    # create the embeddings
    birds = df[-2:].copy(2)
    # Combine the columns into a single text (to be used for embeddings)
    birds['combined_text'] = (
        'Name:' + birds['Name'] + ', ' +
        'Breeding Time:' + birds['Breeding Time'] + ', ' +
        'Region:' + birds['Region'] + ', ' +
        'Characteristics:' + birds['Characteristics'] + ', ' +
        'Weight (kg):' + birds['Weight (kg)'].astype(str) + ', ' +
        'Size (cm):' + birds['Size (cm)'].astype(str)
    )

    # Create the embeddings
    openai_embeddings = OpenaiEmbeddings(client)
    birds['embeddings'] = birds['combined_text'].apply(lambda x: openai_embeddings.get_embedding(x))
    embeddings_dict = {}
    for index, row in birds.iterrows():
        text = row['combined_text']
        embedding = row['embeddings']
        embeddings_dict[text] = embedding
        birds.at[index, 'embeddings'] = embedding

    # Create the embedding storage (CSV or ChromaDB)
    # embedding_storage = CSVEmbeddingStorage('embeddings.csv')
    embedding_storage = ChromaDBEmbeddingStorage()

    # Save the embeddings in given storage
    embedding_storage.save_embeddings(embeddings_dict)

    # Use the embeddings for RAG
    # Load embeddings from CSV
    embeddings = embedding_storage.get_all_embeddings()
    stored_embeddings = list(embeddings.values())
    stored_texts = list(embeddings.keys())

    questions = [
        'What are the characteristics of the Eastern Bluebird?',
    ]

    openai_completions = OpenaiCompletions(client)
    for question in questions:
        question_embedding = openai_embeddings.get_embedding(question)

        # Compute cosine similarity between the question and stored embeddings
        similarities = cosine_similarity([question_embedding], stored_embeddings)[0]
        # Find the most similar text
        most_similar_index = np.argmax(similarities)
        most_similar_text = stored_texts[most_similar_index]

        # Generate the response using the most relevant context
        context = most_similar_text
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        response = openai_completions.get_completion(prompt)
        print(f"Question: {question}")
        print(f"Response: {response}")
        print("--------------------")


if __name__ == "__main__":
    main()