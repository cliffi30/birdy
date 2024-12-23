import argparse
import sys

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from data.birds_csv_loader import load_birds_csv
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from embeddings.openai_embeddings import OpenaiEmbeddings
from embeddings.transformer_embeddings import TransformerEmbeddings
from models.ollama_completions import OllamaCompletions
from models.openai_completions import OpenaiCompletions
from utils.sentiment_analyzer import SentimentAnalyzer


def main(useLlamaCompletions: bool = False, useOpenAiCompletions: bool = True, useLlamaEmbeddings: bool = False, useOpenAiEmbeddings: bool = True):
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
    transformer_embeddings = TransformerEmbeddings()

    # set the embeddings we want to use
    embeddings_variant = openai_embeddings

    birds['embeddings'] = birds['combined_text'].apply(lambda x: embeddings_variant.get_embedding(x))
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
        'Hi, You told me that the Eastern Bluebird is a small bird, but now the bird is huge! Why are you lying?',
        "Ich bin richtig wütend. Ich habe einen Vogel gekauft und er ist nach wenigen Tagen bereits gestorben. Ich möchte Schadenersatz! Der seelische Schmerz ist unerträglich.",
        "Ich muss schon sagen, die Qualität dieses Vogels mit dem langen roten Schnabel ist wirklich hervorragend. Es ist zwar etwas knapp für ihn im Käfig. Ich bin sehr zufrieden.",
    ]

    if useOpenAiCompletions == "true":
        openai_completions = OpenaiCompletions(client)
    if useLlamaCompletions == "true":
        ollama_completions = OllamaCompletions(None)

    sentiment_analyzer = SentimentAnalyzer()
    for question in questions:
        # Calculate the sentiment of the question
        sentiment = sentiment_analyzer.analyze_sentiment(question)

        question_embedding = embeddings_variant.get_embedding(question)

        # Compute cosine similarity between the question and stored embeddings
        similarities = cosine_similarity([question_embedding], stored_embeddings)[0]
        # Find the most similar text
        most_similar_index = np.argmax(similarities)
        most_similar_text = stored_texts[most_similar_index]

        # Generate the response using the most relevant context
        context = most_similar_text
        response = ""
        if useOpenAiCompletions == "true":
            response = openai_completions.get_completion(context, question, sentiment['label'], sentiment['score'])
            print("--------------------")
            print("Model: OpenAI")
        if useLlamaCompletions == "true":
            response = ollama_completions.get_completion(context, question)
            print("--------------------")
            print("Model: Ollama")

        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Response: {response}")
        print("--------------------")




if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Birdy - A bird shop assistant")

    # Adding optional argument
    parser.add_argument("-lc", "--useLlamaCompletions", required=False, default="false", help = "true to use llama with ollama to create completions")
    parser.add_argument("-oac", "--useOpenAiCompletions", required=False, default="true", help = "true to use OpenAI to create completions")
    parser.add_argument("-le", "--useLlamaEmbeddings", required=False, default="false", help = "true to use llama with ollama to create embeddings")
    parser.add_argument("-oae", "--useOpenAiEmbeddings", required=False, default="true", help = "true to use OpenAI to create embeddings")

    # Read arguments from command line
    args = parser.parse_args()

    print(f'useLlamaCompletions: {args.useLlamaCompletions}')
    print(f'useOpenAiCompletions: {args.useOpenAiCompletions}')
    print(f'useLlamaEmbeddings: {args.useLlamaEmbeddings}')
    print(f'useOpenAiEmbeddings: {args.useOpenAiEmbeddings}')

    if (args.useLlamaCompletions == "false" and args.useOpenAiCompletions == "false"):
        print("At least one of the completion models should be enabled")
        sys.exit(1)
    if (args.useLlamaEmbeddings == "false" and args.useOpenAiEmbeddings == "false"):
        print("At least one of the embedding models should be enabled")
        sys.exit(2)
    main(useLlamaCompletions=args.useLlamaCompletions, useOpenAiCompletions=args.useOpenAiCompletions, useLlamaEmbeddings=args.useLlamaEmbeddings, useOpenAiEmbeddings=args.useOpenAiEmbeddings)
