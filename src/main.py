import argparse
import sys

from openai import OpenAI

from config import Config
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from embeddings.neo4j_embedding_storage import Neo4jEmbeddingStorage
from embeddings.openai_embeddings import OpenaiEmbeddings
from embeddings.transformer_embeddings import TransformerEmbeddings
from models.ollama_completions import OllamaCompletions
from models.openai_completions import OpenaiCompletions
from utils.embeddings_helper import build_birds_embeddings
from utils.sentiment_analyzer import SentimentAnalyzer


def main(use_llama_completions: bool = False, use_open_ai_completions: bool = True, use_chroma_embeddings: bool = False,
         recreate_embeddings: bool = False):
    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)

    # Prepare our possible embedders
    openai_embeddings = OpenaiEmbeddings(client)
    transformer_embeddings = TransformerEmbeddings()

    embedder = transformer_embeddings

    # Create the possible embedding storage (CSV, ChromaDB and Neo4j)
    # embedding_storage = CSVEmbeddingStorage('embeddings.csv')
    chroma_db_client = ChromaDBEmbeddingStorage(embedder=embedder.get_langchain_embedder())
    neo4j_storage = Neo4jEmbeddingStorage(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        embedder=embedder.get_langchain_embedder()
    )
    embedding_storage = None
    if use_chroma_embeddings:
        embedding_storage = chroma_db_client
    else:
        embedding_storage = neo4j_storage

    if recreate_embeddings:
        build_birds_embeddings(embedding_storage, embedder)

    questions = [
        'What are the characteristics of the black bird?',
        'Hi, You told me that the Eastern Bluebird is a small bird, but now the bird is huge! Why are you lying?',
        "Ich bin richtig wütend. Ich habe einen Vogel gekauft und er ist nach wenigen Tagen bereits gestorben. Ich möchte Schadenersatz! Der seelische Schmerz ist unerträglich.",
        "Ich muss schon sagen, die Qualität dieses Vogels mit dem langen roten Schnabel ist wirklich hervorragend. Es ist zwar etwas knapp für ihn im Käfig. Ich bin sehr zufrieden.",
    ]

    if use_open_ai_completions:
        openai_completions = OpenaiCompletions(client)
    if use_llama_completions:
        ollama_completions = OllamaCompletions(None)

    sentiment_analyzer = SentimentAnalyzer()
    for question in questions:
        # Calculate the sentiment of the question
        sentiment = sentiment_analyzer.analyze_sentiment(question)

        context = ""
        query_result = neo4j_storage.get_embedding(question, 3)
        for result in query_result:
            context = context + result.page_content + "/n"

        response = ""
        print("--------------------")
        if use_open_ai_completions == "true":
            response = openai_completions.get_completion(context, question, sentiment['label'], sentiment['score'])
            print("Model: OpenAI")
        if use_llama_completions == "true":
            response = ollama_completions.get_completion(context, question)
            print("Model: Ollama")

        print(f"Sentiment: {sentiment}")
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Response: {response}")
        print("--------------------\n\n")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Birdy - A bird shop assistant")

    # Adding optional argument
    parser.add_argument("-lc", "--useLlamaCompletions", required=False, default="false",
                        help="true to use llama with ollama to create completions")
    parser.add_argument("-oac", "--useOpenAiCompletions", required=False, default="true",
                        help="true to use OpenAI to create completions")
    parser.add_argument("-le", "--useLlamaEmbeddings", required=False, default="false",
                        help="true to use llama with ollama to create embeddings")
    parser.add_argument("-oae", "--useOpenAiEmbeddings", required=False, default="true",
                        help="true to use OpenAI to create embeddings")
    # needs to be set to true if the embeddings should be recreated (needed for the first run)
    parser.add_argument("-re", "--recreateEmbeddings", required=False, default=False,
                        help="true to recreate the embeddings")
    parser.add_argument("-chroma", "--useChromaDb", required=False, default=False,
                        help="true to use chromaDB as the embedding storage")

    # Read arguments from command line
    args = parser.parse_args()

    print(f'useLlamaCompletions: {args.useLlamaCompletions}')
    print(f'useOpenAiCompletions: {args.useOpenAiCompletions}')
    print(f'useLlamaEmbeddings: {args.useLlamaEmbeddings}')
    print(f'useOpenAiEmbeddings: {args.useOpenAiEmbeddings}')
    print(f'recreateEmbeddings: {args.recreateEmbeddings}')
    print(f'useChromaDb: {args.useChromaDb}')

    if (args.useLlamaCompletions == "false" and args.useOpenAiCompletions == "false"):
        print("At least one of the completion models should be enabled")
        sys.exit(1)
    if (args.useLlamaEmbeddings == "false" and args.useOpenAiEmbeddings == "false"):
        print("At least one of the embedding models should be enabled")
        sys.exit(2)

    main(use_llama_completions=args.useLlamaCompletions, use_open_ai_completions=args.useOpenAiCompletions,
         use_chroma_embeddings=args.useChromaDb, recreate_embeddings=args.recreateEmbeddings)
