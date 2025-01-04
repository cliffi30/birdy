import argparse
import sys

from enum import Enum
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
from termcolor import colored


class EmbeddingType(Enum):
    Llama = "llama"
    OpenAI = "openai"
    Transformer = "transformer"


def main(use_llama_completions: bool = False, use_open_ai_completions: bool = True, use_embedding_type:EmbeddingType = EmbeddingType.Transformer,
         use_chroma_embeddings: bool = False, recreate_embeddings: bool = False, do_reasoning: bool = True):

    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)

    # Prepare our possible embedders
    if use_embedding_type == EmbeddingType.OpenAI:
        embedder = OpenaiEmbeddings(client)
    elif use_embedding_type == EmbeddingType.Transformer:
        embedder = TransformerEmbeddings()

    # Create the possible embedding storage (CSV, ChromaDB and Neo4j)
    # embedding_storage = CSVEmbeddingStorage('embeddings.csv')
    embedding_storage = None
    if use_chroma_embeddings:
        embedding_storage = ChromaDBEmbeddingStorage(embedder=embedder.get_langchain_embedder(), collection_name="birds_"+use_embedding_type.value)
    else:
        embedding_storage = Neo4jEmbeddingStorage(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            embedder=embedder.get_langchain_embedder()
        )

    if recreate_embeddings:
        build_birds_embeddings(embedding_storage)

    questions = [
        "Was sind die Charakteristiken von einer Schwarzamsel?",
        "Was sind die Charakteristiken von einer Amsel?",
        "Hi, You told me that the Eastern Bluebird is a small bird, but now the bird is huge! Why are you lying?",
        "Ich bin richtig wütend. Ich habe einen Vogel gekauft und er ist nach wenigen Tagen bereits gestorben. Ich möchte Schadenersatz! Der seelische Schmerz ist unerträglich.",
        "Ich muss schon sagen, die Qualität dieses Vogels mit dem langen roten Schnabel ist wirklich hervorragend. Es ist zwar etwas knapp für ihn im Käfig. Ich bin sehr zufrieden."
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
        query_result = embedding_storage.get_embedding(question, 3)
        for result in query_result:
            context = context + result.page_content + "/n"

        response = ""

        print(colored("-------Start question---------\n", "green", attrs=["bold"]))
        if use_open_ai_completions:
            response = openai_completions.get_completion(context, question, sentiment['label'], sentiment['score'])
            print(colored("--------------------\n", "green"))
            print(colored("Model: OpenAI", "white", attrs=["bold"]))
            print_colored("Sentiment:", sentiment)
            print_colored("Context:", context)
            print_colored("Question:", question)
            print_colored("Response:", response)
            if do_reasoning:
                reasoning(context, question, response)
            print(colored("--------------------\n\n", "green"))
        if use_llama_completions:
            response = ollama_completions.get_completion(context, question)
            print(colored("--------------------\n", "green"))
            print(colored("Model: Ollama", "yellow", attrs=["bold"]))
            print_colored("Sentiment:", sentiment)
            print_colored("Context:", context)
            print_colored("Question:", question)
            print_colored("Response:", response)
            if do_reasoning:
                reasoning(context, question, response)
            print(colored("--------------------\n\n", "green"))
        print(colored("-------End question---------\n", "green", attrs=["bold"]))


def print_colored(header: str, text: str):
    print(colored(header, "green", attrs=["bold"]))
    print(f"\n{text}\n")

def reasoning(context: str, question: str, answer: str):
    """
    Generates reasoning for a given context, question, and answer using OpenAI's API.
    Args:
        context (str): The context or background information related to the question.
        question (str): The question that needs reasoning.
        answer (str): The answer to the question for which reasoning is to be generated.
    Returns:
        None
    """

    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)

    openai_completions = OpenaiCompletions(client)
    response = openai_completions.get_reasoning(context, question, answer)
    print(f"Reasoning response:\n{response}")


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    return s.lower() in ['true', '1', 't', 'yes', 'y']

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Birdy - A bird shop assistant")

    # Adding optional argument
    parser.add_argument("-lc", "--useLlamaCompletions", required=False, default=False,
                        help="True to use llama with ollama to create completions")
    parser.add_argument("-oac", "--useOpenAiCompletions", required=False, default=True,
                        help="True to use OpenAI to create completions")
    
    # Embedding arguments
    parser.add_argument("-uet", "--useEmbeddingType", required=False, default=EmbeddingType.Transformer,
                        choices=[EmbeddingType.Llama.value, EmbeddingType.OpenAI.value, EmbeddingType.Transformer.value],
                        help="Select which embedding model to use")

    # needs to be set to true if the embeddings should be recreated (needed for the first run)
    parser.add_argument("-re", "--recreateEmbeddings", required=False, default=False,
                        help="True to recreate the embeddings")
    parser.add_argument("-chroma", "--useChromaDb", required=False, default=False,
                        help="True to use chromaDB as the embedding storage")
    parser.add_argument("-rea", "--doReasoning", required=False, default=True,
                        help="True to do reasoning for the completions")

    # Read arguments from command line
    args = parser.parse_args()

    print(f'useLlamaCompletions: {args.useLlamaCompletions}')
    print(f'useOpenAiCompletions: {args.useOpenAiCompletions}')
    print(f'useEmbeddingType: {args.useEmbeddingType}')
    print(f'recreateEmbeddings: {args.recreateEmbeddings}')
    print(f'useChromaDb: {args.useChromaDb}')
    print(f'doReasoning: {args.doReasoning}')

    if (args.useLlamaCompletions == "false" and args.useOpenAiCompletions == "false"):
        print("At least one of the completion models should be enabled")
        sys.exit(1)

    embedding_type = EmbeddingType(args.useEmbeddingType)

    main(use_llama_completions=str_to_bool(args.useLlamaCompletions), 
         use_open_ai_completions=str_to_bool(args.useOpenAiCompletions),
         use_embedding_type=embedding_type,
         use_chroma_embeddings=str_to_bool(args.useChromaDb), 
         recreate_embeddings=str_to_bool(args.recreateEmbeddings),
         do_reasoning=str_to_bool(args.doReasoning))
