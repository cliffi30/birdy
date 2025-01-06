import argparse
import sys
from enum import Enum

from langchain_openai import ChatOpenAI
from openai import OpenAI
from termcolor import colored

from config import Config
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from embeddings.neo4j_embedding_storage import Neo4jEmbeddingStorage
from embeddings.openai_embeddings import OpenaiEmbeddings
from embeddings.transformer_embeddings import TransformerEmbeddings
from models.completions import Completion
from models.ollama_completions import OllamaCompletions
from models.openai_completions import OpenaiCompletions
from utils.embeddings_helper import build_birds_embeddings
from utils.sentiment_analyzer import SentimentAnalyzer


class EmbeddingType(Enum):
    Llama = "llama"
    OpenAI = "openai"
    Transformer = "transformer"


def main(use_llama_completions: bool = False, use_open_ai_completions: bool = True,
         use_embedding_type: EmbeddingType = EmbeddingType.Transformer,
         use_chroma_embeddings: bool = False, recreate_embeddings: bool = False, do_open_ai_reasoning: bool = True,
         do_qwq_reasoning: bool = False, full_embeddings: bool = False):
    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)
    chat = ChatOpenAI(
        model="gpt-4o",
        max_tokens=1024,
        openai_api_key=config.openai_api_key
    )

    # Prepare our possible embedders
    if use_embedding_type == EmbeddingType.OpenAI:
        embedder = OpenaiEmbeddings(client)
    else:
        embedder = TransformerEmbeddings()

    # Create the possible embedding storage (CSV, ChromaDB and Neo4j)
    # embedding_storage = CSVEmbeddingStorage('embeddings.csv')
    embedding_storage = None
    if use_chroma_embeddings:
        embedding_storage = ChromaDBEmbeddingStorage(embedder=embedder.get_langchain_embedder(),
                                                     collection_name="birds_" + use_embedding_type.value)
    else:
        embedding_storage = Neo4jEmbeddingStorage(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            embedder=embedder.get_langchain_embedder()
        )

    if recreate_embeddings:
        build_birds_embeddings(embedding_storage, include_encyclopedia_embeddings=full_embeddings)

    questions = [
        "Was sind die Charakteristiken von einer Schwarzamsel?",
        "Was sind die Charakteristiken von einer Amsel?",
        "Hi, You told me that the Eastern Bluebird is a small bird, but now the bird is huge! Why are you lying?",
        "Ich bin richtig wütend. Ich habe einen Vogel gekauft und er ist nach wenigen Tagen bereits gestorben. Ich möchte Schadenersatz! Der seelische Schmerz ist unerträglich.",
        "Ich muss schon sagen, die Qualität dieses Vogels mit dem langen roten Schnabel ist wirklich hervorragend. Es ist zwar etwas knapp für ihn im Käfig. Ich bin sehr zufrieden."
    ]

    openai_completions = OpenaiCompletions(client, chat)
    ollama_completions = OllamaCompletions(None)

    completions = openai_completions
    model_text = ""
    if use_open_ai_completions or do_open_ai_reasoning:
        model_text = colored("Model: OpenAI", "white", attrs=["bold"])
        completions = openai_completions
    if use_llama_completions or do_qwq_reasoning:
        model_text = colored("Model: Ollama", "yellow", attrs=["bold"])
        completions = ollama_completions

    sentiment_analyzer = SentimentAnalyzer()
    for question in questions:
        # Calculate the sentiment of the question
        sentiment = sentiment_analyzer.analyze_sentiment(question)

        context = ""
        query_result = embedding_storage.get_embedding(question, 3)
        for result in query_result:
            context = context + result.page_content + "/n"

        print(colored("-------Start question---------\n", "green", attrs=["bold"]))
        response = completions.get_completion(context, [], question, sentiment['label'], sentiment['score'])
        print(model_text)
        print_colored("Sentiment:", sentiment)
        print_colored("Context:", context)
        print_colored("Question:", question)
        print_colored("Response:", response)
        if do_open_ai_reasoning:
            reasoning(openai_completions, context, question, response)
        if do_qwq_reasoning:
            reasoning(ollama_completions, context, question, response)
        print(colored("--------------------\n\n", "green"))
        print(colored("-------End question---------\n", "green", attrs=["bold"]))


def print_colored(header: str, text: str):
    """
    Prints a header and text with the header in green and bold.

    Args:
        header (str): The header text to be printed in green and bold.
        text (str): The main text to be printed below the header.
    """
    print(colored(header, "green", attrs=["bold"]))
    print(f"\n{text}\n")


def reasoning(completions: Completion, context: str, question: str, answer: str):
    """
    Generates reasoning based on the provided context, question, and answer using the specified completions model.
    Args:
        completions (Completions): An instance of a completions model (e.g., OpenaiCompletions, OllamaCompletions).
        context (str): The context or background information for the reasoning.
        question (str): The question to be answered.
        answer (str): The answer to the question.
    Returns:
        None
    Prints:
        The type of the completions model used and the reasoning response from the model.
    """

    response = completions.get_reasoning(context, question, answer)
    header = f"Reasoning response from {completions.get_reasoning_model()}:"
    print_colored(header, response)


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
                        choices=[EmbeddingType.Llama.value, EmbeddingType.OpenAI.value,
                                 EmbeddingType.Transformer.value],
                        help="Select which embedding model to use")

    # needs to be set to true if the embeddings should be recreated (needed for the first run)
    parser.add_argument("-re", "--recreateEmbeddings", required=False, default=False,
                        help="True to recreate the embeddings")
    parser.add_argument("-chroma", "--useChromaDb", required=False, default=False,
                        help="True to use chromaDB as the embedding storage")
    parser.add_argument("-oar", "--doOpenAiReasoning", required=False, default=True,
                        help="True to do reasoning with openAI for the completions")
    parser.add_argument("-qwqr", "--doQwqReasoning", required=False, default=False,
                        help="True to do reasoning with ollama and qwq for the completions")
    parser.add_argument("-full-embeddings", "--fullEmbeddings", required=False, default=False,
                        help="true to create the full embeddings (includes the encyclopedia book; recreate command must be set to true)")

    # Read arguments from command line
    args = parser.parse_args()

    print(f'useLlamaCompletions: {args.useLlamaCompletions}')
    print(f'useOpenAiCompletions: {args.useOpenAiCompletions}')
    print(f'useEmbeddingType: {args.useEmbeddingType}')
    print(f'recreateEmbeddings: {args.recreateEmbeddings}')
    print(f'useChromaDb: {args.useChromaDb}')
    print(f'doOpenAiReasoning: {args.doOpenAiReasoning}')
    print(f'doQwqReasoning: {args.doQwqReasoning}')
    print(f'fullEmbeddings: {args.fullEmbeddings}')

    if (args.useLlamaCompletions == "false" and args.useOpenAiCompletions == "false"):
        print("At least one of the completion models should be enabled")
        sys.exit(1)

    embedding_type = EmbeddingType(args.useEmbeddingType)

    main(use_llama_completions=str_to_bool(args.useLlamaCompletions),
         use_open_ai_completions=str_to_bool(args.useOpenAiCompletions),
         use_embedding_type=embedding_type,
         use_chroma_embeddings=str_to_bool(args.useChromaDb),
         recreate_embeddings=str_to_bool(args.recreateEmbeddings),
         do_open_ai_reasoning=str_to_bool(args.doOpenAiReasoning),
         do_qwq_reasoning=str_to_bool(args.doQwqReasoning),
         full_embeddings=str_to_bool(args.fullEmbeddings))
