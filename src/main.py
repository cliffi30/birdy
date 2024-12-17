from openai import OpenAI

from src.config import Config
from src.data.birds_csv_loader import load_birds_csv
from src.embeddings.openai_embeddings import OpenaiEmbeddings


def main():
    # initialize the config
    config = Config()

    # create a client
    client = OpenAI(api_key=config.openai_api_key)

    # load the birds dataset
    birds = load_birds_csv()

    # create the embeddings
    openai_embeddings = OpenaiEmbeddings(client)
    embeddings = birds.apply(lambda x: openai_embeddings.get_embedding(x))
    print(embeddings.head())

    print(birds.head())
    questions = [
        'What are the characteristics of the Eastern Bluebird?',
    ]


if __name__ == "__main__":
    main()