from openai import OpenAI

from Main_variante import chroma_embedding_storage_name
from config import Config
from embeddings.chromadb_embedding_storage import ChromaDBEmbeddingStorage
from models.openai_completions import OpenaiCompletions

# initialize the config
config = Config()

# create a client
client = OpenAI(api_key=config.openai_api_key)

 # Create the embedding storage (CSV or ChromaDB)
dbClient = ChromaDBEmbeddingStorage(chroma_embedding_storage_name)

question = '''
        Ich habe gestern bei uns vor dem Haus eine Storch gesehen, zumindestens glaube ich das.
        Der Vogel war ca. 12 cm gross und Blau und Gelb.
        War es wirklich ein Storch?
        
        Ihr Hans Muster Meier aus Bern
'''

query_result = dbClient.query_vectorstore(question)

openai_completions = OpenaiCompletions(client)
# Generate the response using the most relevant context

context = ""
for result in query_result:
    context = context + result.page_content + "/n"

response = openai_completions.get_completion(context, question)
print(f"Question: {question}")
print(f"Response: {response}")
print("--------------------")