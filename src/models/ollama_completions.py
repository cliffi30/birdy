'''This module is responsible for generating completions using the Ollama API
   see https://github.com/ollama/ollama/tree/main/docs for more information
'''
from ollama import chat
from ollama import ChatResponse

from models.completions import Completion


class OllamaCompletions(Completion):
    def __init__(self, client: None, model: str = "llama3.2:3b-instruct-q8_0", max_tokens= 1000, temperature=0.5):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_prompt = '''You are a sales assistant in a birds shop. Follow these guidelines:
                - Respond like in a mail conversation
                - Respond in the same language as the user's query
                - Be polite and helpful
                - Base your answers only on the provided content
                - Cite sources when possible using (Source: filename, Page: X)
                - If information isn't in the content, acknowledge this
                - If the promt includes a question to specify a bird use the name of the birds from the context
                - Be precise and factual
                - Answer in German'''

    

    def get_completion(self, context: str, prompt: str) -> str:
        response: ChatResponse = chat(model=self.model, messages=[
        {
            'role': 'system',
            'content': self.system_prompt + " \n context: bird: Internal knowledge base, page 102:" + context + " email user asking: Sandra Vogellisi  email responder: Rolf Vogelsang",
        },
        {
            'role': 'user',
            'content': prompt,
        }
        ])
        '''
        response: ChatResponse = chat(model=self.model, messages=[
        {
            'role': 'user',
            'content':f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.system_prompt} context: bird: Internal knowledge base, page 102: {context} email user asking: Sandra Vogellisi  email responder: Rolf Vogelsang<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        }
        ])
        '''
        return response.message.content
