from openai import OpenAI
from models.completions import Completion


class OpenaiCompletions(Completion):
    def __init__(self, client: OpenAI, model: str = "gpt-4o", max_tokens: int = 1000, temperature:float = 0.5):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_prompt = '''You are a sales assistant with the name Horst Adler in a birds shop. Follow these guidelines:
                - Respond like in a mail conversation
                - Respond in the same language as the user's query
                - Be polite and helpful
                - Base your answers only on the provided content
                - Cite sources when possible using (Source: filename, Page: X)
                - If information isn't in the content, acknowledge this
                - If the prompt includes a question to specify a bird use the name of the birds from the context
                - Be precise and factual'''

    def get_completion(self, context: str, question: str, sentiment_label: str = None, sentiment_score: float = None) -> str:
        prompt = f"Context:{context}\n\nQuestion: {question}\nAnswer:"
        if sentiment_label is not None and sentiment_score is not None:
            # provide the sentiment information to the model
            prompt = f"The user's sentiment is '{sentiment_label}' with a confidence score of {sentiment_score}.\n{prompt}"
        completions = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
        return completions.choices[0].message.content

    def get_reasoning(self, context: str, question: str, answer: str) -> str:
        prompt = f"Check if the answer is correct for the given context and question\n\nContext:{context}\n\nQuestion: {question}\nAnswer: {answer}"
        #prompt = f"Context:{context}\n\nQuestion: {question}\nAnswer: {answer}"
        completions = self.client.chat.completions.create(
                model="o1-mini",
                max_completion_tokens=10000,
                messages=[
                    #{"role": "developer", "content": developer_prompt}
                    {"role": "user", "content": prompt}
                ]
            )
        return completions.choices[0].message.content
