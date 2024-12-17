from openai import OpenAI

from src.models.completions import Completion


class OpenaiCompletions(Completion):
    def __init__(self, client: OpenAI, model: str = "gpt-4o", max_tokens= 1000, temperature=0.5):
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
                - Be precise and factual'''

    def get_completion(self, prompt: str) -> str:
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