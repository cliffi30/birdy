import base64
from typing import io, Tuple

from PIL import Image
from PIL.ImageFile import ImageFile
from langchain_community.chat_models import ChatOpenAI
from models.completions import Completion
from openai import OpenAI


class OpenaiCompletions(Completion):
    def __init__(self, client: OpenAI, chat: ChatOpenAI, model: str = "gpt-4o", max_tokens: int = 1000, temperature: float =0.5):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = "o1-mini"

        self.system_prompt = '''You are a sales assistant with the name Horst Adler in a birds shop. Follow these guidelines:
                - Respond like in a mail conversation
                - Respond in the same language as the user's query
                - Be polite and helpful
                - Base your answers only on the provided content
                - Cite sources when possible using (Source: filename, Page: X)
                - If information isn't in the content, acknowledge this
                - If the prompt includes a question to specify a bird use the name of the birds from the context
                - Be precise and factual'''

    def get_completion(self, context: str, relevant_images: [dict], question: str, sentiment_label: str = None,
                       sentiment_score: float = None) -> Tuple[str, ImageFile]:
        prompt_content = []
        # Add relevant images to prompt
        pil_images = []
        for img in relevant_images:
            try:
                print(img)
                # Convert bytes to PIL
                img = Image.open(io.BytesIO(img["image_data"]))
                pil_images.append(img)

                # Add base64 image to prompt
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                prompt_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        prompt = f"Context:{context}\n\nQuestion: {question}\nAnswer:"
        if sentiment_label is not None and sentiment_score is not None:
            # provide the sentiment information to the model
            prompt = f"The user's sentiment is '{sentiment_label}' with a confidence score of {sentiment_score}.\n{prompt}"

        prompt_content.append({
            "type": "text",
            "text": prompt
        })
        completions = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_content}
            ]
        )
        return completions.choices[0].message.content, pil_images

    def get_reasoning(self, context: str, question: str, answer: str) -> str:
        prompt = f"Check if the answer is correct for the given context and question\n\nContext:{context}\n\nQuestion: {question}\nAnswer: {answer}"
        #prompt = f"Context:{context}\n\nQuestion: {question}\nAnswer: {answer}"
        completions = self.client.chat.completions.create(
                model=self.reasoning_model,
                max_completion_tokens=10000,
                messages=[
                    #{"role": "developer", "content": developer_prompt}
                    {"role": "user", "content": prompt}
                ]
            )
        return completions.choices[0].message.content


    def get_reasoning_model(self) -> str:
        return f"OpenAI: {self.reasoning_model}"