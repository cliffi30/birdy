from abc import ABC, abstractmethod
from typing import Tuple

from PIL.ImageFile import ImageFile


class Completion(ABC):
    @abstractmethod
    def get_completion(self, context: str, relevant_images: [dict], question: str, sentiment_label: str = None, sentiment_score: float = None) -> Tuple[str, ImageFile]:
        """
        Generates a text completion for the given prompt.

        Args:
            self (Completion): The completion instance.
            context (str): The context to generate a completion for.
            relevant_images (List[dict]): The relevant images to include in the completion.
            question (str): The question to generate a completion for.
            sentiment_label (str): The sentiment label for the prompt.
            sentiment_score (float): The sentiment score for the prompt.

        Returns:
            Tuple[str, ImageFile]: The generated text completion including relevant images.
        """
        pass


    @abstractmethod
    def get_reasoning(self, context: str, question: str, answer: str) -> str:
        """
        Generates reasoning for a given context, question, and answer.

        Args:
            self (Completion): The completion instance.
            context (str): The context or background information related to the question.
            question (str): The question that needs reasoning.
            answer (str): The answer to the question for which reasoning is to be generated.

        Returns:
            str: The generated reasoning.
        """
        pass

    @abstractmethod
    def get_reasoning_model(self) -> str:
        """
        Retrieves the reasoning model.
        Returns:
            str: The name or identifier of the reasoning model.
        """
        
        pass