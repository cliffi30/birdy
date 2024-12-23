from abc import ABC, abstractmethod

class Completion(ABC):
    @abstractmethod
    def get_completion(self, context: str, question: str, sentiment_label: str = None, sentiment_score: float = None) -> str:
        """
        Generates a text completion for the given prompt.

        Args:
            self (Completion): The completion instance.
            context (str): The context to generate a completion for.
            question (str): The question to generate a completion for.
            sentiment_label (str): The sentiment label for the prompt.
            sentiment_score (float): The sentiment score for the prompt.

        Returns:
            str: The generated text completion.
        """
        pass
